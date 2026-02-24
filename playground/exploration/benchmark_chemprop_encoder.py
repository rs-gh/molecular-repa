"""
Benchmark script: ChemPropEncoder forward-pass bottleneck analysis.

Runs on CPU or GPU (auto-detected). Designed to be run interactively on an
HPC A100 node via:

    srun --partition=ampere --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash
    source /home/sr2173/git/molecular-repa/.venv/bin/activate
    cd /home/sr2173/git/molecular-repa
    python playground/exploration/benchmark_chemprop_encoder.py

Or as a quick CPU smoke-test locally (GPU timings will say N/A):

    source .venv/bin/activate
    python playground/exploration/benchmark_chemprop_encoder.py --cpu-only

Sections
--------
1. Per-molecule: from_tensor (bond inference) vs MolFromSmiles
2. Full encoder forward – sub-stage wall-clock breakdown
3. GPU idle fraction (CUDA events, GPU only)
4. MolGraph cache memory estimate
5. Projected epoch time under each scenario
"""

import sys
import time
import statistics
import pickle
import argparse
from pathlib import Path

import torch
import lmdb

# ── project paths ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src" / "tabasco" / "src"
sys.path.insert(0, str(SRC_PATH))

LMDB_DIR = REPO_ROOT / "src" / "tabasco" / "data" / "lmdb_geom"
CHEMELEON = Path.home() / ".chemprop" / "chemeleon_mp.pt"


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--n-warmup", type=int, default=2)
parser.add_argument("--n-repeats", type=int, default=5)
parser.add_argument("--n-mol-reps", type=int, default=30)
parser.add_argument("--cpu-only", action="store_true")
args, _ = parser.parse_known_args()

BATCH_SIZE = args.batch_size
N_WARMUP = args.n_warmup
N_REPEATS = args.n_repeats
N_MOL_REPS = args.n_mol_reps
DEVICE = torch.device(
    "cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda"
)
HAS_CUDA = DEVICE.type == "cuda"


# ── helpers ───────────────────────────────────────────────────────────────────


def banner(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def mean_std(times):
    m = statistics.mean(times)
    s = statistics.stdev(times) if len(times) > 1 else 0.0
    return m, s


def sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def stopwatch(fn, n_warmup, n_rep):
    """Run fn() n_warmup + n_rep times; return list of n_rep wall-clock ms."""
    times = []
    for i in range(n_warmup + n_rep):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        if i >= n_warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return times


# ── load a real batch directly from GEOM LMDB (no YAML parse) ────────────────


def load_geom_batch_fast(batch_size: int):
    """
    Read `batch_size` entries directly from the GEOM training LMDB and
    convert them to a batched TensorDict + parallel list of SMILES.
    Bypasses the UnconditionalLMDBDataset to avoid loading the 40-second YAML.
    """
    from rdkit import Chem
    from tabasco.chem.convert import MoleculeConverter
    from tabasco.data.utils import TensorDictCollator

    print(f"\nLoading {batch_size} molecules from GEOM LMDB (direct read) …")
    t0 = time.perf_counter()

    db = lmdb.open(
        str(LMDB_DIR / "train.lmdb"),
        map_size=10 * (1024**3),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    converter = MoleculeConverter()
    samples, smiles_list = [], []
    with db.begin() as txn:
        cursor = txn.cursor()
        for i, val in enumerate(cursor.iternext(keys=False, values=True)):
            if i >= batch_size * 2:  # read extra in case some fail
                break
            data = pickle.loads(val)
            mol = data["molecule"]
            try:
                td = converter.to_tensor(
                    mol=mol,
                    pad_to_size=71,  # max_num_atoms
                    remove_hydrogens=True,
                )
                smi = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
                samples.append(td)
                smiles_list.append(smi)
            except Exception:
                pass
            if len(samples) >= batch_size:
                break
    db.close()

    batch = TensorDictCollator()(samples[:batch_size]).to(DEVICE)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Loaded {len(samples)} mols in {elapsed:.0f} ms")
    print(f"  coords shape: {tuple(batch['coords'].shape)}  device: {DEVICE}")
    return batch, smiles_list[:batch_size]


# ── 1. per-molecule micro-benchmark ───────────────────────────────────────────


def bench_per_molecule(batch, smiles_list):
    banner("1. Per-molecule cost: from_tensor (bond inference) vs MolFromSmiles")

    from tensordict import TensorDict
    from tabasco.chem.convert import MoleculeConverter
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem

    converter = MoleculeConverter()
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    # pick molecule 0 (non-padded)
    i = 0
    smi = smiles_list[i]
    n_real = (~batch["padding_mask"][i]).sum().item()
    print(f"\n  Probing molecule 0: {n_real} heavy atoms")
    print(f"  SMILES: {smi[:80]}{'…' if len(smi)>80 else ''}")

    mol_td_obj = TensorDict(
        {
            "coords": batch["coords"][i].cpu(),
            "atomics": batch["atomics"][i].cpu(),
            "padding_mask": batch["padding_mask"][i].cpu(),
        }
    )

    # ── from_tensor ───────────────────────────────────────────────────────────
    times_infer = stopwatch(
        lambda: converter.from_tensor(
            mol_td_obj, rescale_coords=True, sanitize=False, use_openbabel=False
        ),
        N_WARMUP,
        N_MOL_REPS,
    )
    m_infer, s_infer = mean_std(times_infer)
    mol_infer = converter.from_tensor(
        mol_td_obj, rescale_coords=True, sanitize=False, use_openbabel=False
    )
    print("\n  from_tensor  (DetermineConnectivity + argmax + rescale):")
    print(f"    {m_infer:.2f} ± {s_infer:.2f} ms  (n={N_MOL_REPS})")
    if mol_infer:
        print(f"    → reconstructed {mol_infer.GetNumAtoms()} atoms")

    # ── MolFromSmiles ─────────────────────────────────────────────────────────
    times_smi = stopwatch(lambda: Chem.MolFromSmiles(smi), N_WARMUP, N_MOL_REPS)
    m_smi, s_smi = mean_std(times_smi)
    mol_smi = Chem.MolFromSmiles(smi)
    print("\n  MolFromSmiles (canonical SMILES → RDKit mol):")
    print(f"    {m_smi:.3f} ± {s_smi:.3f} ms  (n={N_MOL_REPS})")
    if mol_smi:
        print(f"    → {mol_smi.GetNumAtoms()} atoms")

    speedup_mol = m_infer / m_smi if m_smi > 0 else float("inf")
    print(f"\n  >> Speedup from_tensor → MolFromSmiles:  {speedup_mol:.0f}×")

    # ── featurizer ────────────────────────────────────────────────────────────
    times_feat = stopwatch(lambda: featurizer(mol_smi), N_WARMUP, N_MOL_REPS)
    m_feat, s_feat = mean_std(times_feat)
    print("\n  SimpleMoleculeMolGraphFeaturizer (RDKit mol → MolGraph):")
    print(f"    {m_feat:.2f} ± {s_feat:.2f} ms  (n={N_MOL_REPS})")

    # ── cache dict lookup ─────────────────────────────────────────────────────
    mg_cached = featurizer(mol_smi)
    cache = {smi: mg_cached}
    times_lookup = stopwatch(lambda: cache.get(smi), N_WARMUP, N_MOL_REPS)
    m_lookup, s_lookup = mean_std(times_lookup)
    print("\n  Dict cache lookup (already cached):")
    print(f"    {m_lookup*1000:.1f} ± {s_lookup*1000:.1f} µs  (n={N_MOL_REPS})")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'Approach':<42}  {'per mol':>8}  {'×{} batch'.format(BATCH_SIZE):>12}")
    print(f"  {'-'*42}  {'-'*8}  {'-'*12}")
    rows = [
        (
            "Current:  from_tensor  + featurize",
            m_infer + m_feat,
            (m_infer + m_feat) * BATCH_SIZE,
        ),
        (
            "Proposed: MolFromSmiles + featurize",
            m_smi + m_feat,
            (m_smi + m_feat) * BATCH_SIZE,
        ),
        ("Cached:   dict lookup  (after 1ep)", m_lookup, m_lookup * BATCH_SIZE),
    ]
    for label, per_mol, per_batch in rows:
        print(f"  {label:<42}  {per_mol:>7.2f}ms  {per_batch:>10.0f}ms")

    return m_infer, m_smi, m_feat, m_lookup


# ── 2. full encoder forward – sub-stage breakdown ─────────────────────────────

_CACHE = {}  # global MolGraph cache for approach C


def bench_encoder_forward(batch, smiles_list):
    banner("2. ChemPropEncoder.forward() – sub-stage wall-clock breakdown")

    from tensordict import TensorDict
    from tabasco.chem.convert import MoleculeConverter
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.data import BatchMolGraph
    from chemprop.nn import BondMessagePassing
    from rdkit import Chem

    # load weights
    if CHEMELEON.exists():
        weights = torch.load(CHEMELEON, map_location="cpu", weights_only=True)
        mp = BondMessagePassing(**weights["hyper_parameters"])
        mp.load_state_dict(weights["state_dict"])
        encoder_dim = weights["hyper_parameters"]["d_h"]
        print(f"\n  CheMeleon loaded ({encoder_dim}-dim)")
    else:
        print(f"\n  [WARN] CheMeleon not found at {CHEMELEON}  →  random weights")
        mp = BondMessagePassing(d_h=2048, depth=3, dropout=0.0)
        encoder_dim = 2048
    mp = mp.to(DEVICE).eval()

    converter = MoleculeConverter()
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    B = batch["coords"].shape[0]
    N = batch["coords"].shape[1]

    coords_cpu = batch["coords"].cpu()
    atomics_cpu = batch["atomics"].cpu()
    padmask_cpu = batch["padding_mask"].cpu()

    def run_stage(approach: str):
        """Run encoder forward once and return per-stage ms dict."""
        ms = {}

        # ── Stage A: Python loop (mol construction + featurize) ───────────────
        molgraphs, atom_counts = [], []
        sync()
        tA0 = time.perf_counter()

        if approach == "from_tensor":
            for i in range(B):
                mol_td = TensorDict(
                    {
                        "coords": coords_cpu[i],
                        "atomics": atomics_cpu[i],
                        "padding_mask": padmask_cpu[i],
                    }
                )
                try:
                    mol = converter.from_tensor(
                        mol_td, rescale_coords=True, sanitize=False, use_openbabel=False
                    )
                    if mol is not None:
                        molgraphs.append(featurizer(mol))
                        atom_counts.append(mol.GetNumAtoms())
                    else:
                        molgraphs.append(None)
                        atom_counts.append(0)
                except Exception:
                    molgraphs.append(None)
                    atom_counts.append(0)

        elif approach == "mol_from_smiles":
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smi) if smi else None
                    if mol is not None:
                        molgraphs.append(featurizer(mol))
                        atom_counts.append(mol.GetNumAtoms())
                    else:
                        molgraphs.append(None)
                        atom_counts.append(0)
                except Exception:
                    molgraphs.append(None)
                    atom_counts.append(0)

        elif approach == "cached":
            for smi in smiles_list:
                mg = _CACHE.get(smi)
                if mg is not None:
                    molgraphs.append(mg)
                    atom_counts.append(mg.V.shape[0])
                else:
                    molgraphs.append(None)
                    atom_counts.append(0)

        tA1 = time.perf_counter()
        ms["A_loop_ms"] = (tA1 - tA0) * 1000

        # ── Stage B: BatchMolGraph ─────────────────────────────────────────────
        valid_mgs = [mg for mg in molgraphs if mg is not None]
        tB0 = time.perf_counter()
        bmg = BatchMolGraph(valid_mgs)
        tB1 = time.perf_counter()
        ms["B_batch_ms"] = (tB1 - tB0) * 1000

        # ── Stage C: CPU → GPU transfer ───────────────────────────────────────
        tC0 = time.perf_counter()
        bmg.to(DEVICE)
        sync()
        tC1 = time.perf_counter()
        ms["C_transfer_ms"] = (tC1 - tC0) * 1000

        # ── Stage D: GNN forward ──────────────────────────────────────────────
        sync()
        tD0 = time.perf_counter()
        with torch.no_grad():
            atom_embs = mp(bmg)
        sync()
        tD1 = time.perf_counter()
        ms["D_gnn_ms"] = (tD1 - tD0) * 1000

        # ── Stage E: scatter to padded output ─────────────────────────────────
        output = torch.zeros(B, N, encoder_dim, device=DEVICE)
        vi, off = 0, 0
        tE0 = time.perf_counter()
        for i in range(B):
            if molgraphs[i] is not None:
                na = atom_counts[vi]
                output[i, :na] = atom_embs[off : off + na]
                off += na
                vi += 1
        sync()
        tE1 = time.perf_counter()
        ms["E_scatter_ms"] = (tE1 - tE0) * 1000
        ms["total_ms"] = sum(ms.values())
        return ms

    def run_approach(label, approach):
        print(f"\n  {label}  [{N_WARMUP} warmup + {N_REPEATS} timed]")
        accum = {
            k: []
            for k in (
                "A_loop_ms",
                "B_batch_ms",
                "C_transfer_ms",
                "D_gnn_ms",
                "E_scatter_ms",
                "total_ms",
            )
        }
        for rep in range(N_WARMUP + N_REPEATS):
            ms = run_stage(approach)
            if rep >= N_WARMUP:
                for k in accum:
                    accum[k].append(ms[k])
            sys.stdout.write(
                f"    rep {rep+1}/{N_WARMUP+N_REPEATS}:  "
                f"total={ms['total_ms']:.0f}ms  "
                f"loop={ms['A_loop_ms']:.0f}ms  "
                f"gnn={ms['D_gnn_ms']:.0f}ms\n"
            )
            sys.stdout.flush()
        return accum

    accum_ft = run_approach("Approach A – from_tensor (current)", "from_tensor")

    accum_smi = run_approach("Approach B – MolFromSmiles (proposed)", "mol_from_smiles")

    # pre-warm the cache
    print(f"\n  Pre-warming MolGraph cache for {B} molecules …")
    featurizer2 = SimpleMoleculeMolGraphFeaturizer()
    global _CACHE
    _CACHE = {}
    for smi in smiles_list:
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                _CACHE[smi] = featurizer2(mol)

    accum_cache = run_approach("Approach C – full cache", "cached")

    # ── comparison table ──────────────────────────────────────────────────────
    def fmt(ts):
        m, s = mean_std(ts)
        return f"{m:7.1f} ±{s:5.1f}"

    labels = {
        "A_loop_ms": "[A] mol construction + featurize (Python loop)",
        "B_batch_ms": "[B] BatchMolGraph()                            ",
        "C_transfer_ms": "[C] .to(device) CPU→GPU                       ",
        "D_gnn_ms": "[D] BondMessagePassing (GNN)                  ",
        "E_scatter_ms": "[E] scatter to padded output                  ",
        "total_ms": "[T] TOTAL                                     ",
    }
    print(f"\n  {'Stage':<52}  {'from_tensor':>14}  {'MolFromSMI':>14}  {'Cached':>14}")
    print(f"  {'-'*52}  {'-'*14}  {'-'*14}  {'-'*14}")
    for k, lbl in labels.items():
        sep = "─" * 70 if k == "total_ms" else ""
        if sep:
            print(f"  {sep}")
        print(
            f"  {lbl}  {fmt(accum_ft[k]):>14}  "
            f"{fmt(accum_smi[k]):>14}  {fmt(accum_cache[k]):>14}  ms"
        )

    m_ft_total = mean_std(accum_ft["total_ms"])[0]
    m_smi_total = mean_std(accum_smi["total_ms"])[0]
    m_cache_total = mean_std(accum_cache["total_ms"])[0]
    m_gnn = mean_std(accum_ft["D_gnn_ms"])[0]
    gpu_idle_pct = 100.0 * (1.0 - m_gnn / m_ft_total) if m_ft_total > 0 else 0.0

    print(
        f"\n  GPU idle fraction (current): {gpu_idle_pct:.1f}%  "
        f"(total={m_ft_total:.0f}ms, GPU={m_gnn:.0f}ms)"
    )
    print(f"  Speedup MolFromSmiles vs current:  {m_ft_total/m_smi_total:.1f}×")
    print(f"  Speedup Full cache vs current:     {m_ft_total/m_cache_total:.1f}×")

    return m_ft_total, m_smi_total, m_cache_total, m_gnn


# ── 3. MolGraph cache memory estimate ─────────────────────────────────────────


def bench_cache_memory(smiles_list):
    banner("3. MolGraph cache memory estimate")

    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem

    featurizer = SimpleMoleculeMolGraphFeaturizer()
    sizes_bytes = []

    sample = [s for s in smiles_list if s][: min(len(smiles_list), 100)]
    for smi in sample:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mg = featurizer(mol)
        size = 0
        for attr in ["V", "E", "edge_index"]:
            val = getattr(mg, attr, None)
            if val is None:
                continue
            size += val.nbytes if hasattr(val, "nbytes") else val.numel() * 4
        sizes_bytes.append(size)

    if not sizes_bytes:
        print("  No valid molecules")
        return

    m_bytes = statistics.mean(sizes_bytes)
    std_bytes = statistics.stdev(sizes_bytes) if len(sizes_bytes) > 1 else 0.0
    n_train = 1_142_099

    print(f"\n  Sample: {len(sizes_bytes)} molecules")
    print(f"  Mean MolGraph size: {m_bytes/1024:.1f} ± {std_bytes/1024:.1f} KB")
    print(f"\n  Projected RAM for full GEOM training set ({n_train:,} mols):")
    for frac, label in [
        (1.00, "100% – full set      "),
        (0.10, " 10% – LRU ~114K cap "),
        (0.02, "  2% – LRU ~23K cap  "),
    ]:
        gb = m_bytes * n_train * frac / 1e9
        print(f"    {label}: {gb:.1f} GB")


# ── 4. projected epoch time ───────────────────────────────────────────────────


def project_epoch_times(m_ft_total, m_smi_total, m_cache_total, m_gnn_ms):
    banner("4. Projected epoch time  (GEOM, B=256, ~4579 steps/epoch)")

    N_STEPS = 4579

    def h(ms):
        total_ms = ms * N_STEPS
        hrs = total_ms / 3_600_000
        return f"{ms:6.0f} ms/step → {hrs:.2f} h/epoch"

    print(f"""
  Encoder forward only (this benchmark):
    Current   (from_tensor):    {h(m_ft_total)}
    Proposed  (MolFromSmiles):  {h(m_smi_total)}
    Full cache (after 1 epoch): {h(m_cache_total)}
    GNN alone (ideal min):      {h(m_gnn_ms)}

  For context:
    Mild (no encoder, just flow model): ~26 min/epoch observed
    Chemprop currently reports:         ~7+ h/epoch observed

  The gap is entirely explained by encoder CPU overhead.
  Once loop time < ~350 ms/step, total step ≈ mild step.
""")


# ── 5. optional torch profiler ────────────────────────────────────────────────


def run_torch_profiler(batch, smiles_list):
    banner("5. PyTorch profiler (top ops, CPU time)")

    if not HAS_CUDA:
        print("  Skipping: requires CUDA for meaningful CPU/GPU op breakdown.")
        return

    from tensordict import TensorDict
    from tabasco.chem.convert import MoleculeConverter
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.data import BatchMolGraph
    from chemprop.nn import BondMessagePassing

    if not CHEMELEON.exists():
        print("  Skipping: CheMeleon weights not found.")
        return

    weights = torch.load(CHEMELEON, map_location="cpu", weights_only=True)
    mp = BondMessagePassing(**weights["hyper_parameters"])
    mp.load_state_dict(weights["state_dict"])
    mp = mp.to(DEVICE).eval()

    converter = MoleculeConverter()
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    B = batch["coords"].shape[0]
    coords_cpu = batch["coords"].cpu()
    atomics_cpu = batch["atomics"].cpu()
    padmask_cpu = batch["padding_mask"].cpu()

    def one_forward():
        mgs = []
        for i in range(B):
            mol_td = TensorDict(
                {
                    "coords": coords_cpu[i],
                    "atomics": atomics_cpu[i],
                    "padding_mask": padmask_cpu[i],
                }
            )
            mol = converter.from_tensor(
                mol_td, rescale_coords=True, sanitize=False, use_openbabel=False
            )
            if mol:
                mgs.append(featurizer(mol))
        bmg = BatchMolGraph(mgs)
        bmg.to(DEVICE)
        with torch.no_grad():
            _ = mp(bmg)
        torch.cuda.synchronize()

    for _ in range(2):
        one_forward()  # warmup

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        one_forward()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "━" * 70)
    print("  ChemPropEncoder Bottleneck Benchmark")
    print(f"  Device: {DEVICE}  |  Batch size: {BATCH_SIZE}")
    print(f"  N_warmup={N_WARMUP}  N_repeats={N_REPEATS}  N_mol_reps={N_MOL_REPS}")
    print("━" * 70)

    if not (LMDB_DIR / "train.lmdb").exists():
        print(f"\n[ERROR] GEOM LMDB not found at {LMDB_DIR}")
        sys.exit(1)

    batch, smiles = load_geom_batch_fast(BATCH_SIZE)

    m_infer, m_smi, m_feat, m_lookup = bench_per_molecule(batch, smiles)
    m_ft, m_smi_fwd, m_cache, m_gnn = bench_encoder_forward(batch, smiles)
    bench_cache_memory(smiles)
    project_epoch_times(m_ft, m_smi_fwd, m_cache, m_gnn)

    if HAS_CUDA:
        run_torch_profiler(batch, smiles)


if __name__ == "__main__":
    main()
