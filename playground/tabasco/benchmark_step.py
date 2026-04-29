"""
Benchmark: wall-clock time per training step for MACEEncoder vs ChemPropEncoder.

Loads a real GEOM LMDB batch (B=256), constructs a FlowMatchingModel with
each encoder, and measures (a) full training step, (b) encoder-only forward,
(c) model forward minus encoder.  Prints a comparison table.

Usage (GPU):
    srun --partition=ampere --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash
    source /home/sr2173/git/molecular-repa/.venv/bin/activate
    cd /home/sr2173/git/molecular-repa
    python playground/tabasco/benchmark_step.py

Usage (CPU, quick smoke-test):
    python playground/tabasco/benchmark_step.py --cpu-only --n-warmup 1 --n-rep 3
"""

import argparse
import pickle
import statistics
import sys
import time
from pathlib import Path

import torch
import lmdb

# -- project paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src" / "tabasco" / "src"
sys.path.insert(0, str(SRC_PATH))

LMDB_DIR = REPO_ROOT / "src" / "tabasco" / "data" / "lmdb_geom"

# -- CLI -----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Benchmark training step: MACEEncoder vs ChemPropEncoder"
)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--n-warmup", type=int, default=3)
parser.add_argument("--n-rep", type=int, default=10)
parser.add_argument("--cpu-only", action="store_true")
parser.add_argument(
    "--no-backward",
    action="store_true",
    help="Skip backward pass (forward-only benchmark)",
)
args, _ = parser.parse_known_args()

BATCH_SIZE = args.batch_size
N_WARMUP = args.n_warmup
N_REP = args.n_rep
DEVICE = torch.device(
    "cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda"
)
HAS_CUDA = DEVICE.type == "cuda"


# -- helpers -------------------------------------------------------------------


def banner(title: str):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


def mean_std(times):
    m = statistics.mean(times)
    s = statistics.stdev(times) if len(times) > 1 else 0.0
    return m, s


def sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def fmt(ms_list):
    m, s = mean_std(ms_list)
    return f"{m:8.1f} +/- {s:5.1f} ms"


# -- load batch from GEOM LMDB ------------------------------------------------


def load_geom_batch(batch_size: int):
    """Read batch_size molecules from the GEOM LMDB and return a batched
    TensorDict (on DEVICE) plus a list of SMILES strings."""
    from rdkit import Chem
    from tabasco.chem.convert import MoleculeConverter
    from tabasco.data.utils import TensorDictCollator

    print(f"\nLoading {batch_size} molecules from GEOM LMDB ...")
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
            if i >= batch_size * 2:
                break
            data = pickle.loads(val)
            mol = data["molecule"]
            try:
                td = converter.to_tensor(
                    mol=mol,
                    pad_to_size=71,  # GEOM max_num_atoms
                    remove_hydrogens=True,
                )
                smi = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
                # Set SMILES on each individual TensorDict before collation,
                # mirroring how lmdb_unconditional.py does it.  TensorDictCollator
                # will automatically create a NonTensorStack when stacking.
                td.set_non_tensor("smiles", smi)
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


# -- build FlowMatchingModel --------------------------------------------------


def build_flow_model(encoder_name: str):
    """Construct a FlowMatchingModel with the specified encoder for REPA.

    Args:
        encoder_name: "mace" or "chemprop"

    Returns:
        FlowMatchingModel on DEVICE with .train() mode.
    """
    from tabasco.models.components.transformer_module import TransformerModule
    from tabasco.models.components.encoders import (
        MACEEncoder,
        ChemPropEncoder,
        Projector,
    )
    from tabasco.models.components.losses import REPALoss
    from tabasco.models.flow_model import FlowMatchingModel
    from tabasco.flow.interpolate import SDEMetricInterpolant, DiscreteInterpolant
    from tabasco.flow.time_factor import InverseTimeFactor
    from tabasco.sample.noise_schedule import SampleNoiseSchedule

    # -- network (matches mild.yaml) --
    net = TransformerModule(
        spatial_dim=3,
        atom_dim=9,
        hidden_dim=128,
        num_layers=16,
        num_heads=8,
        activation="SiLU",
        implementation="reimplemented",
        cross_attention=True,
    )

    # -- encoder + projector --
    if encoder_name == "mace":
        encoder = MACEEncoder(
            model_name="small",
            invariants_only=True,
            num_layers=-1,
            dataset_normalizer=2.0,
        )
        encoder_dim = encoder.encoder_dim  # 192
    elif encoder_name == "chemprop":
        encoder = ChemPropEncoder(pretrained="chemeleon")
        encoder_dim = encoder.encoder_dim  # 2048
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    projector = Projector(hidden_dim=128, encoder_dim=encoder_dim)

    repa_loss = REPALoss(
        encoder=encoder,
        projector=projector,
        lambda_repa=0.8,
        combination_mode="additive",
        similarity_type="cosine",
        time_weighting=False,
    )

    # -- interpolants (match flow_model.yaml) --
    time_factor_coords = InverseTimeFactor(
        max_value=100.0, min_value=0.05, zero_before=0.0, eps=1e-6
    )
    time_factor_atomics = InverseTimeFactor(
        max_value=100.0, min_value=0.05, zero_before=0.0, eps=1e-6
    )
    langevin_schedule = SampleNoiseSchedule(cutoff=0.9)

    coords_interpolant = SDEMetricInterpolant(
        key="coords",
        loss_weight=1.0,
        scale_noise_by_log_num_atoms=False,
        noise_scale=1.0,
        langevin_sampling_schedule=langevin_schedule,
        white_noise_sampling_scale=0.01,
        time_factor=time_factor_coords,
    )
    atomics_interpolant = DiscreteInterpolant(
        key="atomics",
        loss_weight=0.1,
        time_factor=time_factor_atomics,
    )

    model = FlowMatchingModel(
        net=net,
        coords_interpolant=coords_interpolant,
        atomics_interpolant=atomics_interpolant,
        time_distribution="beta",
        time_alpha_factor=1.8,
        interdist_loss=None,
        repa_loss=repa_loss,
        num_random_augmentations=7,
        sample_schedule="log",
        compile=False,  # avoid compile overhead in benchmark
    )

    model = model.to(DEVICE).train()

    # Initialize LazyLinear in the Projector by running a dummy forward pass.
    # The projector's input dim depends on whether cross_attention produces 1 or 2
    # hidden state heads (here: cross_attention=True -> 2 * hidden_dim = 256).
    with torch.no_grad():
        dummy_batch = torch.zeros(1, 71, 3, device=DEVICE)
        dummy_atomics = torch.zeros(1, 71, 9, device=DEVICE)
        dummy_atomics[..., 0] = 1.0  # carbon
        dummy_pad = torch.ones(1, 71, dtype=torch.bool, device=DEVICE)
        dummy_pad[0, :5] = False
        from tensordict import TensorDict as TD

        dummy_td = TD(
            {
                "coords": dummy_batch,
                "atomics": dummy_atomics,
                "padding_mask": dummy_pad,
            },
            batch_size=1,
        )
        _ = model(dummy_td, compute_stats=False)

    return model


# -- benchmarking --------------------------------------------------------------


def benchmark_encoder_only(model, batch, smiles_list, n_warmup, n_rep):
    """Measure encoder-only forward time (frozen, no grad)."""
    encoder = model.repa_loss.encoder
    coords = batch["coords"]
    atomics = batch["atomics"]
    padding_mask = batch["padding_mask"]

    times = []
    for i in range(n_warmup + n_rep):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            encoder(coords, atomics, padding_mask, smiles=smiles_list)
        sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            times.append(elapsed_ms)
        print(f"    encoder rep {i+1}/{n_warmup+n_rep}: {elapsed_ms:.1f} ms")
    return times


def benchmark_model_forward(model, batch, n_warmup, n_rep):
    """Measure model.forward() (includes encoder + network + loss, no backward)."""
    times = []
    for i in range(n_warmup + n_rep):
        sync()
        t0 = time.perf_counter()
        loss, stats = model(batch, compute_stats=False)
        sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            times.append(elapsed_ms)
        print(
            f"    forward rep {i+1}/{n_warmup+n_rep}: {elapsed_ms:.1f} ms  loss={loss.item():.4f}"
        )
    return times


def benchmark_full_step(model, batch, n_warmup, n_rep):
    """Measure full training step: forward + backward + (simulated) optimizer step."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=2e-3
    )

    times_total = []
    times_forward = []
    times_backward = []

    for i in range(n_warmup + n_rep):
        optimizer.zero_grad()

        sync()
        t0 = time.perf_counter()

        # Forward
        loss, stats = model(batch, compute_stats=False)
        sync()
        t_fwd = time.perf_counter()

        # Backward
        loss.backward()
        sync()
        t_bwd = time.perf_counter()

        # Optimizer step
        optimizer.step()
        sync()
        t_end = time.perf_counter()

        fwd_ms = (t_fwd - t0) * 1000
        bwd_ms = (t_bwd - t_fwd) * 1000
        total_ms = (t_end - t0) * 1000

        if i >= n_warmup:
            times_forward.append(fwd_ms)
            times_backward.append(bwd_ms)
            times_total.append(total_ms)

        print(
            f"    step {i+1}/{n_warmup+n_rep}: "
            f"total={total_ms:.1f} ms  fwd={fwd_ms:.1f} ms  bwd={bwd_ms:.1f} ms  "
            f"loss={loss.item():.4f}"
        )

    return times_total, times_forward, times_backward


# -- main ----------------------------------------------------------------------


def main():
    print("\n" + "=" * 72)
    print("  Training Step Benchmark: MACEEncoder vs ChemPropEncoder")
    print(f"  Device: {DEVICE}  |  Batch size: {BATCH_SIZE}")
    print(
        f"  N_warmup={N_WARMUP}  N_rep={N_REP}  backward={'yes' if not args.no_backward else 'no'}"
    )
    print("=" * 72)

    if not (LMDB_DIR / "train.lmdb").exists():
        print(f"\n[ERROR] GEOM LMDB not found at {LMDB_DIR / 'train.lmdb'}")
        sys.exit(1)

    batch, smiles_list = load_geom_batch(BATCH_SIZE)

    # Store results for each encoder
    results = {}

    for enc_name in ["mace", "chemprop"]:
        banner(f"{enc_name.upper()} Encoder")

        print(f"\n  Building FlowMatchingModel with {enc_name} encoder ...")
        t_build = time.perf_counter()
        model = build_flow_model(enc_name)
        build_ms = (time.perf_counter() - t_build) * 1000
        print(f"  Model built in {build_ms:.0f} ms")

        n_params_total = sum(p.numel() for p in model.parameters())
        n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params_total:,} total, {n_params_train:,} trainable")

        # --- encoder-only ---
        print(f"\n  [1] Encoder-only forward ({N_WARMUP}w + {N_REP} timed)")
        enc_times = benchmark_encoder_only(model, batch, smiles_list, N_WARMUP, N_REP)

        # --- forward only ---
        print(f"\n  [2] Model forward only ({N_WARMUP}w + {N_REP} timed)")
        fwd_only_times = benchmark_model_forward(model, batch, N_WARMUP, N_REP)

        # --- full step ---
        if not args.no_backward:
            print(
                f"\n  [3] Full training step: fwd + bwd + optim ({N_WARMUP}w + {N_REP} timed)"
            )
            step_total, step_fwd, step_bwd = benchmark_full_step(
                model, batch, N_WARMUP, N_REP
            )
        else:
            step_total = step_fwd = step_bwd = []

        results[enc_name] = {
            "encoder": enc_times,
            "fwd_only": fwd_only_times,
            "step_total": step_total,
            "step_fwd": step_fwd,
            "step_bwd": step_bwd,
        }

        # Free GPU memory before loading next model
        del model
        if HAS_CUDA:
            torch.cuda.empty_cache()

    # -- comparison table ------------------------------------------------------
    banner("Comparison Table")

    rows = [
        ("Encoder-only forward", "encoder"),
        ("Model forward (fwd only)", "fwd_only"),
    ]
    if not args.no_backward:
        rows += [
            ("Full step (fwd+bwd+optim)", "step_total"),
            ("  -> forward portion", "step_fwd"),
            ("  -> backward portion", "step_bwd"),
        ]

    header = f"  {'Metric':<30}  {'MACE':>22}  {'ChemProp':>22}  {'Ratio':>8}"
    print(header)
    print(f"  {'-'*30}  {'-'*22}  {'-'*22}  {'-'*8}")

    for label, key in rows:
        mace_times = results["mace"][key]
        cp_times = results["chemprop"][key]
        if not mace_times or not cp_times:
            continue
        m_mace, _ = mean_std(mace_times)
        m_cp, _ = mean_std(cp_times)
        ratio = m_cp / m_mace if m_mace > 0 else float("inf")
        print(
            f"  {label:<30}  {fmt(mace_times):>22}  {fmt(cp_times):>22}  {ratio:>7.2f}x"
        )

    # -- derived metrics -------------------------------------------------------
    print()
    for enc_name in ["mace", "chemprop"]:
        r = results[enc_name]
        m_enc, _ = mean_std(r["encoder"])
        m_fwd, _ = mean_std(r["fwd_only"])
        model_minus_enc = m_fwd - m_enc
        print(
            f"  {enc_name.upper():>10}: "
            f"encoder={m_enc:.1f} ms, "
            f"model_fwd={m_fwd:.1f} ms, "
            f"model_minus_encoder={model_minus_enc:.1f} ms"
        )

    # -- projected epoch time --------------------------------------------------
    if not args.no_backward:
        N_STEPS_EPOCH = 4579  # GEOM, B=256
        print(f"\n  Projected epoch time (GEOM, B=256, ~{N_STEPS_EPOCH} steps/epoch):")
        for enc_name in ["mace", "chemprop"]:
            r = results[enc_name]
            if r["step_total"]:
                m_step, _ = mean_std(r["step_total"])
                epoch_h = m_step * N_STEPS_EPOCH / 3_600_000
                print(
                    f"    {enc_name.upper():>10}: "
                    f"{m_step:.0f} ms/step -> {epoch_h:.2f} h/epoch"
                )

    print()


if __name__ == "__main__":
    main()
