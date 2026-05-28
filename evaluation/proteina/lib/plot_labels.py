"""Plot legend helpers — single source of truth for run/step formatting.

Every paper plot under ``evaluation/proteina/{generation,representation}/figures/paper/``
must carry the checkpoint step in its legend so the data lineage is readable
straight off the figure. Use :func:`pretty_run_label` for the legend string.

Step source priority:
1. Explicit ``step=`` arg (always wins — typically read from a row's ``step``
   column / JSONL field).
2. Step parsed from the run id suffix (``_step200k`` / ``_step000000200000``).
3. If neither, raise — see the guardrail in :func:`pretty_run_label`. Plot
   authors must thread step info through; silent omission is a regression of
   the lineage we are fixing.

Note: ``_steplast`` and ``_epNN`` suffixes are NOT parseable to a numbered step.
The same ``_steplast`` run id can map to different real steps across suites
(generation vs representation evaluated last.ckpt at different points in
training), so we never trust the suffix and always require the step to come
from the data row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

__all__ = [
    "format_step",
    "split_run_step",
    "pretty_run_label",
    "RunMeta",
    "RUN_META",
    "get_run_meta",
    "block_label_plan",
    "compose_legend_label",
    "compose_title_suffix",
]

# Matches ``..._step200k`` / ``..._step200000`` at end of run id.
_STEP_RE = re.compile(r"_step(\d+)(k|K)?$")
# Matches ``..._epNN`` at end of run id (epoch — NOT resolvable to step).
_EP_RE = re.compile(r"_ep\d+$")
# Matches ``..._steplast`` at end of run id.
_LAST_RE = re.compile(r"_steplast$")


def format_step(step: Optional[int]) -> str:
    """``200000 -> "200k"``, ``1_300_000 -> "1.3M"``, ``None -> "?"``."""
    if step is None:
        return "?"
    s = int(step)
    if s >= 1_000_000:
        v = s / 1_000_000
        return f"{v:.1f}M".replace(".0M", "M")
    if s >= 1_000:
        v = s / 1_000
        # Use one decimal only if not a round multiple of 1000 (e.g. 185.5k).
        if s % 1000 == 0:
            return f"{int(v)}k"
        return f"{v:.1f}k"
    return str(s)


def split_run_step(run_id: str) -> Tuple[str, Optional[int]]:
    """Split a run id into (family, step).

    ``"repa_l4_128_per_residue_step400k" -> ("repa_l4_128_per_residue", 400000)``
    ``"repa_l4_256_ep22" -> ("repa_l4_256_ep22", None)``  (epoch is not a step)
    ``"repa_l0_128_bs80_steplast" -> ("repa_l0_128_bs80_steplast", None)``
    """
    m = _STEP_RE.search(run_id)
    if m:
        n = int(m.group(1))
        if m.group(2):  # "_step200k"
            n *= 1000
        family = run_id[: m.start()]
        return family, n
    return run_id, None


def pretty_run_label(
    run: str,
    step: Optional[int] = None,
    display: Optional[str] = None,
    *,
    allow_missing_step: bool = False,
) -> str:
    """Build a legend label of the form ``"<display> @ <step>k"``.

    Args:
        run: Run id (may or may not contain a ``_stepNNNk`` suffix).
        step: Explicit step (preferred — typically from a data row's ``step``
            column). Wins over any suffix in ``run``.
        display: Optional pretty display name. Defaults to the run family
            (run id with any ``_stepNNNk`` suffix stripped).
        allow_missing_step: If True, return the display name unchanged when no
            step can be determined. Default False (raise) — keeps every legend
            entry traceable.

    If ``display`` already contains "@" or a step pattern (e.g. ``"@ 200k"``),
    the function is idempotent and returns it unchanged.
    """
    family, suffix_step = split_run_step(run)
    if step is None:
        step = suffix_step

    if display is None:
        display = family

    # Idempotency: if caller already baked the step into the display, don't double-tag.
    if "@" in display or re.search(r"\b\d+(\.\d+)?[kKM]\b", display):
        return display

    if step is None:
        if allow_missing_step:
            return display
        raise ValueError(
            f"pretty_run_label: no step for run={run!r} (display={display!r}). "
            "Pass step= explicitly or use a run id with a _stepNNNk suffix."
        )

    return f"{display} @ {format_step(step)}"


# ---------------------------------------------------------------------------
# Run metadata — model / encoder / training dataset per run key.
# ---------------------------------------------------------------------------
#
# Convention:
#   model:    "baseline" | "REPA L0/L4/L9" | "NGC 60M (12L)"
#   encoder:  None for baseline/reference; one of
#             "CA-GearNet", "CA-GearNet (random init)", "ProteinMPNN",
#             "ESM2", "PW-Structure", "PW-Torsional".
#   dataset:  "PDB" | "AFDB".
#
# Add new runs here once and every paper figure inherits the right legend +
# block-title behavior. Keep ordered roughly by (n, dataset, model, encoder).


@dataclass(frozen=True)
class RunMeta:
    model: str
    encoder: Optional[str]
    dataset: Optional[str]  # None = not applicable / mixed (e.g. external NGC ckpt)


_GEARNET = "CA-GearNet"
_GEARNET_RAND = "CA-GearNet (random init)"
_MPNN = "ProteinMPNN"
_ESM2 = "ESM2"
_PWS = "PW-Structure"
_PWT = "PW-Torsional"


RUN_META: dict[str, RunMeta] = {
    # ---- n=128 / PDB / baseline ----
    "baseline_128_bs24_step200k": RunMeta("baseline", None, "PDB"),
    "baseline_128_bs24_step400k": RunMeta("baseline", None, "PDB"),
    "baseline_128_bs80_step200k": RunMeta("baseline", None, "PDB"),
    "baseline_128_bs80_lr3x_step200k": RunMeta("baseline", None, "PDB"),
    # ---- n=128 / PDB / REPA ----
    "repa_l0_128_bs80_step200k": RunMeta("REPA L0", _GEARNET, "PDB"),
    "repa_l0_128_bs80_steplast": RunMeta("REPA L0", _GEARNET, "PDB"),
    "repa_l0_128_per_residue_step400k": RunMeta("REPA L0", _GEARNET, "PDB"),
    "repa_l4_128_bs24_step200k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_128_bs24_step400k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_128_bs80_step200k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_128_bs80_lr3x_steplast": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_128_bs80_lambda025_step200k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_128_bs80_lambda1_step200k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_128_bs80_lambda2_steplast": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_128_bs80_wd1e2_step200k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_128_per_residue_step400k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l9_128_bs80_step200k": RunMeta("REPA L9", _GEARNET, "PDB"),
    "repa_l9_128_bs80_steplast": RunMeta("REPA L9", _GEARNET, "PDB"),
    "repa_l9_128_per_residue_step400k": RunMeta("REPA L9", _GEARNET, "PDB"),
    # ---- n=128 / PDB / REPA / alt encoders ----
    "repa_l4_128_random_step200k": RunMeta("REPA L4", _GEARNET_RAND, "PDB"),
    "repa_l4_128_pw_structure_step100k": RunMeta("REPA L4", _PWS, "PDB"),
    "repa_l4_128_pw_torsional_step100k": RunMeta("REPA L4", _PWT, "PDB"),
    "repa_mpnn_l4_128_bs80_step200k": RunMeta("REPA L4", _MPNN, "PDB"),
    "repa_esm_l4_128_step200k": RunMeta("REPA L4", _ESM2, "PDB"),
    # ---- n=128 / AFDB / baseline ----
    "baseline_afdb_128_bs80_step200k": RunMeta("baseline", None, "AFDB"),
    "baseline_afdb_128_bs80_step400k": RunMeta("baseline", None, "AFDB"),
    "baseline_afdb_128_bs80_step600k": RunMeta("baseline", None, "AFDB"),
    "baseline_afdb_128_bs80_step800k": RunMeta("baseline", None, "AFDB"),
    "baseline_afdb_128_bs80_step1000k": RunMeta("baseline", None, "AFDB"),
    "baseline_afdb_128_bs80_step1200k": RunMeta("baseline", None, "AFDB"),
    # ---- n=128 / AFDB / REPA ----
    "repa_l4_afdb_128_bs80_step200k": RunMeta("REPA L4", _GEARNET, "AFDB"),
    "repa_l4_afdb_128_bs80_step600k": RunMeta("REPA L4", _GEARNET, "AFDB"),
    "repa_mpnn_l4_afdb_128_bs80_step200k": RunMeta("REPA L4", _MPNN, "AFDB"),
    "repa_mpnn_l4_afdb_128_bs80_step400k": RunMeta("REPA L4", _MPNN, "AFDB"),
    "repa_mpnn_l4_afdb_128_bs80_step600k": RunMeta("REPA L4", _MPNN, "AFDB"),
    "repa_mpnn_l4_afdb_128_bs80_step800k": RunMeta("REPA L4", _MPNN, "AFDB"),
    "repa_mpnn_l4_afdb_128_bs80_step1000k": RunMeta("REPA L4", _MPNN, "AFDB"),
    # ---- n=256 / PDB / baseline ----
    "baseline_256_ep21": RunMeta("baseline", None, "PDB"),
    # ---- n=256 / PDB / REPA ----
    "repa_l0_256_ep26": RunMeta("REPA L0", _GEARNET, "PDB"),
    "repa_l0_256_per_sample_steplast": RunMeta("REPA L0", _GEARNET, "PDB"),
    "repa_l4_256_ep13_step300k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_256_ep22": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_256_ep31_step500k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_256_per_residue_lambda1_step200k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_256_per_residue_lambda1_step300k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_256_per_residue_lambda2_step200k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_256_per_residue_lambda2_step300k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l4_256_per_sample_step400k": RunMeta("REPA L4", _GEARNET, "PDB"),
    "repa_l9_256_ep17": RunMeta("REPA L9", _GEARNET, "PDB"),
    "repa_l9_256_ep25": RunMeta("REPA L9", _GEARNET, "PDB"),
    "repa_l9_256_per_sample_steplast": RunMeta("REPA L9", _GEARNET, "PDB"),
    # ---- n=256 / PDB / REPA / alt encoders ----
    "repa_l4_256_random_ep17": RunMeta("REPA L4", _GEARNET_RAND, "PDB"),
    "repa_mpnn_l4_256_per_residue_step300k": RunMeta("REPA L4", _MPNN, "PDB"),
    "repa_esm_l9_t30_256_steplast": RunMeta("REPA L9", _ESM2, "PDB"),
    # ---- n=256 / AFDB / baseline ----
    "baseline_afdb_256_ep20": RunMeta("baseline", None, "AFDB"),
    "baseline_afdb_256_step400k": RunMeta("baseline", None, "AFDB"),
    "baseline_afdb_256_step700k": RunMeta("baseline", None, "AFDB"),
    "baseline_afdb_256_step900k": RunMeta("baseline", None, "AFDB"),
    # ---- n=256 / AFDB / REPA ----
    "repa_l4_afdb_256_ep20": RunMeta("REPA L4", _GEARNET, "AFDB"),
    "repa_l4_afdb_256_step400k": RunMeta("REPA L4", _GEARNET, "AFDB"),
    "repa_l4_afdb_256_step700k": RunMeta("REPA L4", _GEARNET, "AFDB"),
    "repa_mpnn_l4_afdb_256_step400k": RunMeta("REPA L4", _MPNN, "AFDB"),
    "repa_mpnn_l9_afdb_256_step400k": RunMeta("REPA L9", _MPNN, "AFDB"),
    "repa_mpnn_l9_afdb_256_step700k": RunMeta("REPA L9", _MPNN, "AFDB"),
    # ---- external / reference ----
    # NGC pretrained 60M: NVIDIA's released 12-layer proteina checkpoint,
    # trained on AFDB. Encoder=None since this is an external reference, not
    # one of our REPA-aligned runs.
    "pretrained_dfs_60m": RunMeta("NGC 60M (12L)", None, "AFDB"),
    "pretrained_dfs_60m_n128_paper": RunMeta("NGC 60M (12L)", None, "AFDB"),
}


def get_run_meta(run_key: str) -> Optional[RunMeta]:
    """Lookup a run's metadata. Returns ``None`` for non-trained references
    (``random_gauss``, ``seq_onehot``, ``untrained_proteina``, probe-feature
    sentinels, ...). Callers should fall back to ``pretty_run_label`` for those.
    """
    return RUN_META.get(run_key)


_META_FIELDS = ("model", "encoder", "dataset")


def block_label_plan(
    run_keys: Iterable[str],
    *,
    fields: Iterable[str] = _META_FIELDS,
) -> Tuple[dict, frozenset]:
    """Inspect a block's run keys and decide what to put in the title vs the legend.

    Returns ``(shared_meta, varying_fields)``:

    * ``shared_meta``  — ``{field: value}`` for fields where every run in the
      block (that has a ``RunMeta`` entry) reports the same value. Fields that
      vary, or that are unknown for every run, are absent.
    * ``varying_fields`` — frozenset of field names that differ across at
      least two runs (or are missing for some).

    Runs without a ``RunMeta`` entry are ignored for the purposes of computing
    shared fields (they don't drag every field into the legend).
    """
    field_list = tuple(fields)
    values_by_field: dict[str, set] = {f: set() for f in field_list}
    for key in run_keys:
        meta = get_run_meta(key)
        if meta is None:
            continue
        for f in field_list:
            values_by_field[f].add(getattr(meta, f))

    shared: dict = {}
    varying = set()
    for f, vals in values_by_field.items():
        # Drop None from the comparison: an absent encoder for a baseline
        # shouldn't count as "encoder varies" inside an otherwise-uniform block.
        non_null = {v for v in vals if v is not None}
        if len(non_null) <= 1 and len(vals) <= 1:
            # Field is uniform across the block (incl. all-None or all-same).
            if non_null:
                shared[f] = next(iter(non_null))
        elif len(non_null) == 1 and None in vals:
            # All non-baselines share one value; baselines have None. Treat as
            # shared — the baseline simply doesn't have that field.
            shared[f] = next(iter(non_null))
        else:
            varying.add(f)
    return shared, frozenset(varying)


def _format_meta_field(field: str, value: Optional[str]) -> str:
    if value is None:
        return ""
    return value


def compose_legend_label(
    run_key: str,
    *,
    step: Optional[int] = None,
    variant_tag: Optional[str] = None,
    varying_fields: Iterable[str] = (),
    fallback_display: Optional[str] = None,
    allow_missing_step: bool = True,
) -> str:
    """Build a legend label using only the metadata fields that vary within the block.

    Output shape (fields elided when not in ``varying_fields`` or not applicable):

        ``"<model> [<variant_tag>] [/ <encoder>] [(<dataset>)] @ <step>"``

    Examples:
        compose_legend_label("repa_l4_128_bs80_step200k", step=200_000,
                             varying_fields={"model"})
            -> "REPA L4 @ 200k"
        compose_legend_label("repa_mpnn_l4_128_bs80_step200k", step=200_000,
                             varying_fields={"model", "encoder"})
            -> "REPA L4 / ProteinMPNN @ 200k"
        compose_legend_label("repa_l4_afdb_128_bs80_step200k", step=200_000,
                             varying_fields={"dataset"})
            -> "REPA L4 (AFDB) @ 200k"
        compose_legend_label("repa_l4_128_bs80_lambda2_steplast", step=200_000,
                             variant_tag="λ=2.0", varying_fields={"model"})
            -> "REPA L4 λ=2.0 @ 200k"
    """
    meta = get_run_meta(run_key)
    varying = frozenset(varying_fields)

    if meta is None:
        # Fallback for analytic / reference rows with no RunMeta entry.
        display = fallback_display or run_key
        return pretty_run_label(
            run_key, step=step, display=display, allow_missing_step=allow_missing_step
        )

    parts: list[str] = [meta.model]
    if variant_tag:
        parts.append(variant_tag)
    head = " ".join(parts)

    if "encoder" in varying and meta.encoder is not None:
        head = f"{head} / {meta.encoder}"
    if "dataset" in varying:
        head = f"{head} ({meta.dataset})"

    return pretty_run_label(
        run_key, step=step, display=head, allow_missing_step=allow_missing_step
    )


def compose_title_suffix(
    run_keys: Iterable[str],
    *,
    fields: Iterable[str] = ("encoder", "dataset"),
    separator: str = ", ",
) -> str:
    """Return a parenthesised title suffix like ``" (CA-GearNet, PDB)"`` for the
    metadata fields that are shared across all runs in the block. Empty string
    if no listed field is shared.

    Only ``encoder`` and ``dataset`` are pulled into the title by default
    (model is the typical sweep axis and rarely shared across an interesting
    block). Pass ``fields=("model","encoder","dataset")`` to include model
    when it's shared (e.g. lambda/wd sweeps that hold model fixed).
    """
    shared, _ = block_label_plan(run_keys, fields=fields)
    pieces = [shared[f] for f in fields if f in shared and shared[f] is not None]
    if not pieces:
        return ""
    return f" ({separator.join(pieces)})"
