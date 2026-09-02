"""Run the fairness audit for a saved model version and print / save the report.

    python -m src.fairness.report v1
    python -m src.fairness.report v1 --split all
    python -m src.fairness.report v1 --data data/raw/cs-training.csv
    python -m src.fairness.report v1 --out /tmp/v1_fairness.json --print-only

Loads ``models/<version>/``, scores the dataset's validation split (or the
whole dataset with ``--split all``) through the exact serving decision
policy, groups applicants by the feature layer's age bands, and reports
per-group approval / denial rates, adverse-impact ratios, and the
four-fifths-rule verdict.

Cutoffs
-------
The decision cutoffs come, in order of precedence, from ``--approve-below``
/ ``--deny-at-or-above``, then from the version's own ``metadata.json``
(``recommended_cutoffs``, written by ``src.model.train`` for every version
from v2 on), then from the code defaults in ``src.model.decision``. So
``python -m src.fairness.report v2`` audits v2 under v2's cutoffs -- the
same numbers serving applies via ``AIU_APPROVE_BELOW`` /
``AIU_DENY_AT_OR_ABOVE`` -- and the summary says which source was used.

The audit only *measures* decisions the model already made -- it never
scores or re-decides an application.

Artifact
--------
By default the JSON report is written to
``models/<version>/fairness_audit.json``, alongside the model it audits, per
the versioned-artifacts convention. An existing report file is never
silently overwritten: pass ``--force`` to replace one, or ``--out`` to write
elsewhere. ``--print-only`` skips writing entirely. The model's own
immutable files (``model.json``, ``eval_report.json``, ...) are never
touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fairness.audit import DEFAULT_DATA_PATH, run_audit
from src.model.cutoffs import resolve_cutoffs

_ARTIFACT_NAME = "fairness_audit.json"
_IMMUTABLE_MODEL_FILES = {
    "model.json",
    "feature_names.json",
    "eval_report.json",
    "calibration.png",
    "metadata.json",
}


def default_out_path(version: str, models_root: str | Path = "models") -> Path:
    return Path(models_root) / version / _ARTIFACT_NAME




def _check_writable(path: Path, force: bool) -> None:
    if path.name in _IMMUTABLE_MODEL_FILES:
        raise ValueError(
            f"{path.name} is an immutable model artifact; point --out at a "
            "different filename"
        )
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists; pass --force to overwrite or --out to "
            "write a new file"
        )


def _format_summary(report: dict) -> str:
    lines = [
        f"Fairness audit -- model {report['model_version']} -- "
        f"split={report['split']} -- n={report['n']}",
        f"protected attribute: {report['protected_attribute']} "
        "(see 'limitations' in the JSON)",
        f"cutoffs: approve below {report['thresholds']['approve_below']}, "
        f"deny at or above {report['thresholds']['deny_at_or_above']}"
        + (
            f"  [{report['cutoffs_source']}]"
            if report.get("cutoffs_source")
            else ""
        ),
        "",
        f"{'group':<10} {'n':>7} {'approve%':>9} {'refer%':>9} "
        f"{'deny%':>9} {'appr.AIR':>9} {'deny.ratio':>11}",
    ]
    air = report["disparate_impact"]["approval"]["ratios"]
    dratio = report["denial_rate_disparity"]["ratios"]
    for g in report["groups"]:
        name = g["group"]
        a = air[name]["adverse_impact_ratio"]
        d = dratio[name]["denial_rate_ratio"]
        lines.append(
            f"{name:<10} {g['n']:>7d} {g['approval_rate'] * 100:>8.1f}% "
            f"{g['referred_rate'] * 100:>8.1f}% {g['denial_rate'] * 100:>8.1f}% "
            f"{'n/a' if a is None else format(a, '.3f'):>9} "
            f"{'n/a' if d is None else format(d, '.3f'):>11}"
        )
    ff = report["four_fifths_rule"]
    lines += [
        "",
        f"reference group (approval): "
        f"{report['disparate_impact']['approval']['reference_group']}",
        f"min approval adverse-impact ratio: "
        f"{'n/a' if ff['approval_air_min'] is None else format(ff['approval_air_min'], '.3f')} "
        f"({ff['approval_air_min_group']})",
        f"max denial-rate ratio: "
        f"{'n/a' if ff['denial_ratio_max'] is None else format(ff['denial_ratio_max'], '.3f')} "
        f"({ff['denial_ratio_max_group']})",
        f"four-fifths rule (approval AIR >= 0.80 and denial ratio <= 1.25): "
        f"{'PASS' if ff['passes'] else 'REVIEW'}",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="model version dir name, e.g. v1")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--split",
        choices=["val", "all"],
        default="val",
        help="score the stratified validation split (default) or every row",
    )
    parser.add_argument(
        "--approve-below",
        type=float,
        default=None,
        help="override the approve cutoff (default: the version's recorded "
        "recommended_cutoffs, else src.model.decision's default)",
    )
    parser.add_argument(
        "--deny-at-or-above",
        type=float,
        default=None,
        help="override the deny cutoff (same fallback as --approve-below)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"where to write the JSON report "
        f"(default: <models-root>/<version>/{_ARTIFACT_NAME})",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing report file"
    )
    parser.add_argument(
        "--print-only", action="store_true", help="print the report, write nothing"
    )
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}")
    model_dir = args.models_root / args.version
    if not model_dir.exists():
        raise SystemExit(f"Model version not found: {model_dir}")

    out_path = args.out or default_out_path(args.version, args.models_root)
    if not args.print_only:
        try:
            _check_writable(out_path, args.force)
        except (FileExistsError, ValueError) as exc:
            raise SystemExit(str(exc))

    approve_below, deny_at_or_above, cutoffs_source = resolve_cutoffs(
        model_dir, args.approve_below, args.deny_at_or_above
    )

    try:
        report = run_audit(
            args.version,
            args.data,
            models_root=args.models_root,
            split=args.split,
            approve_below=approve_below,
            deny_at_or_above=deny_at_or_above,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    report["cutoffs_source"] = cutoffs_source

    print(_format_summary(report))
    print()

    if args.print_only:
        print(json.dumps(report, indent=2))
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
