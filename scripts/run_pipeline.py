"""End-to-end command-line run of the pipeline: data -> model -> SHAP -> notice.

    python -m scripts.run_pipeline --row 5
    python -m scripts.run_pipeline --row 5 --threshold 0.15
    python -m scripts.run_pipeline --row 5 --print-prompt-only   # no API call

Loads models/<version>/, scores one row from the dataset, explains it with
SHAP, and -- when the decision is a denial -- generates the ECOA-style
adverse-action notice.

The probability -> decision threshold here is a demo policy that lives in
this script only. The model layer produces a probability; the LLM layer is
handed the resulting decision and never sees the score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.explain import explain_row, top_contributors
from src.llm.adverse_action import result_to_json
from src.llm.client import DEFAULT_MODEL
from src.llm.prompt import build_system_prompt, build_user_prompt
from src.llm.reasons import select_reasons
from src.model.artifacts import load_model
from src.model.dataset import build_model_frame, split_xy
from src.model.train import DEFAULT_DATA_PATH


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, default=0, help="row index in the dataset")
    parser.add_argument("--version", default="v1", help="model version dir name")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="demo-only probability cutoff for a denial")
    parser.add_argument("--max-reasons", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model id")
    parser.add_argument("--print-prompt-only", action="store_true",
                        help="print the prompts that would be sent and stop")
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}")

    model, feature_names = load_model(args.models_root / args.version)
    df = build_model_frame(args.data)
    X, _ = split_xy(df)
    row = X.iloc[args.row].to_dict()

    explanation = explain_row(model, row, feature_names=feature_names)
    probability = explanation["predicted_probability"]
    decision = "denied" if probability >= args.threshold else "approved"

    print(f"row {args.row}: P(serious delinquency) = {probability:.4f}")
    print(f"decision (threshold {args.threshold}): {decision}")
    print("\ntop risk-increasing drivers (SHAP, log-odds):")
    for c in top_contributors(explanation, k=args.max_reasons):
        print(f"  {c['feature']:<42} value={c['value']:<12} shap={c['shap_value']:+.4f}")

    if decision != "denied":
        print("\nNo adverse-action notice required for an approval.")
        return

    reasons = select_reasons(explanation, max_reasons=args.max_reasons)
    statements = [r["statement"] for r in reasons]

    if args.print_prompt_only:
        print("\n--- SYSTEM PROMPT ---\n" + build_system_prompt())
        print("\n--- USER PROMPT ---\n" + build_user_prompt(statements, decision=decision))
        return

    # Imported here so --print-prompt-only needs no anthropic package / key.
    from src.llm.adverse_action import generate_adverse_action

    result = generate_adverse_action(
        explanation=explanation,
        decision=decision,
        max_reasons=args.max_reasons,
        client=None,
        model=args.model,
    )
    print("\n--- ADVERSE-ACTION NOTICE ---\n" + result.notice_text)
    print("\n--- AUDIT TRAIL ---\n" + result_to_json(result))


if __name__ == "__main__":
    main()
