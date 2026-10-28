import argparse
import json
from pathlib import Path

from .config import load_config
from .estimator import DependenceModel, fit_model, prepare_claims, scored_claim
from .external import export_public_responses, prepare_external_claims
from .io import read_data, write_data
from .metrics import evaluate_outputs


def execute_fit(config, prepared, model_path, report_path=None):
    model, report = fit_model(prepared, config)
    write_data(model_path, model.as_dict())
    if report_path:
        write_data(report_path, report)
    return model, report


def execute_score(config, prepared, model, split, include_ground_truth=False):
    return [scored_claim(item, config, model, include_ground_truth=include_ground_truth) for item in prepared if split == "all" or item["claim"].get("split") == split]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="causalquery")
    parser.add_argument("command", choices=["fit", "infer", "evaluate", "export-responses", "score-responses"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="artifacts/model.json")
    parser.add_argument("--output", default="artifacts/results.json")
    parser.add_argument("--report", default="artifacts/training_report.json")
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data = read_data(args.dataset)
    if data.get("format") == "causalquery-public-responses-v1":
        labelled_splits = {"train", "dev"} if args.command == "fit" else {args.split}
        prepared = prepare_external_claims(data, config, include_labels=args.command in {"fit", "evaluate"}, label_splits=labelled_splits if args.split != "all" else None)
    else:
        prepared = prepare_claims(data, config, include_labels=args.command in {"fit", "evaluate"})
    if args.command == "export-responses":
        write_data(args.output, export_public_responses(prepared))
        print(json.dumps({"output": str(Path(args.output).resolve()), "claims": len(prepared)}, ensure_ascii=False, indent=2))
        return
    if args.command == "fit":
        _, report = execute_fit(config, prepared, args.model, args.report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    model = DependenceModel.from_dict(read_data(args.model))
    outputs = execute_score(config, prepared, model, args.split, include_ground_truth=args.command == "evaluate")
    if args.command == "evaluate":
        result = {"evaluation": evaluate_outputs(outputs), "claims": outputs}
    else:
        result = {"claims": outputs}
    write_data(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "claims": len(outputs), **({"evaluation": result["evaluation"]} if "evaluation" in result else {})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
