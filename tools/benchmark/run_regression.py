#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cmdlog_metrics import compute_metrics_from_jsonl


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _check_acceptance(metrics: Dict[str, Any], gates: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    checks: List[Dict[str, Any]] = []
    ok = True

    def add(name: str, passed: bool, value: Any, rule: str) -> None:
        nonlocal ok
        checks.append({"name": name, "pass": passed, "value": value, "rule": rule})
        ok = ok and passed

    min_frames = int(gates.get("min_frames_total", 0))
    if min_frames > 0:
        v = int(metrics.get("frames_total", 0))
        add("min_frames_total", v >= min_frames, v, f">= {min_frames}")

    if "min_hit_window_rate" in gates:
        thr = float(gates["min_hit_window_rate"])
        v = float(metrics.get("hit_window_rate", 0.0))
        add("min_hit_window_rate", v >= thr, v, f">= {thr}")

    if "min_predicted_window_ok_rate" in gates:
        thr = float(gates["min_predicted_window_ok_rate"])
        v = float(metrics.get("predicted_window_ok_rate", 0.0))
        add("min_predicted_window_ok_rate", v >= thr, v, f">= {thr}")

    if "min_fire_enable_rate" in gates:
        thr = float(gates["min_fire_enable_rate"])
        v = float(metrics.get("fire_enable_rate", 0.0))
        add("min_fire_enable_rate", v >= thr, v, f">= {thr}")

    if "max_pnp_error_mean" in gates:
        thr = float(gates["max_pnp_error_mean"])
        v = float(metrics.get("pnp_error_mean", 0.0))
        add("max_pnp_error_mean", v <= thr, v, f"<= {thr}")

    if "max_loss_avg_frames" in gates:
        thr = float(gates["max_loss_avg_frames"])
        v = float(metrics.get("loss_avg_frames", 0.0))
        add("max_loss_avg_frames", v <= thr, v, f"<= {thr}")

    if "max_relock_avg_frames" in gates:
        thr = float(gates["max_relock_avg_frames"])
        v = float(metrics.get("relock_avg_frames", 0.0))
        add("max_relock_avg_frames", v <= thr, v, f"<= {thr}")

    return ok, checks


def _check_regression(
    metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    tol: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    checks: List[Dict[str, Any]] = []
    ok = True
    higher = tol.get("higher_is_better", {})
    lower = tol.get("lower_is_better", {})

    for k, allowed_drop in higher.items():
        if k not in baseline:
            continue
        cur = float(metrics.get(k, 0.0))
        base = float(baseline.get(k, 0.0))
        drop = base - cur
        passed = drop <= float(allowed_drop)
        checks.append({
            "name": k,
            "direction": "higher_is_better",
            "baseline": base,
            "current": cur,
            "delta": cur - base,
            "allowed_drop": float(allowed_drop),
            "pass": passed,
        })
        ok = ok and passed

    for k, allowed_increase in lower.items():
        if k not in baseline:
            continue
        cur = float(metrics.get(k, 0.0))
        base = float(baseline.get(k, 0.0))
        inc = cur - base
        passed = inc <= float(allowed_increase)
        checks.append({
            "name": k,
            "direction": "lower_is_better",
            "baseline": base,
            "current": cur,
            "delta": cur - base,
            "allowed_increase": float(allowed_increase),
            "pass": passed,
        })
        ok = ok and passed

    return ok, checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run classic-route regression from scenario logs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/benchmark/classic_acceptance_manifest.json"),
        help="Scenario manifest JSON path",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("config/benchmark/classic_baseline.json"),
        help="Baseline metrics JSON path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/classic_regression_report.json"),
        help="Output report path",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current metrics as new baseline",
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest, {})
    scenarios = manifest.get("scenarios", [])
    acceptance_gates = manifest.get("acceptance_gates", {})
    regression_tolerance = manifest.get("regression_tolerance", {})

    baseline_doc = _load_json(args.baseline, {"schema_version": "1.0", "scenarios": {}})
    baseline_scenarios = baseline_doc.get("scenarios", {})

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "baseline": str(args.baseline),
        "overall_pass": True,
        "scenarios": [],
    }

    current_for_baseline: Dict[str, Any] = {}

    if not scenarios:
        print("No scenarios found in manifest.")
        return 2

    for sc in scenarios:
        name = sc["name"]
        log_path = Path(sc["log_path"])
        result: Dict[str, Any] = {
            "name": name,
            "log_path": str(log_path),
            "exists": log_path.exists(),
            "pass": False,
        }
        if not log_path.exists():
            result["error"] = "log file not found"
            report["overall_pass"] = False
            report["scenarios"].append(result)
            continue

        metrics = compute_metrics_from_jsonl(log_path)
        current_for_baseline[name] = metrics
        result["metrics"] = metrics

        scenario_gates = dict(acceptance_gates)
        scenario_gates.update(sc.get("acceptance_overrides", {}))
        acc_ok, acc_checks = _check_acceptance(metrics, scenario_gates)
        result["acceptance"] = {"pass": acc_ok, "checks": acc_checks}

        baseline_metrics = baseline_scenarios.get(name, {})
        reg_ok = True
        reg_checks: List[Dict[str, Any]] = []
        if baseline_metrics:
            reg_ok, reg_checks = _check_regression(metrics, baseline_metrics, regression_tolerance)
        result["regression"] = {"pass": reg_ok, "checks": reg_checks, "has_baseline": bool(baseline_metrics)}

        scenario_pass = acc_ok and reg_ok
        result["pass"] = scenario_pass
        report["overall_pass"] = report["overall_pass"] and scenario_pass
        report["scenarios"].append(result)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.update_baseline:
        baseline_out = {
            "schema_version": "1.0",
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "manifest": str(args.manifest),
            "scenarios": current_for_baseline,
        }
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(baseline_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[baseline] updated: {args.baseline}")

    for sc in report["scenarios"]:
        name = sc["name"]
        status = "PASS" if sc.get("pass") else "FAIL"
        msg = f"[{status}] {name}"
        if not sc.get("exists", True):
            msg += " (missing log)"
        else:
            m = sc.get("metrics", {})
            msg += (
                f" frames={m.get('frames_total', 0)}"
                f" hit={m.get('hit_window_rate', 0.0):.3f}"
                f" pnp_mean={m.get('pnp_error_mean', 0.0):.3f}"
                f" relock={m.get('relock_avg_frames', 0.0):.2f}"
            )
        print(msg)

    print(f"[report] {args.report}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

