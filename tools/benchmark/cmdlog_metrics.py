#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get(obj: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in ("1", "true", "yes", "y"):
            return True
        if vv in ("0", "false", "no", "n"):
            return False
    return default


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return None


def _safe_div(n: float, d: float) -> float:
    return n / d if d > 1e-12 else 0.0


def compute_metrics_from_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")

    frames_total = 0
    valid_frames = 0
    fire_enable_frames = 0
    center_window_ok_frames = 0
    predicted_window_ok_frames = 0
    hit_window_ok_frames = 0
    pnp_err_samples = 0
    pnp_err_sum = 0.0
    pnp_err_max = 0.0

    gate_reason_counts: Dict[str, int] = {}
    gate_fail_reason_counts: Dict[str, int] = {}
    route_mode_counts: Dict[str, int] = {}

    loss_event_count = 0
    loss_total_frames = 0
    loss_max_frames = 0
    in_loss = False
    loss_len = 0

    recapture_event_count = 0
    recapture_total_frames = 0
    recapture_max_frames = 0
    relock_success_count = 0
    relock_total_frames = 0
    relock_max_frames = 0
    in_recapture = False
    recapture_len = 0

    prev_frame_id = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                j = json.loads(raw)
            except json.JSONDecodeError:
                continue

            frame_id = _get(j, "frame", "frame_id")
            if frame_id is None:
                frame_id = _get(j, "frame_id")
            # Drop duplicated frame records in replay logs.
            if frame_id is not None and frame_id == prev_frame_id:
                continue
            prev_frame_id = frame_id

            frames_total += 1

            valid = _to_bool(_get(j, "output", "valid", default=_get(j, "valid", default=False)))
            fire_enable = _to_bool(_get(j, "output", "fire_enable", default=_get(j, "fire_enable", default=False)))
            route_mode = _get(j, "output", "route_mode", default=_get(j, "route_mode", default="UNKNOWN"))
            gate_reason = _get(j, "gate", "reason", default=_get(j, "gate_reason", default="UNKNOWN"))
            center_ok = _to_bool(
                _get(j, "gate", "center_in_window_ok", default=_get(j, "center_in_window_ok", default=False))
            )
            predicted_ok = _to_bool(
                _get(j, "gate", "predicted_window_ok", default=_get(j, "predicted_window_ok", default=False))
            )

            if valid:
                valid_frames += 1
            if fire_enable:
                fire_enable_frames += 1
            if center_ok:
                center_window_ok_frames += 1
            if predicted_ok:
                predicted_window_ok_frames += 1
            if center_ok and predicted_ok:
                hit_window_ok_frames += 1

            gate_reason_counts[gate_reason] = gate_reason_counts.get(gate_reason, 0) + 1
            if not fire_enable:
                gate_fail_reason_counts[gate_reason] = gate_fail_reason_counts.get(gate_reason, 0) + 1
            route_mode_counts[route_mode] = route_mode_counts.get(route_mode, 0) + 1

            pnp_err = _get(j, "output", "selected_pnp_laser_reproj_error")
            if pnp_err is None:
                pnp_err = _get(j, "pnp", "selected_laser_reproj_error")
            if pnp_err is None:
                pnp_err = _get(j, "selected_pnp_error")
            pnp_err_f = _to_float(pnp_err)
            if pnp_err_f is not None and pnp_err_f >= 0.0:
                pnp_err_samples += 1
                pnp_err_sum += pnp_err_f
                pnp_err_max = max(pnp_err_max, pnp_err_f)

            if not valid:
                if not in_loss:
                    in_loss = True
                    loss_len = 0
                loss_len += 1
            elif in_loss:
                in_loss = False
                loss_event_count += 1
                loss_total_frames += loss_len
                loss_max_frames = max(loss_max_frames, loss_len)
                loss_len = 0

            if route_mode == "CLASSIC_RECAPTURE":
                if not in_recapture:
                    in_recapture = True
                    recapture_len = 0
                recapture_len += 1
            elif in_recapture:
                in_recapture = False
                recapture_event_count += 1
                recapture_total_frames += recapture_len
                recapture_max_frames = max(recapture_max_frames, recapture_len)
                if valid:
                    relock_success_count += 1
                    relock_total_frames += recapture_len
                    relock_max_frames = max(relock_max_frames, recapture_len)
                recapture_len = 0

    metrics = {
        "frames_total": frames_total,
        "valid_frames": valid_frames,
        "fire_enable_frames": fire_enable_frames,
        "center_in_window_ok_frames": center_window_ok_frames,
        "predicted_window_ok_frames": predicted_window_ok_frames,
        "hit_window_ok_frames": hit_window_ok_frames,
        "valid_rate": _safe_div(valid_frames, frames_total),
        "fire_enable_rate": _safe_div(fire_enable_frames, frames_total),
        "center_window_ok_rate": _safe_div(center_window_ok_frames, frames_total),
        "predicted_window_ok_rate": _safe_div(predicted_window_ok_frames, frames_total),
        "hit_window_rate": _safe_div(hit_window_ok_frames, frames_total),
        "pnp_error_samples": pnp_err_samples,
        "pnp_error_mean": _safe_div(pnp_err_sum, pnp_err_samples),
        "pnp_error_max": pnp_err_max,
        "gate_reason_counts": gate_reason_counts,
        "gate_fail_reason_counts": gate_fail_reason_counts,
        "route_mode_counts": route_mode_counts,
        "loss_event_count": loss_event_count,
        "loss_avg_frames": _safe_div(loss_total_frames, loss_event_count),
        "loss_max_frames": loss_max_frames,
        "recapture_event_count": recapture_event_count,
        "recapture_avg_frames": _safe_div(recapture_total_frames, recapture_event_count),
        "recapture_max_frames": recapture_max_frames,
        "relock_success_count": relock_success_count,
        "relock_avg_frames": _safe_div(relock_total_frames, relock_success_count),
        "relock_max_frames": relock_max_frames,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute metrics from cmd_log.jsonl")
    parser.add_argument("log_path", type=Path, help="Path to cmd_log.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Optional metrics output path")
    args = parser.parse_args()

    metrics = compute_metrics_from_jsonl(args.log_path)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

