"""
Lưu lại dự đoán đã đưa ra, rồi đối chiếu với kết quả thật.

VÌ SAO CẦN: predictions.json được sinh lại mỗi lần build và chỉ chứa các trận
SẮP tới. Trận đá xong là biến mất, kéo theo cả dự đoán đã đưa ra cho nó. Không
có bản lưu thì không thể trả lời câu hỏi quan trọng nhất: "mô hình này đoán
đúng bao nhiêu phần?"

CÁCH LÀM: trước khi ghi đè predictions.json, đọc bản của lần build trước, tìm
những trận nay đã đá xong, gắn kết quả thật cùng người ghi bàn, rồi nối vào
history.json.

Điểm tinh tế: dự đoán được lưu là bản của lần build GẦN NHẤT trước giờ bóng
lăn, chứ không phải bản đầu tiên. Đó mới là dự đoán mà người dùng thực sự nhìn
thấy trước trận, nên cũng là bản đáng đem ra chấm điểm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sources.odds_history import build_name_map, normalise

# Giới hạn số trận giữ lại mỗi giải. Đủ cho vài mùa mà file vẫn nhẹ để điện
# thoại tải nhanh.
MAX_HISTORY = 400


def _key(date: str, home: str, away: str) -> str:
    return f"{date}|{normalise(home)}|{normalise(away)}"


def load(path: Path) -> list[dict]:
    """Đọc lịch sử đã lưu. File chưa có thì trả rỗng."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        print(f"  history.json hỏng ({path.name}) — bắt đầu lại từ rỗng")
        return []


def _result_index(results: pd.DataFrame) -> dict[str, dict]:
    """Bảng tra kết quả theo (ngày, đội nhà, đội khách)."""
    index = {}
    for _, r in results.iterrows():
        index[_key(pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                   r["home_team"], r["away_team"])] = {
            "home_goals": int(r["home_goals"]),
            "away_goals": int(r["away_goals"]),
            "result": r["result"],
        }
    return index


def _grade(pred: dict, actual: dict) -> dict:
    """
    Chấm điểm một dự đoán đã có kết quả.

    Ghi cả cửa được chọn lẫn xác suất mô hình gán cho kết quả THẬT — con số
    thứ hai quan trọng hơn: đoán trúng cửa với xác suất 40% không đáng khen
    bằng đoán trúng với 70%, và đoán sai khi đã gán 15% cho kết quả thật thì
    tệ hơn nhiều so với đoán sai khi gán 33%.
    """
    outcome = (
        (pred.get("market") or {}).get("prediction", {}).get("outcome")
        if (pred.get("market") or {}).get("available")
        else pred["prediction"]["outcome"]
    ) or pred["prediction"]["outcome"]

    letter = actual["result"]
    key = {"H": "home", "D": "draw", "A": "away"}[letter]
    prob_actual = float(outcome.get(key, 0.0))

    picked = max(outcome, key=outcome.get)
    signal = pred.get("signal") or {}
    best = signal.get("best") or {}

    # Kiểm tra riêng cửa mà app đã nêu trong phần nhận định — cửa đó có thể
    # không phải 1X2 (ví dụ "Tài 2.5"), nên chỉ chấm khi nó thuộc nhóm 1X2.
    signal_key = best.get("key")
    signal_correct = None
    if signal_key in ("home", "draw", "away"):
        signal_correct = signal_key == key

    score_pred = (signal.get("score") or {}).get("top")
    score_correct = None
    if score_pred:
        score_correct = (
            int(score_pred["home"]) == actual["home_goals"]
            and int(score_pred["away"]) == actual["away_goals"]
        )

    # Brier cho một trận: tổng bình phương sai lệch trên cả ba cửa.
    brier = sum(
        (float(outcome.get(k, 0.0)) - (1.0 if k == key else 0.0)) ** 2
        for k in ("home", "draw", "away")
    )

    return {
        "actual_result": letter,
        "top_pick": picked,
        "top_pick_correct": picked == key,
        "probability_of_actual": round(prob_actual, 4),
        "signal_pick": best.get("pick"),
        "signal_correct": signal_correct,
        "signal_strength": signal.get("strength"),
        "score_predicted": (
            f"{score_pred['home']}-{score_pred['away']}" if score_pred else None
        ),
        "score_correct": score_correct,
        "brier": round(brier, 4),
    }


def archive_finished(
    league: str,
    league_dir: Path,
    results: pd.DataFrame,
    espn_details: dict | None = None,
) -> tuple[list[dict], int]:
    """
    Chuyển các trận đã đá từ predictions.json sang history.json.

    Trả về (lịch sử sau khi cập nhật, số trận vừa thêm).
    """
    history_file = league_dir / "history.json"
    predictions_file = league_dir / "predictions.json"

    history = load(history_file)
    seen = {_key(h["fixture"]["date"], h["home_team"], h["away_team"]) for h in history}

    if not predictions_file.exists():
        return history, 0

    try:
        previous = json.loads(predictions_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return history, 0

    index = _result_index(results)
    details = espn_details or {}

    # Tên đội của ESPN khác nguồn chính, nên quy về một mối trước khi ghép.
    espn_names = {k[1] for k in details} | {k[2] for k in details}
    our_names = set(results["home_team"]) | set(results["away_team"])
    name_map = build_name_map({normalise(n) for n in espn_names},
                              {normalise(n) for n in our_names})

    def espn_lookup(date: str, home: str, away: str) -> dict:
        for (d, h, a), value in details.items():
            if d != date:
                continue
            h_key = name_map.get(normalise(h), normalise(h))
            a_key = name_map.get(normalise(a), normalise(a))
            if h_key == normalise(home) and a_key == normalise(away):
                return value
        return {}

    added = 0
    for pred in previous:
        date = pred["fixture"]["date"]
        key = _key(date, pred["home_team"], pred["away_team"])
        if key in seen:
            continue
        actual = index.get(key)
        if not actual:
            continue    # chưa đá xong, hoặc kết quả chưa về

        detail = espn_lookup(date, pred["home_team"], pred["away_team"])

        history.append({
            "home_team": pred["home_team"],
            "away_team": pred["away_team"],
            "fixture": pred["fixture"],
            "prediction": {
                "outcome": pred["prediction"]["outcome"],
                "expected_goals": pred["expected_goals"],
                "signal": pred.get("signal"),
                "from_market": bool((pred.get("market") or {}).get("available")),
                "market_outcome": (
                    (pred.get("market") or {}).get("prediction", {}).get("outcome")
                ),
                "confidence": (pred.get("confidence") or {}).get("level"),
            },
            "actual": {
                **actual,
                "half_time": detail.get("half_time"),
                "goals": detail.get("goals", []),
                "details_source": detail.get("source"),
            },
            "grade": _grade(pred, actual),
        })
        seen.add(key)
        added += 1

    history.sort(key=lambda h: h["fixture"]["date"], reverse=True)
    return history[:MAX_HISTORY], added


def accuracy_summary(history: list[dict]) -> dict:
    """
    Thống kê độ chính xác trên toàn bộ lịch sử đã lưu.

    Kèm mốc so sánh "luôn chọn cửa có xác suất cao nhất theo tần suất chung"
    để con số độ chính xác có bối cảnh, thay vì đứng trơ một mình.
    """
    graded = [h for h in history if h.get("grade")]
    if not graded:
        return {"matches": 0}

    n = len(graded)
    correct = sum(1 for h in graded if h["grade"]["top_pick_correct"])
    brier = sum(h["grade"]["brier"] for h in graded) / n
    score_hits = [h for h in graded if h["grade"].get("score_correct") is not None]
    score_correct = sum(1 for h in score_hits if h["grade"]["score_correct"])

    by_strength = {}
    for level in ("rõ", "vừa", "mơ hồ"):
        subset = [h for h in graded if h["grade"].get("signal_strength") == level]
        checked = [h for h in subset if h["grade"].get("signal_correct") is not None]
        if checked:
            by_strength[level] = {
                "matches": len(checked),
                "correct": sum(1 for h in checked if h["grade"]["signal_correct"]),
                "accuracy": round(
                    sum(1 for h in checked if h["grade"]["signal_correct"]) / len(checked), 4
                ),
            }

    return {
        "matches": n,
        "top_pick_correct": correct,
        "top_pick_accuracy": round(correct / n, 4),
        "mean_brier": round(brier, 4),
        "score_exact": {
            "checked": len(score_hits),
            "correct": score_correct,
            "accuracy": round(score_correct / len(score_hits), 4) if score_hits else None,
        },
        "by_signal_strength": by_strength,
    }
