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


# Tăng số này mỗi khi cách chấm điểm đổi, để repair() chấm lại toàn bộ lịch sử
# thay vì giữ nguyên điểm cũ tính theo luật đã lỗi thời.
GRADE_VERSION = 2


def _pick_correct(key: str, home_goals: int, away_goals: int) -> bool | None:
    """
    Cửa đã chọn có trúng không, xét đúng theo loại kèo của nó.

    VÌ SAO CẦN: trước đây chỗ này chỉ chấm được 1X2, mọi cửa khác trả về None
    rồi giao diện lại lấy tạm kết quả 1X2 ra thay thế. Hậu quả là một trận
    2-1 với cửa đã chọn là "Tài 2.5" bị đánh dấu SAI, dù ba bàn rõ ràng là
    tài — app chấm nhầm bằng việc đội nhà thắng hay thua.

    Các mốc tài/xỉu đều là số lẻ .5 nên không bao giờ có trường hợp hoà mốc.
    """
    total = home_goals + away_goals

    if key in ("home", "draw", "away"):
        actual_key = ("home" if home_goals > away_goals
                      else "away" if home_goals < away_goals else "draw")
        return key == actual_key

    if key.startswith("over_") or key.startswith("under_"):
        try:
            line = float(key.split("_", 1)[1])
        except ValueError:
            return None
        return total > line if key.startswith("over_") else total < line

    if key == "btts_yes":
        return home_goals > 0 and away_goals > 0
    if key == "btts_no":
        return home_goals == 0 or away_goals == 0

    return None


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

    # Chấm riêng cửa mà app đã nêu trong phần nhận định. Cửa đó thường KHÔNG
    # phải 1X2 (hay gặp "Tài 2.5" hoặc "Cả hai đội ghi bàn"), nên phải chấm
    # theo đúng luật của loại kèo đó.
    signal_key = best.get("key")
    signal_correct = (
        _pick_correct(signal_key, actual["home_goals"], actual["away_goals"])
        if signal_key else None
    )

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
        "v": GRADE_VERSION,
        "actual_result": letter,
        "top_pick": picked,
        "top_pick_correct": picked == key,
        "probability_of_actual": round(prob_actual, 4),
        "signal_pick": best.get("pick"),
        "signal_market": best.get("market"),
        "signal_correct": signal_correct,
        "signal_strength": signal.get("strength"),
        "score_predicted": (
            f"{score_pred['home']}-{score_pred['away']}" if score_pred else None
        ),
        "score_correct": score_correct,
        "brier": round(brier, 4),
    }


def _stored_as_prediction(entry: dict) -> dict:
    """
    Dựng lại hình dạng mà _grade() cần từ một bản ghi đã nằm trong lịch sử.

    Cần thiết vì lịch sử lưu bản rút gọn (outcome mô hình và outcome thị
    trường tách riêng), còn _grade() nhận nguyên bản dự đoán của predictions
    .json. Không có cầu nối này thì không thể chấm lại điểm sau khi sửa tỷ số.
    """
    pred = entry.get("prediction") or {}
    market_outcome = pred.get("market_outcome")
    return {
        "prediction": {"outcome": pred.get("outcome") or {}},
        "market": (
            {"available": True, "prediction": {"outcome": market_outcome}}
            if pred.get("from_market") and market_outcome
            else {"available": False}
        ),
        "signal": pred.get("signal"),
    }


def repair(history: list[dict], index: dict[str, dict], espn_lookup) -> tuple[int, int]:
    """
    Sửa lại các bản ghi đã lưu: tỷ số sai và phần chi tiết còn thiếu.

    VÌ SAO CẦN: trước đây việc lưu lịch sử chỉ chạy MỘT LẦN cho mỗi trận. Nếu
    đúng khoảnh khắc đó nguồn kết quả chưa chốt hoặc ESPN chưa kịp công bố
    người ghi bàn, bản ghi bị đóng băng vĩnh viễn ở trạng thái sai hoặc thiếu —
    lần chạy sau thấy trận đã có trong lịch sử nên bỏ qua luôn.

    Hậu quả thật đã gặp: bốn trận C1 ngày 10/09 bị lưu đúng bằng tỷ số HIỆP 1
    (Bayern 0-0 Bodø/Glimt trong khi kết quả thật là 5-0), và 26/36 trận không
    có người ghi bàn.

    Trả về (số trận sửa tỷ số hoặc chấm lại, số trận bổ sung chi tiết).
    """
    fixed_scores = fixed_details = regraded = 0

    for entry in history:
        date = entry["fixture"]["date"]
        home, away = entry["home_team"], entry["away_team"]
        actual = entry.get("actual") or {}

        truth = index.get(_key(date, home, away))
        if truth and (
            actual.get("home_goals") != truth["home_goals"]
            or actual.get("away_goals") != truth["away_goals"]
        ):
            print(
                f"  sửa tỷ số: {date} {home} {actual.get('home_goals')}-"
                f"{actual.get('away_goals')} {away} -> "
                f"{truth['home_goals']}-{truth['away_goals']}"
            )
            actual.update(truth)
            entry["actual"] = actual
            entry["grade"] = _grade(_stored_as_prediction(entry), truth)
            fixed_scores += 1
        elif (entry.get("grade") or {}).get("v") != GRADE_VERSION:
            # Luật chấm điểm đã đổi — chấm lại bằng chính kết quả đang lưu,
            # nếu không thì mọi trận cũ vẫn giữ điểm sai của luật trước.
            entry["grade"] = _grade(_stored_as_prediction(entry), {
                "home_goals": actual["home_goals"],
                "away_goals": actual["away_goals"],
                "result": actual["result"],
            })
            regraded += 1

        # Chỉ dựa vào details_source: mảng goals rỗng là kết quả HỢP LỆ của
        # một trận hoà 0-0, lấy nó làm dấu hiệu "chưa có dữ liệu" thì trận
        # không bàn thắng nào sẽ bị tra lại ở mọi lần chạy về sau.
        # "corners" not in actual: bản ghi lưu trước khi app biết đọc số góc.
        # Kiểm tra bằng sự CÓ MẶT của khoá chứ không phải giá trị, vì None là
        # câu trả lời hợp lệ cho trận mà ESPN không có thống kê.
        if not actual.get("details_source") or "corners" not in actual:
            detail = espn_lookup(date, home, away)
            if detail and detail.get("source"):
                actual["half_time"] = detail.get("half_time")
                actual["goals"] = detail.get("goals", [])
                actual["corners"] = detail.get("corners")
                actual["details_source"] = detail.get("source")
                entry["actual"] = actual
                fixed_details += 1

    if regraded:
        print(f"  chấm lại điểm cho {regraded} trận theo luật mới")
    return fixed_scores + regraded, fixed_details


def archive_finished(
    league: str,
    league_dir: Path,
    results: pd.DataFrame,
    espn_details: dict | None = None,
) -> tuple[list[dict], int, int]:
    """
    Chuyển các trận đã đá từ predictions.json sang history.json, đồng thời
    sửa lại những bản ghi cũ bị sai tỷ số hoặc thiếu người ghi bàn.

    Trả về (lịch sử sau khi cập nhật, số trận vừa thêm, số bản ghi được sửa).
    """
    history_file = league_dir / "history.json"
    predictions_file = league_dir / "predictions.json"

    history = load(history_file)
    seen = {_key(h["fixture"]["date"], h["home_team"], h["away_team"]) for h in history}

    if not predictions_file.exists():
        return history, 0, 0

    try:
        previous = json.loads(predictions_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return history, 0, 0

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
                # Giữ lại dự đoán góc để đối chiếu với số góc thật. Cần thiết
                # vì mô hình góc rất yếu (chỉ hơn tần suất chung ~1,5%), nên
                # người dùng phải tự kiểm chứng được thay vì tin lời app.
                "corners": (pred.get("corners") or {}).get("expected"),
            },
            "actual": {
                **actual,
                "half_time": detail.get("half_time"),
                "goals": detail.get("goals", []),
                "corners": detail.get("corners"),
                "details_source": detail.get("source"),
            },
            "grade": _grade(pred, actual),
        })
        seen.add(key)
        added += 1

    # Chạy sau khi thêm mới, để trận vừa vào cũng được soát luôn.
    fixed_scores, fixed_details = repair(history, index, espn_lookup)
    if fixed_details:
        print(f"  bổ sung chi tiết cho {fixed_details} trận đã lưu")

    history.sort(key=lambda h: h["fixture"]["date"], reverse=True)
    return history[:MAX_HISTORY], added, fixed_scores + fixed_details


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
