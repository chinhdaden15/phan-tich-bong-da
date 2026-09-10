"""
Cập nhật nhanh những thứ đổi ngay sau trận: bảng xếp hạng, lịch sử, đội hình.

VÌ SAO TÁCH RIÊNG KHỎI build.py: build.py khớp lại toàn bộ mô hình nên mất vài
phút, chỉ chạy được 4 lần/ngày. Nhưng hai thứ người dùng mở app để xem —
BẢNG XẾP HẠNG và KẾT QUẢ — thì đổi ngay khi trận kết thúc, và đợi tới 6 tiếng
là quá lâu.

Script này không khớp lại mô hình, chỉ gọi API và ghi JSON, nên chạy được mỗi
giờ mà vẫn nằm trong hạn 2.000 phút/tháng của GitHub Actions.

Ba việc:
  1. Bảng xếp hạng   — gọi thẳng API, nhanh
  2. Lịch sử         — lưu trận vừa đá xong kèm người ghi bàn
  3. Đội hình ra sân — ESPN chỉ công bố khoảng một tiếng trước giờ bóng lăn

    python pipeline/refresh_live.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import history as history_mod
from config import load_env
from model.extra_markets import player_scoring_rates, scorers_from_lineup
from sources import espn, football_data_org, results as results_src
from sources.odds_history import normalise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public" / "data"

# Chỉ xét các trận đá trong khoảng này. Đội hình chưa có trước đó, và trận đã
# bắt đầu thì không cần cập nhật nữa.
WINDOW_HOURS_AHEAD = 4.0
WINDOW_HOURS_BEHIND = 2.0

SCORER_LEAGUES = {"premier-league": ["PL"], "la-liga": ["PD"]}

# Champions League phải lấy phong độ ghi bàn từ GIẢI QUỐC NỘI của cầu thủ.
#
# Lấy từ chính C1 là vô dụng: giải mới đá một vòng nên gần như chưa ai ghi bàn,
# mọi người rơi hết về mức nền theo vị trí và các cầu thủ cùng vị trí nhận xác
# suất giống hệt nhau — Deniz Undav bằng đúng Grischa Prömel. Số bàn ở La Liga
# hay Bundesliga mới nói lên cầu thủ đó nguy hiểm đến đâu.
CL_SCORER_COMPETITIONS = ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL"]

CURRENT_SEASON = 2026


def _scorers(code: str, season: int | None = None) -> list[dict]:
    params = {"limit": 100}
    if season is not None:
        params["season"] = season
    try:
        return football_data_org._get(
            f"competitions/{code}/scorers", params,
            cache_hours=12.0 if season is None else 24 * 7,
        ).get("scorers", [])
    except Exception:
        return []


def scoring_rates(league: str) -> dict:
    """
    Tỷ lệ ghi bàn của cầu thủ, gộp mùa này và mùa trước.

    Với Champions League thì gộp qua nhiều giải quốc nội, vì cầu thủ dự C1 đến
    từ khắp châu Âu và số bàn ở giải quốc nội mới là thước đo có ý nghĩa.
    """
    codes = (CL_SCORER_COMPETITIONS if league == "champions-league"
             else SCORER_LEAGUES.get(league, []))
    if not codes:
        return {}

    current, previous = [], []
    for code in codes:
        current.extend(_scorers(code))
        previous.extend(_scorers(code, CURRENT_SEASON - 1))
    return player_scoring_rates(current, previous)


def kickoff(pred: dict) -> datetime | None:
    raw = (pred.get("fixture") or {}).get("kickoff_utc")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def refresh_league(league: str, rates: dict) -> int:
    """Gắn đội hình và người ghi bàn cho các trận sắp đá. Trả số trận cập nhật."""
    path = DATA / league / "predictions.json"
    if not path.exists():
        return 0

    try:
        preds = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0

    now = datetime.now(timezone.utc)
    updated = 0

    for pred in preds:
        ko = kickoff(pred)
        if ko is None:
            continue
        hours = (ko - now).total_seconds() / 3600
        if not (-WINDOW_HOURS_BEHIND <= hours <= WINDOW_HOURS_AHEAD):
            continue

        day = ko.date()
        event_id = espn.find_event(
            league, day, pred["home_team"], pred["away_team"], normalise_fn=normalise
        )
        if not event_id:
            continue

        lineup = espn.lineups(league, event_id)
        if not lineup.get("available"):
            continue

        eg = (pred.get("market") or {}).get("expected_goals") or pred["expected_goals"]
        sides = {}
        for side in ("home", "away"):
            data = lineup["sides"].get(side)
            if not data:
                continue
            lam = eg["home"] if side == "home" else eg["away"]
            sides[side] = {
                "team": data["team"],
                "formation": data.get("formation"),
                "starters": data["starters"],
                "scorers": scorers_from_lineup(data["starters"], rates, lam),
            }

        if sides:
            pred["lineup"] = {
                "available": True,
                "sides": sides,
                "fetched_at": now.isoformat(timespec="seconds"),
            }
            updated += 1
            print(f"  {pred['home_team']} v {pred['away_team']}: có đội hình "
                  f"({hours:+.1f}h so với giờ bóng lăn)")

    if updated:
        path.write_text(
            json.dumps(preds, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            encoding="utf-8",
        )
    return updated


def league_results(league: str):
    """Kết quả đã đá của giải, dùng để lưu lịch sử."""
    import pandas as pd

    if league == "champions-league":
        rows = []
        for m in football_data_org.season_matches("champions-league", CURRENT_SEASON):
            if m["home_goals"] is None:
                continue
            rows.append({
                "date": pd.Timestamp(m["date"]),
                "home_team": m["home_team"], "away_team": m["away_team"],
                "home_goals": m["home_goals"], "away_goals": m["away_goals"],
                "result": ("H" if m["home_goals"] > m["away_goals"]
                           else "A" if m["home_goals"] < m["away_goals"] else "D"),
            })
        return pd.DataFrame(rows, columns=[
            "date", "home_team", "away_team", "home_goals", "away_goals", "result"])

    return results_src.load_with_recent(
        [league], list(range(CURRENT_SEASON - 1, CURRENT_SEASON + 1)), verbose=False
    )


def refresh_standings(league: str) -> bool:
    """Ghi lại bảng xếp hạng nếu có thay đổi. Trả True khi có cập nhật."""
    path = DATA / league / "standings.json"
    if not path.exists():
        return False

    try:
        rows = football_data_org.standings(league)
    except Exception:
        return False
    if not rows:
        return False

    # Bảng của các giải quốc nội do build.py tự tính từ kết quả và có thêm
    # trường form; không ghi đè bằng bản API vốn thiếu trường đó. Chỉ cập nhật
    # phần số liệu, giữ nguyên form đã có.
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        current = []
    form_by_team = {normalise(r["team"]): r.get("form", []) for r in current}
    for row in rows:
        row["form"] = form_by_team.get(normalise(row["team"]), [])

    if current and json.dumps(current, sort_keys=True) == json.dumps(rows, sort_keys=True):
        return False

    path.write_text(
        json.dumps(rows, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return True


def refresh_history(league: str) -> int:
    """Lưu các trận vừa đá xong vào lịch sử. Trả số trận vừa thêm."""
    league_dir = DATA / league
    if not (league_dir / "predictions.json").exists():
        return 0
    try:
        results = league_results(league)
    except Exception as exc:
        print(f"  {league}: bỏ qua lịch sử ({type(exc).__name__})")
        return 0
    if results.empty:
        return 0

    details = espn.details_for_dates(league, espn.recent_days(7))
    archived, added = history_mod.archive_finished(
        league, league_dir, results, espn_details=details
    )
    if added:
        (league_dir / "history.json").write_text(
            json.dumps(archived, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")),
            encoding="utf-8",
        )
    return added


def main() -> None:
    load_env()
    lineups = tables = archived = 0

    for league in ("premier-league", "la-liga", "champions-league"):
        if refresh_standings(league):
            tables += 1
            print(f"  {league}: bảng xếp hạng có thay đổi")

        added = refresh_history(league)
        if added:
            archived += added
            print(f"  {league}: lưu {added} trận vừa đá xong")

        lineups += refresh_league(league, scoring_rates(league))

    parts = []
    if tables:
        parts.append(f"{tables} bảng xếp hạng")
    if archived:
        parts.append(f"{archived} trận vào lịch sử")
    if lineups:
        parts.append(f"{lineups} đội hình")
    print("Đã cập nhật: " + ", ".join(parts) if parts else "Không có gì thay đổi")


if __name__ == "__main__":
    main()
