"""
Cập nhật nhanh đội hình ra sân và dự đoán người ghi bàn cho các trận sắp đá.

VÌ SAO TÁCH RIÊNG KHỎI build.py: ESPN chỉ công bố đội hình khoảng MỘT TIẾNG
trước giờ bóng lăn. build.py chạy 4 lần/ngày nên gần như luôn bỏ lỡ khung giờ
đó. Script này nhẹ (không khớp lại mô hình, chỉ đọc và ghi JSON) nên chạy được
mỗi giờ mà vẫn nằm trong hạn 2.000 phút/tháng của GitHub Actions.

    python pipeline/refresh_lineups.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_env
from model.extra_markets import player_scoring_rates, scorers_from_lineup
from sources import espn, football_data_org
from sources.odds_history import normalise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public" / "data"

# Chỉ xét các trận đá trong khoảng này. Đội hình chưa có trước đó, và trận đã
# bắt đầu thì không cần cập nhật nữa.
WINDOW_HOURS_AHEAD = 4.0
WINDOW_HOURS_BEHIND = 2.0

SCORER_LEAGUES = {"premier-league": "PL", "la-liga": "PD", "champions-league": "CL"}
CURRENT_SEASON = 2026


def scoring_rates(league: str) -> dict:
    """Tỷ lệ ghi bàn của cầu thủ, gộp mùa này và mùa trước."""
    code = SCORER_LEAGUES.get(league)
    if not code:
        return {}
    try:
        current = football_data_org._get(
            f"competitions/{code}/scorers", {"limit": 100}, cache_hours=12.0
        ).get("scorers", [])
    except Exception:
        current = []
    try:
        previous = football_data_org._get(
            f"competitions/{code}/scorers",
            {"limit": 100, "season": CURRENT_SEASON - 1},
            cache_hours=24 * 7,
        ).get("scorers", [])
    except Exception:
        previous = []
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


def main() -> None:
    load_env()
    total = 0
    for league in ("premier-league", "la-liga", "champions-league"):
        rates = scoring_rates(league)
        total += refresh_league(league, rates)
    print(f"Cập nhật đội hình cho {total} trận" if total
          else "Chưa trận nào công bố đội hình")


if __name__ == "__main__":
    main()
