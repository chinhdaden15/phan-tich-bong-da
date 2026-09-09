"""
Chi tiết trận đấu từ API công khai của ESPN: người ghi bàn, kiến tạo, thẻ phạt.

Vì sao dùng ESPN: football-data.org gói miễn phí KHÔNG trả về mảng `goals`
(đó là tính năng gói trả phí), còn API-Football gói Free chặn mùa hiện tại.
ESPN có endpoint công khai không cần key, trả đủ người ghi bàn kèm phút và
người kiến tạo.

Đây là API không có tài liệu chính thức nên có thể đổi bất cứ lúc nào. Mọi lời
gọi đều bọc try/except và thất bại chỉ làm mất phần người ghi bàn, không làm
hỏng cả lần build.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Mã giải của ESPN
LEAGUE_CODES = {
    "premier-league": "eng.1",
    "la-liga": "esp.1",
    "champions-league": "uefa.champions",
    "serie-a": "ita.1",
    "bundesliga": "ger.1",
    "ligue-1": "fra.1",
}

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "espn"
CACHE_HOURS = 6.0

# ESPN chặn user-agent mặc định của một số thư viện.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _cached_get(url: str, slug: str, cache_hours: float = CACHE_HOURS) -> dict | None:
    """GET có cache trên đĩa. Trả None nếu hỏng — người gọi tự xử lý."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{slug}.json"

    if cache_file.exists():
        age = (time.time() - cache_file.stat().st_mtime) / 3600
        if age < cache_hours:
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    try:
        response = requests.get(url, timeout=25, headers=HEADERS)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # Còn bản cũ thì dùng tạm; không thì báo không có.
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def scoreboard(league: str, day: date) -> list[dict]:
    """Danh sách trận của một ngày, kèm id sự kiện để tra chi tiết."""
    code = LEAGUE_CODES.get(league)
    if not code:
        return []

    stamp = day.strftime("%Y%m%d")
    payload = _cached_get(
        f"{BASE}/{code}/scoreboard?dates={stamp}", f"{code}_{stamp}"
    )
    if not payload:
        return []

    out = []
    for event in payload.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            continue
        competitors = comps[0].get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        status = ((comps[0].get("status") or {}).get("type") or {})
        out.append({
            "event_id": event.get("id"),
            "date": day.isoformat(),
            "home_team": (home.get("team") or {}).get("displayName"),
            "away_team": (away.get("team") or {}).get("displayName"),
            "home_goals": _to_int(home.get("score")),
            "away_goals": _to_int(away.get("score")),
            "completed": bool(status.get("completed")),
        })
    return out


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def match_details(league: str, event_id: str) -> dict:
    """
    Người ghi bàn, kiến tạo và tỷ số hiệp 1 của một trận.

    Tỷ số hiệp 1 suy từ các bàn có period.number == 1, vì ESPN không trả
    riêng trường tỷ số hiệp một trong endpoint này.
    """
    code = LEAGUE_CODES.get(league)
    if not code or not event_id:
        return {}

    payload = _cached_get(
        f"{BASE}/{code}/summary?event={event_id}", f"{code}_sum_{event_id}",
        cache_hours=24 * 30,   # trận đã đá thì chi tiết không đổi nữa
    )
    if not payload:
        return {}

    header = payload.get("header") or {}
    comps = (header.get("competitions") or [{}])[0]
    competitors = comps.get("competitors") or []
    team_side = {}
    for c in competitors:
        tid = (c.get("team") or {}).get("id")
        if tid:
            team_side[tid] = c.get("homeAway")

    goals, ht_home, ht_away = [], 0, 0
    for event in payload.get("keyEvents", []):
        type_text = ((event.get("type") or {}).get("text") or "")
        if not event.get("scoringPlay") and "goal" not in type_text.lower():
            continue
        if not event.get("scoringPlay"):
            continue

        participants = event.get("participants") or []
        scorer = ((participants[0] or {}).get("athlete") or {}).get("displayName") if participants else None
        assist = (
            ((participants[1] or {}).get("athlete") or {}).get("displayName")
            if len(participants) > 1 else None
        )
        team = event.get("team") or {}
        side = team_side.get(team.get("id"))

        goals.append({
            "minute": (event.get("clock") or {}).get("displayValue"),
            "period": (event.get("period") or {}).get("number"),
            "scorer": scorer,
            "assist": assist,
            "team": team.get("displayName"),
            "side": side,
            "type": type_text,
        })

        if (event.get("period") or {}).get("number") == 1:
            if side == "home":
                ht_home += 1
            elif side == "away":
                ht_away += 1

    goals.sort(key=lambda g: _minute_value(g.get("minute")))
    return {
        "goals": goals,
        "half_time": {"home": ht_home, "away": ht_away} if goals else None,
        "source": "espn",
    }


def _minute_value(display: str | None) -> float:
    """'46'' -> 46.0, '45+2'' -> 45.2. Dùng để sắp xếp bàn thắng theo thứ tự."""
    if not display:
        return 999.0
    text = str(display).replace("'", "").strip()
    if "+" in text:
        base, _, extra = text.partition("+")
        try:
            return float(base) + float(extra) / 10.0
        except ValueError:
            return 999.0
    try:
        return float(text)
    except ValueError:
        return 999.0


def details_for_dates(league: str, days: list[date]) -> dict[tuple, dict]:
    """
    Tra chi tiết mọi trận đã đá trong các ngày cho trước.

    Trả về dict khoá (ngày, tên đội nhà, tên đội khách) -> chi tiết, để tầng
    trên tự ghép với dữ liệu của mình bằng bộ chuẩn hoá tên riêng.
    """
    out = {}
    for day in days:
        for match in scoreboard(league, day):
            if not match["completed"]:
                continue
            detail = match_details(league, match["event_id"])
            if not detail:
                continue
            key = (match["date"], match["home_team"], match["away_team"])
            out[key] = {**detail, "espn_score": (match["home_goals"], match["away_goals"])}
    return out


def recent_days(back: int = 10) -> list[date]:
    """Các ngày gần đây, dùng để quét trận vừa đá."""
    today = datetime.utcnow().date()
    return [today - timedelta(days=i) for i in range(back + 1)]
