"""
Client cho football-data.org v4 — lịch thi đấu, bảng xếp hạng, và Cúp C1.

Vai trò trong dự án: openfootball lo dữ liệu lịch sử, còn nguồn này lo
**lịch sắp tới** và **Champions League** (openfootball không có C1).

Gói miễn phí: 12 giải, 10 request/phút, cần API key miễn phí đăng ký tại
https://www.football-data.org/client/register

Vì có giới hạn tốc độ, mọi lời gọi đều đi qua cache trên đĩa và bộ điều tốc.
Pipeline chạy trên GitHub Actions nên chỉ tốn vài chục request mỗi ngày.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.football-data.org/v4"

# Mã giải của football-data.org (gói miễn phí)
COMPETITIONS = {
    "premier-league": "PL",
    "la-liga": "PD",
    "champions-league": "CL",
    "serie-a": "SA",
    "bundesliga": "BL1",
    "ligue-1": "FL1",
    "eredivisie": "DED",
    "primeira-liga": "PPL",
    "championship": "ELC",
}

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "football_data_org"

# Gói miễn phí cho 10 request/phút. Giãn 6.5s giữa các lần gọi là an toàn
# tuyệt đối mà vẫn đủ nhanh cho pipeline chạy nền.
MIN_INTERVAL_SECONDS = 6.5

_last_call_at = 0.0


def _is_current_season(season: int) -> bool:
    """Mùa giải châu Âu chạy từ tháng 7 năm N đến tháng 6 năm N+1."""
    today = datetime.now(timezone.utc).date()
    running = today.year if today.month >= 7 else today.year - 1
    return season >= running


class MissingApiKey(RuntimeError):
    """Nêu rõ cách khắc phục thay vì để KeyError khó hiểu."""

    def __init__(self) -> None:
        super().__init__(
            "Thiếu FOOTBALL_DATA_ORG_KEY. Đăng ký key miễn phí tại "
            "https://www.football-data.org/client/register rồi đặt vào file .env "
            "hoặc biến môi trường."
        )


def _api_key() -> str:
    key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
    if not key:
        raise MissingApiKey()
    return key


def _throttle() -> None:
    """Giãn cách các lời gọi để không bao giờ chạm trần 10 req/phút."""
    global _last_call_at
    wait = MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _get(path: str, params: dict | None = None, cache_hours: float = 6.0) -> dict:
    """GET có cache theo thời gian sống và có điều tốc."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = path.strip("/").replace("/", "_")
    if params:
        slug += "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    cache_file = CACHE_DIR / f"{slug}.json"

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_hours:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    _throttle()
    response = requests.get(
        f"{BASE_URL}/{path.lstrip('/')}",
        headers={"X-Auth-Token": _api_key()},
        params=params or {},
        timeout=30,
    )

    if response.status_code == 429:
        # Chạm trần tốc độ: chờ rồi thử lại một lần.
        time.sleep(61)
        _throttle()
        response = requests.get(
            f"{BASE_URL}/{path.lstrip('/')}",
            headers={"X-Auth-Token": _api_key()},
            params=params or {},
            timeout=30,
        )

    if response.status_code == 403:
        raise PermissionError(
            f"football-data.org từ chối {path} — giải này không nằm trong gói miễn phí"
        )
    if response.status_code == 404:
        # Thường gặp với bảng xếp hạng của giải chưa đá trận nào trong mùa.
        raise LookupError(f"football-data.org chưa có dữ liệu cho {path}")
    response.raise_for_status()

    payload = response.json()
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def upcoming_matches(league: str, days_ahead: int = 14) -> list[dict]:
    """Lịch thi đấu sắp tới của một giải."""
    if league not in COMPETITIONS:
        raise KeyError(f"Giải không được hỗ trợ: {league!r}")

    today = datetime.now(timezone.utc).date()
    payload = _get(
        f"competitions/{COMPETITIONS[league]}/matches",
        {
            "dateFrom": today.isoformat(),
            "dateTo": (today + timedelta(days=days_ahead)).isoformat(),
        },
        cache_hours=6.0,
    )

    out = []
    for m in payload.get("matches", []):
        out.append({
            "id": m.get("id"),
            "league": league,
            "utc_date": m.get("utcDate"),
            "status": m.get("status"),
            "matchday": m.get("matchday"),
            "stage": m.get("stage"),
            "home_team": (m.get("homeTeam") or {}).get("name"),
            "home_crest": (m.get("homeTeam") or {}).get("crest"),
            "away_team": (m.get("awayTeam") or {}).get("name"),
            "away_crest": (m.get("awayTeam") or {}).get("crest"),
        })
    return [m for m in out if m["home_team"] and m["away_team"]]


def standings(league: str) -> list[dict]:
    """Bảng xếp hạng hiện tại — dùng cho cả hiển thị lẫn hệ số động lực."""
    if league not in COMPETITIONS:
        raise KeyError(f"Giải không được hỗ trợ: {league!r}")

    # Cache RẤT ngắn: bảng xếp hạng là thứ đổi ngay sau mỗi trận, và đây chính
    # là dữ liệu người dùng mở app để xem. Trước đây để 12 giờ, hệ quả là build
    # lúc 21:38 vẫn đọc bản cache từ 10:26 và bỏ lỡ toàn bộ vòng đấu buổi tối.
    # Giới hạn của gói miễn phí là 10 request/PHÚT chứ không theo ngày, nên gọi
    # lại thường xuyên hoàn toàn thoải mái.
    payload = _get(f"competitions/{COMPETITIONS[league]}/standings", cache_hours=0.5)

    for table in payload.get("standings", []):
        # Bỏ qua bảng sân nhà/sân khách, chỉ lấy bảng tổng.
        if table.get("type") != "TOTAL":
            continue
        rows = []
        for r in table.get("table", []):
            team = r.get("team") or {}
            rows.append({
                "position": r.get("position"),
                "team": team.get("name"),
                "crest": team.get("crest"),
                "played": r.get("playedGames"),
                "won": r.get("won"),
                "draw": r.get("draw"),
                "lost": r.get("lost"),
                "points": r.get("points"),
                "goals_for": r.get("goalsFor"),
                "goals_against": r.get("goalsAgainst"),
                "goal_difference": r.get("goalDifference"),
            })
        if rows:
            return rows
    return []


def recent_results(league: str, days_back: int = 400) -> list[dict]:
    """
    Kết quả đã đá — dùng cho C1, nơi openfootball không có dữ liệu.

    Gói miễn phí giới hạn chiều sâu lịch sử, nên với các giải quốc nội
    hãy ưu tiên openfootball.
    """
    if league not in COMPETITIONS:
        raise KeyError(f"Giải không được hỗ trợ: {league!r}")

    today = datetime.now(timezone.utc).date()
    payload = _get(
        f"competitions/{COMPETITIONS[league]}/matches",
        {
            "dateFrom": (today - timedelta(days=days_back)).isoformat(),
            "dateTo": today.isoformat(),
            "status": "FINISHED",
        },
        # Ngắn, vì đây chính là thứ phải mới sau mỗi vòng đấu. Giới hạn của
        # gói miễn phí là 10 request/phút chứ không giới hạn theo ngày, nên
        # gọi lại vài giờ một lần hoàn toàn thoải mái.
        cache_hours=2.0,
    )

    out = []
    for m in payload.get("matches", []):
        full_time = ((m.get("score") or {}).get("fullTime") or {})
        home, away = full_time.get("home"), full_time.get("away")
        if home is None or away is None:
            continue
        out.append({
            "date": (m.get("utcDate") or "")[:10],
            "league": league,
            "stage": m.get("stage"),
            "home_team": (m.get("homeTeam") or {}).get("name"),
            "away_team": (m.get("awayTeam") or {}).get("name"),
            "home_goals": int(home),
            "away_goals": int(away),
            "result": "H" if home > away else ("A" if home < away else "D"),
        })
    return [m for m in out if m["home_team"] and m["away_team"]]


def season_matches(league: str, season: int, finished_only: bool = True) -> list[dict]:
    """
    Toàn bộ trận của một mùa cụ thể.

    Dùng cho Champions League, nơi openfootball không có dữ liệu. Tham số
    season là năm bắt đầu mùa giải (2025 nghĩa là mùa 2025-26).
    """
    if league not in COMPETITIONS:
        raise KeyError(f"Giải không được hỗ trợ: {league!r}")

    # Mùa đã kết thúc thì kết quả cố định vĩnh viễn; mùa đang diễn ra đổi sau
    # mỗi vòng nên phải hỏi lại thường xuyên.
    payload = _get(
        f"competitions/{COMPETITIONS[league]}/matches",
        {"season": season},
        cache_hours=0.5 if _is_current_season(season) else 24 * 30,
    )

    out = []
    for m in payload.get("matches", []):
        full_time = ((m.get("score") or {}).get("fullTime") or {})
        home, away = full_time.get("home"), full_time.get("away")
        played = home is not None and away is not None

        if finished_only and not played:
            continue

        home_team = (m.get("homeTeam") or {}).get("name")
        away_team = (m.get("awayTeam") or {}).get("name")
        if not home_team or not away_team:
            continue

        out.append({
            "id": m.get("id"),
            "date": (m.get("utcDate") or "")[:10],
            "kickoff_utc": m.get("utcDate"),
            "league": league,
            "season": season,
            "stage": m.get("stage"),
            "matchday": m.get("matchday"),
            "status": m.get("status"),
            "home_team": home_team,
            "away_team": away_team,
            "home_crest": (m.get("homeTeam") or {}).get("crest"),
            "away_crest": (m.get("awayTeam") or {}).get("crest"),
            "home_goals": int(home) if played else None,
            "away_goals": int(away) if played else None,
        })
    return out


def squad(team_id: int) -> list[dict]:
    """
    Đội hình đăng ký của một CLB: tên và vị trí từng cầu thủ.

    Có sẵn trên gói miễn phí và đổi rất chậm (chỉ khi chuyển nhượng), nên cache
    một tuần. Đây là nguồn cho phép ước lượng người ghi bàn NGAY khi có lịch,
    thay vì phải đợi đội hình ra sân công bố sát giờ bóng lăn.
    """
    payload = _get(f"teams/{team_id}", cache_hours=24 * 7)
    out = []
    for player in payload.get("squad") or []:
        name = player.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "position": player.get("position"),
            "shirt": player.get("shirtNumber"),
        })
    return out


def competition_teams(league: str) -> list[dict]:
    """Danh sách CLB của một giải, kèm mã đội để tra đội hình đăng ký."""
    if league not in COMPETITIONS:
        raise KeyError(f"Giải không được hỗ trợ: {league!r}")
    payload = _get(f"competitions/{COMPETITIONS[league]}/teams", cache_hours=24 * 7)
    return [
        {"id": t.get("id"), "name": t.get("name"), "short": t.get("shortName")}
        for t in payload.get("teams", []) if t.get("id") and t.get("name")
    ]
