"""
Nạp kết quả lịch sử từ openfootball/football.json trên GitHub.

Chọn nguồn này làm xương sống cho dữ liệu lịch sử vì ba lý do:
  - Hoàn toàn miễn phí, không cần API key, không giới hạn tốc độ
  - Phục vụ qua hạ tầng GitHub nên gần như không bao giờ sập
    (football-data.co.uk đã 503 ngay lần thử đầu tiên khi dựng dự án này)
  - Có hơn 20 mùa cho các giải hàng đầu châu Âu

Hạn chế: không có kèo nhà cái và không có Cúp C1. Kèo lấy bổ sung từ
football-data.co.uk khi nguồn đó sống; C1 lấy từ football-data.org.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

# Mã giải của openfootball
LEAGUES = {
    "premier-league": "en.1",
    "championship": "en.2",
    "la-liga": "es.1",
    "serie-a": "it.1",
    "bundesliga": "de.1",
    "ligue-1": "fr.1",
    "eredivisie": "nl.1",
    "primeira-liga": "pt.1",
    # Các giải dưới đây thêm vào chủ yếu để mô hình Champions League có dữ
    # liệu quốc nội cho những CLB như Celtic, Galatasaray, Club Brugge,
    # Olympiakos, Red Bull Salzburg — nếu thiếu, mô hình không biết gì về họ
    # ngoài dăm trận cúp.
    "scottish-premiership": "sco.1",
    "turkish-super-lig": "tr.1",
    "belgian-pro-league": "be.1",
    "greek-super-league": "gr.1",
    "austrian-bundesliga": "at.1",
}

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "openfootball"

# Mùa đã kết thúc thì kết quả không bao giờ đổi -> cache vĩnh viễn.
# Mùa đang diễn ra đổi sau mỗi vòng đấu -> phải có hạn dùng, nếu không
# pipeline chạy hằng ngày vẫn đọc mãi bản tải lần đầu và bảng xếp hạng
# đứng yên suốt mùa. Đây từng là lỗi thật: file mùa 2026-27 nằm nguyên
# trong cache 10 tiếng trong khi các trận mới đã đá xong.
CURRENT_SEASON_TTL_HOURS = 3.0


def season_slug(start_year: int) -> str:
    """2024 -> '2024-25'"""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _is_current_season(start_year: int, today: date | None = None) -> bool:
    """
    Mùa giải châu Âu chạy từ tháng 7 năm N đến tháng 6 năm N+1.

    Dùng để biết file cache nào còn thay đổi và file nào đã cố định.
    """
    d = today or date.today()
    running = d.year if d.month >= 7 else d.year - 1
    return start_year >= running


def _load_payload(league: str, start_year: int, use_cache: bool, timeout: int) -> dict:
    """Đọc dữ liệu mùa, tôn trọng hạn dùng cache của mùa đang diễn ra."""
    code = LEAGUES[league]
    slug = season_slug(start_year)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{code}_{slug}.json"

    fresh = False
    if use_cache and cache_file.exists() and cache_file.stat().st_size > 0:
        if not _is_current_season(start_year):
            fresh = True          # mùa đã xong, không bao giờ đổi
        else:
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            fresh = age_hours < CURRENT_SEASON_TTL_HOURS

    if fresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = f"{BASE_URL}/{slug}/{code}.json"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # Nguồn lỗi mà vẫn còn bản cũ thì dùng tạm, còn hơn là hỏng cả lần build.
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        raise

    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def download_season(
    league: str, start_year: int, use_cache: bool = True, timeout: int = 30
) -> pd.DataFrame:
    """Tải một mùa của một giải, có cache trên đĩa."""
    if league not in LEAGUES:
        raise KeyError(f"Giải không được hỗ trợ: {league!r}. Có: {sorted(LEAGUES)}")

    payload = _load_payload(league, start_year, use_cache, timeout)
    return _to_frame(payload, league, start_year)


def _full_time_score(score) -> tuple[int, int] | None:
    """
    Rút tỷ số toàn trận, chấp nhận mọi biến thể mà nguồn cộng đồng dùng.

    Dữ liệu không đồng nhất giữa các mùa: đa số trận có score là dict
    {"ft": [x, y], "ht": [...]}, nhưng một số trận (27/380 ở mùa 2025-26)
    lại để score thẳng thành list [x, y]. Trả None nếu trận chưa đá.
    """
    if isinstance(score, dict):
        full_time = score.get("ft")
    elif isinstance(score, (list, tuple)):
        full_time = score
    else:
        return None

    if not isinstance(full_time, (list, tuple)) or len(full_time) != 2:
        return None
    if full_time[0] is None or full_time[1] is None:
        return None
    try:
        return int(full_time[0]), int(full_time[1])
    except (TypeError, ValueError):
        return None


def _to_frame(payload: dict, league: str, start_year: int) -> pd.DataFrame:
    """Trải danh sách trận thành bảng phẳng, bỏ các trận chưa đá."""
    rows = []
    for match in payload.get("matches", []):
        full_time = _full_time_score(match.get("score"))
        # Trận chưa diễn ra thì không có tỷ số toàn trận — bỏ qua khi khớp mô hình.
        if full_time is None:
            continue

        home_goals, away_goals = full_time
        rows.append({
            "date": match.get("date"),
            "league": league,
            "season": start_year,
            "round": match.get("round"),
            "home_team": str(match.get("team1", "")).strip(),
            "away_team": str(match.get("team2", "")).strip(),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": "H" if home_goals > away_goals else ("A" if home_goals < away_goals else "D"),
        })

    if not rows:
        return pd.DataFrame(
            columns=["date", "league", "season", "round", "home_team",
                     "away_team", "home_goals", "away_goals", "result"]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df = df[(df["home_team"] != "") & (df["away_team"] != "")]
    return df.sort_values("date").reset_index(drop=True)


def load(leagues: list[str], seasons: list[int], use_cache: bool = True) -> pd.DataFrame:
    """Nạp nhiều giải, nhiều mùa vào một bảng."""
    frames = []
    for league in leagues:
        for year in seasons:
            try:
                df = download_season(league, year, use_cache=use_cache)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"  bỏ qua {league} {season_slug(year)}: {type(exc).__name__}")
    if not frames:
        raise RuntimeError("Không nạp được mùa nào")
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def upcoming_fixtures(
    league: str, start_year: int, use_cache: bool = True, timeout: int = 30
) -> pd.DataFrame:
    """
    Lịch các trận CHƯA đá của mùa hiện tại.

    File mùa của openfootball chứa toàn bộ lịch cả mùa ngay từ đầu, nên
    trận chưa diễn ra chỉ đơn giản là trận không có tỷ số. Nhờ vậy app xem
    được lịch thi đấu mà không cần bất kỳ API key nào — football-data.org
    chỉ còn cần cho Champions League.
    """
    if league not in LEAGUES:
        raise KeyError(f"Giải không được hỗ trợ: {league!r}")

    payload = _load_payload(league, start_year, use_cache, timeout)

    rows = []
    for match in payload.get("matches", []):
        if _full_time_score(match.get("score")) is not None:
            continue   # đã đá rồi
        home = str(match.get("team1", "")).strip()
        away = str(match.get("team2", "")).strip()
        if not home or not away:
            continue
        rows.append({
            "date": match.get("date"),
            "time": match.get("time"),
            "league": league,
            "season": start_year,
            "round": match.get("round"),
            "home_team": home,
            "away_team": away,
        })

    if not rows:
        return pd.DataFrame(
            columns=["date", "time", "league", "season", "round", "home_team", "away_team"]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
