"""
Nạp kết quả lịch sử và kèo đóng cửa từ football-data.co.uk.

Đây là nguồn quan trọng nhất của toàn dự án: miễn phí không giới hạn,
có hơn 20 mùa giải, và quan trọng nhất là kèm **kèo đóng cửa** của nhà cái —
mốc so sánh duy nhất cho biết mô hình thực sự tốt hay chỉ trông có vẻ tốt.

Dữ liệu cho dùng cá nhân miễn phí; hãy tải có mức độ và luôn dùng cache.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Mã giải của football-data.co.uk
DIVISIONS = {
    "premier-league": "E0",
    "championship": "E1",
    "la-liga": "SP1",
    "serie-a": "I1",
    "bundesliga": "D1",
    "ligue-1": "F1",
    "eredivisie": "N1",
    "primeira-liga": "P1",
}

# Cột kèo đóng cửa, xếp theo độ "sắc" giảm dần.
# Pinnacle là nhà cái sắc nhất; Avg là trung bình thị trường; B365 là dự phòng.
CLOSING_ODDS_SETS = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("B365H", "B365D", "B365A"),
]

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "historical"

# Nguồn trả 503 cho User-Agent mặc định của requests, nên phải khai báo tường minh.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}


def season_code(start_year: int) -> str:
    """2024 -> '2425' (mùa 2024-25), đúng quy ước đặt tên của nguồn."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_season(
    league: str, start_year: int, use_cache: bool = True, timeout: int = 30
) -> pd.DataFrame:
    """Tải một mùa của một giải. Đã tải rồi thì đọc từ đĩa."""
    if league not in DIVISIONS:
        raise KeyError(f"Giải không được hỗ trợ: {league!r}. Có: {sorted(DIVISIONS)}")

    div = DIVISIONS[league]
    code = season_code(start_year)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{div}_{code}.csv"

    if use_cache and cache_file.exists() and cache_file.stat().st_size > 0:
        raw = cache_file.read_bytes()
    else:
        url = f"{BASE_URL}/{code}/{div}.csv"
        # timeout của requests là thời gian CHỜ GIỮA HAI GÓI, không phải tổng
        # thời gian. Máy chủ nhỏ giọt vài byte một sẽ liên tục làm mới bộ đếm
        # và lời gọi treo vô hạn — đã gặp đúng chuyện này, tiến trình đứng im
        # hàng chục phút ở 0% CPU. Đọc theo luồng rồi tự chặn tổng thời gian.
        deadline = time.monotonic() + max(timeout, 1) * 2
        response = requests.get(url, timeout=timeout, headers=HEADERS, stream=True)
        response.raise_for_status()
        chunks = []
        for chunk in response.iter_content(chunk_size=65536):
            chunks.append(chunk)
            if time.monotonic() > deadline:
                raise requests.exceptions.Timeout(
                    f"quá {max(timeout, 1) * 2}s khi tải {div}_{code}"
                )
        raw = b"".join(chunks)
        cache_file.write_bytes(raw)

    # Nguồn dùng mã hoá không nhất quán giữa các mùa; latin-1 đọc được tất cả.
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    return _clean(df, league, start_year)


def _clean(df: pd.DataFrame, league: str, start_year: int) -> pd.DataFrame:
    """Chuẩn hoá cột, ép kiểu, và gắn kèo đóng cửa tốt nhất hiện có."""
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV thiếu cột bắt buộc: {missing}")

    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()

    # dayfirst=True: nguồn dùng định dạng ngày kiểu Anh (dd/mm/yy).
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    out = pd.DataFrame({
        "date": df["date"],
        "league": league,
        "season": start_year,
        "home_team": df["HomeTeam"].astype(str).str.strip(),
        "away_team": df["AwayTeam"].astype(str).str.strip(),
        "home_goals": df["FTHG"].astype(int),
        "away_goals": df["FTAG"].astype(int),
    })
    out["result"] = out.apply(
        lambda r: "H" if r.home_goals > r.away_goals
        else ("A" if r.home_goals < r.away_goals else "D"),
        axis=1,
    )

    # Cược chấp châu Á: mốc chấp và giá hai cửa.
    #
    # Cột "C" là giá đóng cửa (sát giờ bóng lăn) nên sắc hơn giá mở, ưu tiên
    # dùng trước. Mốc chấp ghi dưới góc nhìn ĐỘI NHÀ: -1.5 nghĩa là đội nhà
    # chấp 1.5 trái, phải thắng cách biệt từ 2 bàn mới ăn.
    for line_col, home_col, away_col in (("AHCh", "AvgCAHH", "AvgCAHA"),
                                         ("AHh", "AvgAHH", "AvgAHA"),
                                         ("AHh", "B365AHH", "B365AHA")):
        if all(c in df.columns for c in (line_col, home_col, away_col)):
            trio = df[[line_col, home_col, away_col]].apply(pd.to_numeric, errors="coerce")
            if trio.notna().all(axis=1).mean() > 0.5:
                out["ah_line"] = trio[line_col].values
                out["ah_odds_home"] = trio[home_col].values
                out["ah_odds_away"] = trio[away_col].values
                break
    else:
        out["ah_line"] = out["ah_odds_home"] = out["ah_odds_away"] = pd.NA

    # Kèo tài/xỉu 2.5, cần để ghim chiều tổng bàn.
    for over_col, under_col in (("AvgC>2.5", "AvgC<2.5"), ("Avg>2.5", "Avg<2.5"),
                                ("B365>2.5", "B365<2.5")):
        if over_col in df.columns and under_col in df.columns:
            pair = df[[over_col, under_col]].apply(pd.to_numeric, errors="coerce")
            if pair.notna().all(axis=1).mean() > 0.5:
                out["odds_over25"] = pair[over_col].values
                out["odds_under25"] = pair[under_col].values
                break
    else:
        out["odds_over25"] = out["odds_under25"] = pd.NA

    # Lấy bộ kèo đóng cửa sắc nhất mà mùa đó có
    for cols in CLOSING_ODDS_SETS:
        if all(c in df.columns for c in cols):
            odds = df[list(cols)].apply(pd.to_numeric, errors="coerce")
            if odds.notna().all(axis=1).mean() > 0.5:
                out["odds_home"] = odds[cols[0]].values
                out["odds_draw"] = odds[cols[1]].values
                out["odds_away"] = odds[cols[2]].values
                out["odds_source"] = cols[0]
                break
    else:
        out["odds_home"] = out["odds_draw"] = out["odds_away"] = pd.NA
        out["odds_source"] = None

    return out.sort_values("date").reset_index(drop=True)


def load_leagues(
    leagues: list[str], seasons: list[int], use_cache: bool = True
) -> pd.DataFrame:
    """Nạp nhiều giải, nhiều mùa vào một bảng duy nhất."""
    frames = []
    for league in leagues:
        for year in seasons:
            try:
                frames.append(download_season(league, year, use_cache=use_cache))
            except Exception as exc:
                # Mùa hiện tại có thể chưa tồn tại; bỏ qua và đi tiếp.
                print(f"  bỏ qua {league} {year}: {type(exc).__name__}: {exc}")
    if not frames:
        raise RuntimeError("Không nạp được mùa nào")
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
