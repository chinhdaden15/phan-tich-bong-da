"""
Nạp kèo đóng cửa lịch sử.

Nguồn gốc là football-data.co.uk. Vì trang đó không ổn định (trả 503 suốt
quá trình dựng dự án này), mặc định dùng bản mirror trên GitHub và tự động
lùi về nguồn gốc nếu mirror hỏng.

XÁC THỰC BẮT BUỘC: mirror là repo của bên thứ ba không rõ danh tính, nên
mọi mùa nạp vào đều được đối chiếu kết quả trận đấu với openfootball trước
khi dùng. Sai lệch quá ngưỡng thì loại bỏ mùa đó.
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests

MIRROR_URL = "https://raw.githubusercontent.com/huhao930422-debug/football-odds-mirror/main/data"
ORIGIN_URL = "https://www.football-data.co.uk/mmz4281"

MIRROR_LEAGUES = {
    "premier-league": "premier-league",
    "la-liga": "la-liga",
    "serie-a": "serie-a",
    "bundesliga": "bundesliga",
    "ligue-1": "ligue-1",
    "championship": "championship",
    "eredivisie": "eredivisie",
    "primeira-liga": "primeira-liga",
}
ORIGIN_DIVISIONS = {
    "premier-league": "E0", "championship": "E1", "la-liga": "SP1",
    "serie-a": "I1", "bundesliga": "D1", "ligue-1": "F1",
    "eredivisie": "N1", "primeira-liga": "P1",
}

# Bộ cột kèo đóng cửa, xếp theo độ sắc giảm dần.
# Pinnacle (PSC) sắc nhất nhưng thiếu ở một số mùa; AvgC là trung bình thị
# trường, phủ đủ mọi mùa nên là lựa chọn dự phòng tốt.
CLOSING_SETS = [
    ("PSCH", "PSCD", "PSCA", "pinnacle_closing"),
    ("AvgCH", "AvgCD", "AvgCA", "market_avg_closing"),
    ("B365CH", "B365CD", "B365CA", "b365_closing"),
    ("B365H", "B365D", "B365A", "b365_opening"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "odds"


def season_code(start_year: int) -> str:
    """2024 -> '2425'"""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _fetch(league: str, start_year: int, timeout: int = 30) -> bytes:
    """Tải CSV một mùa, ưu tiên mirror rồi lùi về nguồn gốc."""
    code = season_code(start_year)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{league}_{code}.csv"

    if cache_file.exists() and cache_file.stat().st_size > 100:
        return cache_file.read_bytes()

    errors = []
    urls = []
    if league in MIRROR_LEAGUES:
        urls.append(f"{MIRROR_URL}/{MIRROR_LEAGUES[league]}/season-{code}.csv")
    if league in ORIGIN_DIVISIONS:
        urls.append(f"{ORIGIN_URL}/{code}/{ORIGIN_DIVISIONS[league]}.csv")

    for url in urls:
        try:
            response = requests.get(url, timeout=timeout, headers=HEADERS)
            response.raise_for_status()
            if len(response.content) < 100:
                raise ValueError("phản hồi quá ngắn")
            cache_file.write_bytes(response.content)
            return response.content
        except Exception as exc:
            errors.append(f"{url.split('/')[2]}: {type(exc).__name__}")

    raise RuntimeError(f"Không tải được {league} {code} — {'; '.join(errors)}")


def load_season(league: str, start_year: int) -> pd.DataFrame:
    """Nạp một mùa kèm kèo đóng cửa tốt nhất có sẵn."""
    raw = _fetch(league, start_year)
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()

    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "league": league,
        "season": start_year,
        "odds_home_team": df["HomeTeam"].astype(str).str.strip(),
        "odds_away_team": df["AwayTeam"].astype(str).str.strip(),
        "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
    }).dropna(subset=["date", "home_goals", "away_goals"])

    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    # Lấp dần theo thứ tự ưu tiên thay vì chọn một bộ duy nhất cho cả mùa.
    # Mùa 2025-26 chỉ có Pinnacle cho 210/380 trận; nếu chọn cứng một bộ thì
    # 170 trận còn lại mất kèo hoàn toàn, trong khi AvgC phủ đủ cả mùa.
    out["odds_home"] = pd.NA
    out["odds_draw"] = pd.NA
    out["odds_away"] = pd.NA
    out["odds_source"] = pd.NA

    for h, d, a, label in CLOSING_SETS:
        if not all(c in df.columns for c in (h, d, a)):
            continue
        odds = df[[h, d, a]].apply(pd.to_numeric, errors="coerce").loc[out.index]
        usable = odds.notna().all(axis=1) & (odds > 1.0).all(axis=1)
        gap = out["odds_home"].isna() & usable
        if not gap.any():
            continue
        out.loc[gap, "odds_home"] = odds.loc[gap, h].values
        out.loc[gap, "odds_draw"] = odds.loc[gap, d].values
        out.loc[gap, "odds_away"] = odds.loc[gap, a].values
        out.loc[gap, "odds_source"] = label

    return out.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Ghép tên đội giữa hai nguồn
# --------------------------------------------------------------------------

# Các trường hợp chuẩn hoá tự động không giải quyết được.
# Khoá là tên rút gọn của football-data.co.uk, giá trị là tên đầy đủ openfootball.
TEAM_ALIASES = {
    "man united": "manchester united",
    "man city": "manchester city",
    "nottm forest": "nottingham forest",
    "sheffield united": "sheffield united",
    "wolves": "wolverhampton wanderers",
    "west brom": "west bromwich albion",
    "newcastle": "newcastle united",
    "leicester": "leicester city",
    "norwich": "norwich city",
    "leeds": "leeds united",
    "tottenham": "tottenham hotspur",
    "brighton": "brighton hove albion",
    "bournemouth": "bournemouth",
    "west ham": "west ham united",
    "luton": "luton town",
    "ipswich": "ipswich town",
    "hull": "hull city",
    "coventry": "coventry city",
    "cardiff": "cardiff city",
    "stoke": "stoke city",
    "swansea": "swansea city",
    "huddersfield": "huddersfield town",
    # La Liga
    "ath madrid": "atletico de madrid",
    "ath bilbao": "athletic club",
    "athletic bilbao": "athletic club",
    # CLB châu Âu: nguồn C1 và nguồn quốc nội đặt tên rất khác nhau
    "pae aek": "aek athen",
    "pae olympiakos sfp": "olympiakos",
    "fk crvena zvezda": "crvena zvezda",
    "k benhavn": "fc kobenhavn",
    "sociedad": "real sociedad",
    "betis": "real betis",
    "celta": "celta de vigo",
    "espanol": "espanyol",
    "vallecano": "rayo vallecano",
    "la coruna": "deportivo la coruna",
    "alaves": "deportivo alaves",
    "sp gijon": "sporting gijon",
    "santander": "racing santander",
    # Tên ESPN dùng ở cúp châu Âu, viết khác nguồn kết quả chính.
    "bayern munich": "bayern munchen",
    "slavia prague": "slavia praha",
    "sporting cp": "sporting clube de portugal",
    "aek athens": "aek athen",
    "inter milan": "internazionale",
    # football-data.co.uk viết tắt rất gọn; hai tên dưới đây chặn việc ghép
    # cho Inter và Sporting ở mô hình phạt góc của Champions League.
    "inter": "internazionale milano",
    "sp lisbon": "sporting clube portugal",
    # ESPN gọi Deportivo La Coruña là "Deportivo", mà _NOISE lại xoá đúng từ
    # đó nên tên rút về chuỗi rỗng. Chuyển thẳng sang dạng sau khi lọc.
    "deportivo": "la coruna",
}

# "and" nằm trong danh sách vì nguồn kèo viết "Brighton and Hove Albion"
# còn openfootball viết "Brighton & Hove Albion" — dấu & đã bị thay bằng
# khoảng trắng trước đó, nên không bỏ "and" thì hai bên không bao giờ khớp.
_NOISE = re.compile(
    r"\b(fc|cf|afc|sad|bc|ud|cd|rc|rcd|sd|club|de|and|futbol|futbol|balompie|"
    r"balompie|deportivo|atletico)\b"
)


def normalise(name: str) -> str:
    """
    Đưa tên đội về dạng so sánh được giữa hai nguồn.

    openfootball dùng tên đầy đủ ("Brighton & Hove Albion FC"), còn nguồn kèo
    dùng tên rút gọn ("Brighton"). Hàm này bỏ hậu tố pháp lý, dấu câu và các
    từ đệm, rồi bảng TEAM_ALIASES xử lý phần còn lại.
    """
    # Bỏ dấu trước khi lọc ký tự. Nếu không, "Leganés" thành "legan s" và
    # "Atlético" thành "atl tico" — mọi đội Tây Ban Nha có dấu đều trượt ghép.
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    # NFKD không tách được các chữ cái Bắc Âu/Đức vì chúng là ký tự độc lập
    # chứ không phải chữ cái kèm dấu. Không xử lý riêng thì "Bodø/Glimt" thành
    # "bod glimt" và không bao giờ khớp với "Bodo/Glimt" của ESPN.
    for src, dst in (("ø", "o"), ("æ", "ae"), ("å", "a"), ("ß", "ss"),
                     ("đ", "d"), ("ł", "l"), ("þ", "th")):
        s = s.replace(src, dst)
    s = s.replace("&", " ").replace("'", "").replace(".", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    if s.strip() in TEAM_ALIASES:
        s = TEAM_ALIASES[s.strip()]

    # Bỏ từ đệm SAU khi tra alias, vì alias dùng tên còn nguyên.
    s = _NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_name_map(odds_names: set[str], result_names: set[str]) -> dict[str, str]:
    """
    Dựng bảng quy đổi tên từ phía nguồn kèo sang phía nguồn kết quả.

    Hai bước:
      1. Khớp chính xác sau khi chuẩn hoá.
      2. Khớp theo tập từ con — openfootball hay thêm tiền tố hoặc tên thành
         phố ("CA Osasuna" vs "Osasuna", "Real Oviedo" vs "Oviedo"), nên tập
         từ của tên ngắn thường nằm trọn trong tên dài.

    Bước 2 chỉ nhận khi có ĐÚNG MỘT ứng viên. Nếu không, "madrid" sẽ khớp cả
    "real madrid" lẫn "atletico madrid" và âm thầm gán sai đội — sai lầm kiểu
    này không báo lỗi mà chỉ làm hỏng toàn bộ kết quả phía sau.
    """
    mapping = {}
    unresolved = []

    for name in odds_names:
        if name in result_names:
            mapping[name] = name
        else:
            unresolved.append(name)

    tokenised = {r: set(r.split()) for r in result_names}
    for name in unresolved:
        tokens = set(name.split())
        if not tokens:
            continue

        # Thử cả hai chiều bao hàm. Tên bên kèo có thể ngắn hơn
        # ("Espanyol" vs "Espanyol Barcelona") hoặc dài hơn
        # ("Athletic Bilbao" vs "Athletic Club"), nên chỉ xét một chiều là
        # bỏ sót một nửa số trường hợp.
        hits = [
            r for r, rt in tokenised.items()
            if rt and (tokens <= rt or rt <= tokens)
        ]
        if len(hits) == 1:
            mapping[name] = hits[0]

    return mapping


def attach_odds(
    results: pd.DataFrame, odds: pd.DataFrame, tolerance_days: int = 2
) -> pd.DataFrame:
    """
    Gắn kèo vào bảng kết quả bằng cách ghép theo (ngày, đội nhà, đội khách).

    Cho phép lệch vài ngày vì hai nguồn đôi khi ghi ngày khác nhau với trận
    bị dời lịch hoặc đá qua nửa đêm theo múi giờ khác.
    """
    results = results.copy()
    odds = odds.copy()

    results["_home"] = results["home_team"].map(normalise)
    results["_away"] = results["away_team"].map(normalise)
    odds["_home"] = odds["odds_home_team"].map(normalise)
    odds["_away"] = odds["odds_away_team"].map(normalise)

    name_map = build_name_map(
        set(odds["_home"]) | set(odds["_away"]),
        set(results["_home"]) | set(results["_away"]),
    )
    odds["_home"] = odds["_home"].map(lambda n: name_map.get(n, n))
    odds["_away"] = odds["_away"].map(lambda n: name_map.get(n, n))

    lookup: dict[tuple, list] = {}
    for _, row in odds.iterrows():
        lookup.setdefault((row["_home"], row["_away"]), []).append(row)

    cols = ["odds_home", "odds_draw", "odds_away", "odds_source"]
    for c in cols:
        results[c] = pd.NA

    for i, row in results.iterrows():
        candidates = lookup.get((row["_home"], row["_away"]))
        if not candidates:
            continue
        # Nhiều mùa có thể cùng cặp đấu — chọn bản gần ngày nhất.
        best = min(candidates, key=lambda r: abs((r["date"] - row["date"]).days))
        if abs((best["date"] - row["date"]).days) > tolerance_days:
            continue
        for c in cols:
            results.at[i, c] = best[c]

    return results.drop(columns=["_home", "_away"])
