"""
Dựng bộ dữ liệu châu Âu gộp: kết quả quốc nội nhiều giải + trận Champions League.

Trận C1 đóng vai trò CẦU NỐI — chúng là dữ liệu duy nhất cho biết mặt bằng
giữa các giải quốc nội chênh nhau bao nhiêu. Không có chúng, mô hình chỉ biết
thứ hạng trong từng giải và sẽ xếp nhà vô địch Scotland ngang nhà vô địch Anh.
"""

from __future__ import annotations

import pandas as pd

from . import football_data_org, openfootball, results as results_src
from .odds_history import build_name_map, normalise

# Các giải quốc nội nạp để làm nền cho mô hình châu Âu. Danh sách này chọn
# theo mức độ phủ các CLB dự Champions League — thêm giải nhỏ giúp mô hình
# biết về Celtic, Galatasaray, Club Brugge thay vì chỉ có vài trận cúp.
DOMESTIC_LEAGUES = [
    "premier-league", "la-liga", "serie-a", "bundesliga", "ligue-1",
    "eredivisie", "primeira-liga", "scottish-premiership",
    "turkish-super-lig", "belgian-pro-league", "greek-super-league",
    "austrian-bundesliga",
]

COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals", "bridge"]


def build_dataset(
    domestic_seasons: list[int],
    cl_seasons: list[int],
    domestic_leagues: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """
    Trả về (bảng gộp, ánh xạ đội -> giải, ánh xạ tên C1 -> tên đã chuẩn hoá).

    Ánh xạ tên thứ ba là cần thiết vì lịch C1 sắp tới cũng dùng tên của
    football-data.org, nên lúc dự đoán phải quy đổi y hệt lúc huấn luyện.
    """
    leagues = domestic_leagues or DOMESTIC_LEAGUES

    # Dùng nguồn gộp để mặt bằng giải và sức mạnh CLB phản ánh cả vòng đấu
    # vừa diễn ra, không chỉ tới lần cập nhật hằng tuần của openfootball.
    domestic = results_src.load_with_recent(leagues, domestic_seasons, verbose=False)
    domestic["home_team"] = domestic["home_team"].map(normalise)
    domestic["away_team"] = domestic["away_team"].map(normalise)

    team_league: dict[str, str] = {}
    for _, r in domestic.iterrows():
        team_league[r["home_team"]] = r["league"]
        team_league[r["away_team"]] = r["league"]

    rows = []
    for season in cl_seasons:
        try:
            rows.extend(football_data_org.season_matches("champions-league", season))
        except Exception as exc:
            print(f"  bỏ qua C1 {season}: {type(exc).__name__}: {exc}")

    if not rows:
        raise RuntimeError("Không lấy được trận Champions League nào để làm cầu nối")

    cl = pd.DataFrame(rows)
    cl["date"] = pd.to_datetime(cl["date"], errors="coerce")
    cl = cl.dropna(subset=["date", "home_goals", "away_goals"])

    raw_names = set(cl["home_team"]) | set(cl["away_team"])
    name_map = build_name_map(
        {normalise(n) for n in raw_names},
        set(domestic["home_team"]) | set(domestic["away_team"]),
    )

    def resolve(name: str) -> str:
        key = normalise(name)
        return name_map.get(key, key)

    cl["home_team"] = cl["home_team"].map(resolve)
    cl["away_team"] = cl["away_team"].map(resolve)

    # CLB dự C1 nhưng giải quốc nội không nằm trong danh sách nạp sẽ được gom
    # vào nhóm "unknown". Chúng vẫn có tham số riêng, chỉ là ước lượng dựa
    # hoàn toàn vào các trận cúp nên kém chắc chắn — confidence() sẽ báo.
    for team in set(cl["home_team"]) | set(cl["away_team"]):
        team_league.setdefault(team, "unknown")

    domestic["bridge"] = False
    cl["bridge"] = True
    pool = pd.concat(
        [domestic[COLUMNS], cl[COLUMNS]], ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    # Ánh xạ tên gốc (chưa chuẩn hoá) -> tên trong mô hình, để lúc dự đoán
    # lịch sắp tới không phải chuẩn hoá lại và mạo hiểm lệch quy tắc.
    display_map = {n: resolve(n) for n in raw_names}
    return pool, team_league, display_map


def resolve_against_pool(names, pool: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """
    Quy đổi một danh sách tên CLB bất kỳ sang tên dùng trong mô hình.

    Phải có hàm này riêng vì bảng ánh xạ dựng lúc huấn luyện chỉ chứa các CLB
    đã dự C1 những mùa trước. Mùa mới luôn có đội lần đầu góp mặt — Fenerbahçe
    và LASK từng bị bỏ sót đúng vì lý do đó, rồi âm thầm nhận tham số 0 và bị
    mô hình coi như "đội trung bình châu Âu".

    Trả về (ánh xạ, danh sách tên không quy đổi được).
    """
    known = set(pool["home_team"]) | set(pool["away_team"])
    normalised = {normalise(n) for n in names}
    name_map = build_name_map(normalised, known)

    mapping, unresolved = {}, []
    for original in names:
        key = normalise(original)
        target = name_map.get(key, key)
        if target in known:
            mapping[original] = target
        else:
            unresolved.append(original)
    return mapping, unresolved


def upcoming_champions_league(season: int) -> pd.DataFrame:
    """Lịch C1 chưa đá của một mùa."""
    matches = football_data_org.season_matches(
        "champions-league", season, finished_only=False
    )
    pending = [m for m in matches if m["home_goals"] is None]
    if not pending:
        return pd.DataFrame(columns=["kickoff_utc", "home_team", "away_team", "stage"])

    df = pd.DataFrame(pending)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("kickoff_utc").reset_index(drop=True)
