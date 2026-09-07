"""
Kết quả trận đấu gộp từ hai nguồn, mỗi nguồn làm đúng việc nó giỏi.

VẤN ĐỀ: openfootball chỉ cập nhật khoảng một lần mỗi tuần ("auto-update week
36"). Ngày 7/9 mà kết quả mới nhất vẫn là 31/8 — toàn bộ vòng đấu cuối tuần
chưa có. Bảng xếp hạng và cột thắng/hoà/thua vì thế trễ tới bảy ngày, và tăng
tần suất chạy pipeline cũng không giúp được gì vì nguồn chưa có dữ liệu.

CÁCH GIẢI:
  - openfootball       lịch sử sâu (hơn 20 mùa, không cần key, không giới hạn)
  - football-data.org  kết quả mùa hiện tại (API thật, có ngay sau trận đấu)

Trận nào có ở cả hai thì lấy bản của football-data.org, vì nó mới hơn. Tên đội
luôn quy về tên của openfootball để mô hình không nhìn thấy hai đội khác nhau
cho cùng một CLB.
"""

from __future__ import annotations

import pandas as pd

from . import football_data_org, openfootball
from .odds_history import build_name_map, normalise

# Giải mà football-data.org gói miễn phí có phủ. Giải ngoài danh sách này
# chỉ dùng openfootball và chấp nhận độ trễ.
LIVE_LEAGUES = set(football_data_org.COMPETITIONS)


def _match_key(date, home: str, away: str) -> tuple:
    """Khoá nhận dạng một trận, dùng để loại trùng giữa hai nguồn."""
    return (pd.Timestamp(date).strftime("%Y-%m-%d"), normalise(home), normalise(away))


def load_with_recent(
    leagues: list[str], seasons: list[int], days_back: int = 45, verbose: bool = True
) -> pd.DataFrame:
    """
    Nạp lịch sử từ openfootball rồi bổ sung kết quả mới từ football-data.org.

    days_back=45 đủ phủ khoảng trễ tối đa của openfootball với biên an toàn,
    mà vẫn chỉ tốn một lời gọi API cho mỗi giải.
    """
    history = openfootball.load(leagues, seasons)

    # Giữ cả hai dạng tên: dạng chuẩn hoá để so khớp, dạng gốc để hiển thị.
    # So tên đã chuẩn hoá với tập tên thô là lỗi tôi vừa mắc — mọi CLB đều
    # trượt, kể cả "Arsenal FC" vốn giống hệt nhau ở hai nguồn.
    display_by_key = {}
    for name in set(history["home_team"]) | set(history["away_team"]):
        display_by_key.setdefault(normalise(name), name)
    known_keys = set(display_by_key)

    existing = {
        _match_key(r["date"], r["home_team"], r["away_team"])
        for _, r in history.iterrows()
    }

    added_rows = []
    for league in leagues:
        if league not in LIVE_LEAGUES:
            continue
        try:
            recent = football_data_org.recent_results(league, days_back=days_back)
        except Exception as exc:
            if verbose:
                print(f"  kết quả mới {league}: bỏ qua ({type(exc).__name__})")
            continue
        if not recent:
            continue

        # Quy tên của football-data.org về tên openfootball trước khi so trùng,
        # nếu không "Deportivo Alavés" và "Alavés" sẽ thành hai đội riêng biệt.
        raw = {m["home_team"] for m in recent} | {m["away_team"] for m in recent}
        name_map = build_name_map({normalise(n) for n in raw}, known_keys)

        def resolve(name: str) -> str | None:
            key = name_map.get(normalise(name), normalise(name))
            return display_by_key.get(key)

        new_for_league = 0
        unresolved = set()
        for m in recent:
            home = resolve(m["home_team"])
            away = resolve(m["away_team"])
            if home is None or away is None:
                unresolved.add(m["home_team"] if home is None else m["away_team"])
                continue
            key = _match_key(m["date"], home, away)
            if key in existing:
                continue
            existing.add(key)
            added_rows.append({
                "date": pd.Timestamp(m["date"]),
                "league": league,
                "season": seasons[-1],
                "round": None,
                "home_team": home,
                "away_team": away,
                "home_goals": m["home_goals"],
                "away_goals": m["away_goals"],
                "result": m["result"],
            })
            new_for_league += 1

        if verbose:
            newest = max((m["date"] for m in recent), default="—")
            note = f"  kết quả mới {league}: +{new_for_league} trận (nguồn có đến {newest})"
            if unresolved:
                note += f" | chưa ghép tên: {', '.join(sorted(unresolved)[:3])}"
            print(note)

    if not added_rows:
        return history

    merged = pd.concat([history, pd.DataFrame(added_rows)], ignore_index=True)
    return merged.sort_values("date").reset_index(drop=True)
