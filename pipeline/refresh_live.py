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
CURRENT_SEASON = 2026


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

    if league == "nations-league":
        from sources import international

        pool = international.results()
        # Gộp kết quả mới vào tệp tích luỹ y như build.py, để job chạy ba tiếng
        # một lần cũng lưu được trận vừa đá. Nguồn ESPN chỉ dùng được trên máy
        # cá nhân nên phần chính là endpoint scores của nhà cái.
        try:
            fresh = international.recent()
        except Exception:
            fresh = pd.DataFrame()
        season = international.merge_season_results(
            DATA / league / "season_results.json",
            international.scores_from_odds(international.fetch_scores()),
            fresh,
        )
        for extra in (fresh, season):
            if not len(extra):
                continue
            newer = extra[extra["date"] > pool["date"].max()]
            if len(newer):
                add = newer[["date", "home_team", "away_team", "home_goals",
                             "away_goals", "result"]].copy()
                add["league"] = "nations-league"
                add["season"] = add["date"].dt.year
                pool = pd.concat([pool, add], ignore_index=True)
        return (pool.drop_duplicates(subset=["date", "home_team", "away_team"],
                                     keep="last")
                    .sort_values("date").reset_index(drop=True))

    return results_src.load_with_recent(
        [league], list(range(CURRENT_SEASON - 1, CURRENT_SEASON + 1)), verbose=False
    )


def refresh_standings(league: str) -> bool:
    """Ghi lại bảng xếp hạng nếu có thay đổi. Trả True khi có cập nhật."""
    path = DATA / league / "standings.json"
    if not path.exists():
        return False

    if league == "nations-league":
        # Giải này không có ở football-data.org gói miễn phí; bảng xếp hạng
        # được tính từ kết quả trong build.py nên bỏ qua ở đây.
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


def has_finished_since_last_run(league: str) -> bool:
    """
    Có trận nào đã đá xong mà chưa vào lịch sử không?

    Cửa thoát sớm, và nó quan trọng về chi phí: phần nạp kết quả để lưu lịch sử
    là việc nặng nhất của script (openfootball + football-data.org, vài phút).
    Phần lớn lần chạy trong ngày không có trận nào vừa kết thúc, nên chỉ cần
    đọc predictions.json là biết và bỏ qua toàn bộ phần nặng.
    """
    league_dir = DATA / league
    predictions_file = league_dir / "predictions.json"
    if not predictions_file.exists():
        return False

    try:
        preds = json.loads(predictions_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    now = datetime.now(timezone.utc)
    # Chờ ít nhất 2 tiếng sau giờ bóng lăn: trận kéo dài ~2 tiếng và nguồn cần
    # thêm thời gian để chốt kết quả.
    cutoff = now - timedelta(hours=2)
    return any(
        (ko := kickoff(p)) is not None and ko <= cutoff
        for p in preds
    )


def refresh_history(league: str) -> int:
    """Lưu các trận vừa đá xong vào lịch sử. Trả số trận vừa thêm."""
    league_dir = DATA / league
    if not (league_dir / "predictions.json").exists():
        return 0
    if not has_finished_since_last_run(league):
        return 0
    try:
        results = league_results(league)
    except Exception as exc:
        print(f"  {league}: bỏ qua lịch sử ({type(exc).__name__})")
        return 0
    if results.empty:
        return 0

    details = espn.details_for_dates(league, espn.recent_days(7))
    archived, added, mended = history_mod.archive_finished(
        league, league_dir, results, espn_details=details
    )
    if mended:
        print(f"  {league}: sửa lại {mended} bản ghi lịch sử")
    if added or mended:
        (league_dir / "history.json").write_text(
            json.dumps(archived, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")),
            encoding="utf-8",
        )
    return added


def touch_index() -> None:
    """
    Cập nhật dấu thời gian trong index.json.

    Cần thiết vì app đọc trường này để hiện "dữ liệu mới bao lâu". Trước đây
    chỉ build.py ghi nó, nên khi script này cập nhật bảng xếp hạng thì app vẫn
    hiển thị giờ của lần build cũ — người dùng thấy dữ liệu "6 tiếng trước"
    trong khi nó vừa được làm mới.
    """
    path = DATA / "index.json"
    if not path.exists():
        return
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    index["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index["last_light_refresh"] = index["generated_at"]
    path.write_text(
        json.dumps(index, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    load_env()
    tables = archived = 0

    for league in ("premier-league", "la-liga", "champions-league", "nations-league"):
        if refresh_standings(league):
            tables += 1
            print(f"  {league}: bảng xếp hạng có thay đổi")

        added = refresh_history(league)
        if added:
            archived += added
            print(f"  {league}: lưu {added} trận vừa đá xong")

    if tables or archived:
        touch_index()

    parts = []
    if tables:
        parts.append(f"{tables} bảng xếp hạng")
    if archived:
        parts.append(f"{archived} trận vào lịch sử")
    print("Đã cập nhật: " + ", ".join(parts) if parts else "Không có gì thay đổi")


if __name__ == "__main__":
    main()
