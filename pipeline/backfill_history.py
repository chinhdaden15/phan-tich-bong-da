"""
Khôi phục lịch sử dự đoán từ các phiên bản cũ của predictions.json trong git.

Chạy một lần, để cứu những dự đoán đã đưa ra TRƯỚC KHI có hệ thống lưu lịch sử.
Từ nay build.py tự lưu, nên script này chỉ dùng lại nếu lịch sử bị mất.

Nguyên tắc chọn bản dự đoán: với mỗi trận, lấy snapshot của commit CUỐI CÙNG
có thời điểm trước giờ bóng lăn. Đó mới là dự đoán người dùng thực sự nhìn thấy
trước trận. Lấy bản đầu tiên sẽ chấm điểm một dự đoán đã lỗi thời.

    python pipeline/backfill_history.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import history as history_mod
from config import load_env
from sources import espn, football_data_org, results as results_src

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public" / "data"

CURRENT_SEASON = 2026
LEAGUES = ["premier-league", "la-liga", "champions-league"]


def commits_for(path: str) -> list[tuple[str, datetime]]:
    """Các commit đã đụng vào file, kèm thời điểm, cũ trước mới sau."""
    out = subprocess.run(
        ["git", "log", "--format=%H %cI", "--reverse", "--", path],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip()
    rows = []
    for line in out.splitlines():
        sha, _, stamp = line.partition(" ")
        try:
            rows.append((sha, datetime.fromisoformat(stamp)))
        except ValueError:
            continue
    return rows


def snapshot(sha: str, path: str) -> list[dict]:
    out = subprocess.run(["git", "show", f"{sha}:{path}"],
                         capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def kickoff_of(pred: dict) -> datetime | None:
    raw = (pred.get("fixture") or {}).get("kickoff_utc")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            pass
    date = (pred.get("fixture") or {}).get("date")
    if not date:
        return None
    try:
        # Không có giờ thì coi như cuối ngày, để mọi commit trong ngày đều
        # được tính là "trước trận" — thà lấy bản muộn hơn còn hơn bỏ sót.
        return datetime.fromisoformat(f"{date}T23:59:00+00:00")
    except ValueError:
        return None


def latest_before_kickoff(league: str) -> dict[str, dict]:
    """Với mỗi trận, bản dự đoán của commit cuối cùng trước giờ bóng lăn."""
    path = f"public/data/{league}/predictions.json"
    chosen: dict[str, dict] = {}

    for sha, when in commits_for(path):
        for pred in snapshot(sha, path):
            ko = kickoff_of(pred)
            if ko is None or when >= ko:
                continue    # commit sinh ra SAU trận -> không phải dự đoán trước trận
            key = history_mod._key(
                pred["fixture"]["date"], pred["home_team"], pred["away_team"]
            )
            chosen[key] = pred   # commit sau ghi đè commit trước
    return chosen


def main() -> None:
    load_env()
    total = 0

    for league in LEAGUES:
        league_dir = DATA / league
        if not league_dir.exists():
            continue

        candidates = latest_before_kickoff(league)
        if not candidates:
            print(f"{league}: không có dự đoán cũ nào trong git")
            continue

        if league == "champions-league":
            rows = football_data_org.season_matches("champions-league", CURRENT_SEASON)
            import pandas as pd
            # season_matches() không trả sẵn trường result, tính tại chỗ.
            results = pd.DataFrame([{
                "date": pd.Timestamp(m["date"]), "home_team": m["home_team"],
                "away_team": m["away_team"], "home_goals": m["home_goals"],
                "away_goals": m["away_goals"],
                "result": ("H" if m["home_goals"] > m["away_goals"]
                           else "A" if m["home_goals"] < m["away_goals"] else "D"),
            } for m in rows if m["home_goals"] is not None])
        else:
            results = results_src.load_with_recent(
                [league], list(range(CURRENT_SEASON - 1, CURRENT_SEASON + 1)),
                verbose=False,
            )

        if results.empty:
            print(f"{league}: chưa có kết quả để đối chiếu")
            continue

        # Ghi tạm các bản dự đoán cũ vào đúng chỗ archive_finished đọc, rồi
        # khôi phục lại file thật sau khi xong.
        predictions_file = league_dir / "predictions.json"
        backup = predictions_file.read_text(encoding="utf-8") if predictions_file.exists() else None
        predictions_file.write_text(
            json.dumps(list(candidates.values()), ensure_ascii=False), encoding="utf-8"
        )

        try:
            details = espn.details_for_dates(league, espn.recent_days(14))
            archived, added, _mended = history_mod.archive_finished(
                league, league_dir, results, espn_details=details
            )
        finally:
            if backup is not None:
                predictions_file.write_text(backup, encoding="utf-8")

        (league_dir / "history.json").write_text(
            json.dumps(archived, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        with_goals = sum(1 for h in archived if h["actual"].get("goals"))
        print(f"{league}: khôi phục {added} trận (lịch sử: {len(archived)}, "
              f"{with_goals} trận có người ghi bàn)")
        total += added

    print(f"\nTổng cộng khôi phục {total} trận")


if __name__ == "__main__":
    main()
