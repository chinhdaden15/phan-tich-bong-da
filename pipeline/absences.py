"""
Danh sách cầu thủ vắng mặt do người dùng tự nhập.

VÌ SAO PHẢI LÀM THỦ CÔNG: không có nguồn miễn phí nào cho dữ liệu chấn thương
của mùa đang diễn ra.
  - API-Football gói Free chặn mùa hiện tại ("try from 2022 to 2024")
  - Endpoint chấn thương của ESPN tồn tại nhưng trả về rỗng
  - Transfermarkt phải scrape, vùng xám về điều khoản và dễ vỡ

Lớp hiệu chỉnh trong model/adjustments.py đã sẵn sàng, chỉ thiếu đầu vào.
Người theo dõi bóng đá thường xuyên biết Rodri hay Haaland có đá hay không
nhanh hơn bất kỳ API nào, nên nhập tay vài cái tên cho trận quan trọng là
cách thực tế nhất hiện giờ.

Định dạng data/absences.json:

    {
      "premier-league": {
        "Manchester City FC": [
          {"name": "Rodri", "position": "Midfielder",
           "minutes_share": 0.85, "importance": 1.8, "reason": "chấn thương"}
        ]
      }
    }

position:       Goalkeeper | Defender | Midfielder | Attacker
minutes_share:  tỷ lệ số phút cầu thủ đó thường đá (0..1)
importance:     1.0 trụ cột thường, 1.5 ngôi sao, 2.0 là trần
"""

from __future__ import annotations

import json
from pathlib import Path

ABSENCES_FILE = Path(__file__).resolve().parents[1] / "data" / "absences.json"

VALID_POSITIONS = {"Goalkeeper", "Defender", "Midfielder", "Attacker"}


def load(path: Path | None = None) -> dict[str, dict[str, list[dict]]]:
    """
    Đọc file vắng mặt. Không có file thì trả về rỗng, không phải lỗi.

    Mọi bản ghi đều được kiểm tra: sai định dạng thì bỏ qua bản ghi đó và in
    cảnh báo, thay vì để một giá trị lạ lọt vào mô hình và bóp méo dự đoán
    mà không ai biết.
    """
    target = path or ABSENCES_FILE
    if not target.exists():
        return {}

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  absences.json không đọc được ({type(exc).__name__}) — bỏ qua")
        return {}

    cleaned: dict[str, dict[str, list[dict]]] = {}
    for league, teams in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(teams, dict):
            continue
        for team, players in teams.items():
            if not isinstance(players, list):
                continue
            valid = []
            for player in players:
                if not isinstance(player, dict) or not player.get("name"):
                    continue
                position = player.get("position")
                if position not in VALID_POSITIONS:
                    print(f"  bỏ qua {player.get('name')!r} ({team}): "
                          f"vị trí {position!r} không hợp lệ")
                    continue
                try:
                    minutes = float(player.get("minutes_share", 0.7))
                    importance = float(player.get("importance", 1.0))
                except (TypeError, ValueError):
                    print(f"  bỏ qua {player.get('name')!r} ({team}): số liệu không hợp lệ")
                    continue
                valid.append({
                    "name": str(player["name"]),
                    "position": position,
                    "minutes_share": min(max(minutes, 0.0), 1.0),
                    "importance": min(max(importance, 0.0), 2.0),
                    "reason": str(player.get("reason", "vắng mặt")),
                })
            if valid:
                cleaned.setdefault(league, {})[team] = valid

    return cleaned


def for_team(data: dict, league: str, team: str) -> list[dict]:
    """Danh sách vắng mặt của một đội, rỗng nếu không có."""
    return (data.get(league) or {}).get(team, [])


def summary(data: dict) -> dict:
    """Thống kê ngắn để in ra log và hiển thị trong app."""
    teams = sum(len(t) for t in data.values())
    players = sum(len(p) for t in data.values() for p in t.values())
    return {"leagues": len(data), "teams": teams, "players": players}
