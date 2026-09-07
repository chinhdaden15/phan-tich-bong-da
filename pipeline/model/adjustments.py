"""
Lớp hiệu chỉnh bàn thắng kỳ vọng theo bối cảnh trận đấu.

Dixon-Coles cho biết sức mạnh **trung bình dài hạn** của đội. Nó không biết
Rodri vừa đứt dây chằng, đội khách vừa đá C1 giữa tuần, hay đội chủ nhà đã
trụ hạng an toàn và chẳng còn gì để đá. Lớp này bù vào phần đó.

TÍNH TRUNG THỰC — đọc kỹ trước khi tin số:
Các hệ số Dixon-Coles được ước lượng từ dữ liệu bằng hợp lý cực đại. Các hệ số
trong file này thì KHÔNG: chúng là suy đoán có cơ sở, lấy độ lớn từ các nghiên
cứu công khai về bóng đá, và cố ý đặt ở mức dè dặt. Lý do là không có sẵn dữ
liệu chấn thương lịch sử miễn phí đủ dài để khớp chúng một cách đàng hoàng.

Hệ quả thực tế: mỗi hiệu chỉnh đều bị chặn biên độ, và hàm explain() trả về
đóng góp của từng yếu tố để app hiển thị minh bạch thay vì đưa ra một con số
duy nhất không giải thích được. Khi đã tích luỹ đủ dữ liệu chấn thương, hãy
khớp lại các hệ số này bằng backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Phần đóng góp của MỘT SUẤT ĐÁ CHÍNH ở mỗi vị trí vào sức mạnh đội.
#
# Cách suy ra: giả định đóng góp theo NHÓM vị trí trong một đội hình điển hình
# (1 thủ môn, 4 hậu vệ, 4 tiền vệ, 2 tiền đạo), rồi chia đều cho số suất:
#
#   tấn công   thủ môn 1% /1 suất   hậu vệ 14% /4   tiền vệ 35% /4   tiền đạo 50% /3
#   phòng ngự  thủ môn 20% /1       hậu vệ 46% /4   tiền vệ 27% /4   tiền đạo  7% /3
#
# Chia cho số suất là bước bắt buộc. Nếu dùng thẳng tỷ lệ nhóm, một tiền vệ
# vắng mặt sẽ bị tính như thể cả tuyến giữa biến mất — sai số này từng khiến
# mô hình hạ xG của Man City từ 2.02 xuống 1.41 chỉ vì thiếu một người.
POSITION_WEIGHTS = {
    "Goalkeeper":  {"attack": 0.010, "defence": 0.200},
    "Defender":    {"attack": 0.035, "defence": 0.115},
    "Midfielder":  {"attack": 0.088, "defence": 0.068},
    "Attacker":    {"attack": 0.167, "defence": 0.023},
}
DEFAULT_POSITION = {"attack": 0.090, "defence": 0.080}

# Tiệm cận trên của mất mát. Không đội nào mất 25% sức tấn công vì danh sách
# chấn thương, kể cả khi vắng nửa đội hình — cầu thủ dự bị vẫn thi đấu.
MAX_ATTACK_LOSS = 0.25
MAX_DEFENCE_LOSS = 0.25

# Nghỉ ngơi: dưới 4 ngày giữa hai trận thì hiệu suất giảm nhẹ.
# Độ lớn ~2%/ngày thiếu, phù hợp với các nghiên cứu về mật độ lịch thi đấu.
REST_PENALTY_PER_DAY = 0.02
REST_BASELINE_DAYS = 4
MAX_REST_PENALTY = 0.06

# Động lực: chỉ có ý nghĩa ở giai đoạn cuối mùa, và độ lớn khiêm tốn.
MOTIVATION_MAX = 0.05
LATE_SEASON_THRESHOLD = 0.75   # từ 75% số vòng trở đi


@dataclass
class MatchContext:
    """Mọi thông tin ngoài sức mạnh dài hạn có thể ảnh hưởng tới trận đấu."""

    home_missing: list[dict] = field(default_factory=list)
    away_missing: list[dict] = field(default_factory=list)
    home_rest_days: int | None = None
    away_rest_days: int | None = None
    home_position: int | None = None
    away_position: int | None = None
    total_teams: int | None = None
    season_progress: float | None = None   # 0.0 đầu mùa -> 1.0 cuối mùa
    neutral_venue: bool = False


def _player_importance(player: dict) -> tuple[float, float]:
    """
    Quy đổi một cầu thủ vắng mặt thành phần tấn công và phòng ngự bị mất.

    player cần có:
        position       "Goalkeeper" | "Defender" | "Midfielder" | "Attacker"
        minutes_share  tỷ lệ số phút cầu thủ đã đá trên tổng số phút CỦA CHÍNH
                       ANH TA có thể đá (0..1). Đá đủ 90 phút mọi trận = 1.0.
                       KHÔNG phải tỷ lệ trên tổng số phút của cả đội.
        importance     tuỳ chọn, bội số chất lượng so với người thay thế.
                       1.0 = trụ cột trung bình, 1.5 = ngôi sao, 2.0 = trần.
    """
    weights = POSITION_WEIGHTS.get(player.get("position"), DEFAULT_POSITION)

    minutes_share = float(player.get("minutes_share", 0.5))
    minutes_share = min(max(minutes_share, 0.0), 1.0)

    # Bội số chất lượng: cầu thủ giỏi hơn hẳn phần còn lại thì mất mát lớn hơn.
    # Chặn ở 2.0 để một ngôi sao không thể một mình xoá sổ cả đội.
    quality = float(player.get("importance", 1.0))
    quality = min(max(quality, 0.0), 2.0)

    factor = minutes_share * quality
    return weights["attack"] * factor, weights["defence"] * factor


def _saturate(raw: float, ceiling: float) -> float:
    """
    Bão hoà mềm: loss = ceiling * (1 - exp(-raw / ceiling)).

    Dùng thay cho trần cứng min(raw, ceiling) vì trần cứng phá mất khả năng
    phân biệt: khi một tiền đạo chủ lực vắng mặt đã đủ chạm trần thì "vắng 1
    người" và "vắng 5 người" cho ra cùng một con số. Hàm này luôn tăng theo
    raw, xấp xỉ raw khi raw nhỏ, và không bao giờ vượt ceiling.
    """
    if raw <= 0:
        return 0.0
    return ceiling * (1.0 - math.exp(-raw / ceiling))


def squad_impact(missing: list[dict]) -> dict:
    """Tổng hợp mất mát tấn công/phòng ngự từ danh sách vắng mặt."""
    attack_loss = 0.0
    defence_loss = 0.0
    detail = []

    for player in missing:
        a, d = _player_importance(player)
        attack_loss += a
        defence_loss += d
        detail.append({
            "name": player.get("name", "?"),
            "position": player.get("position", "?"),
            "reason": player.get("reason", "vắng mặt"),
            "attack_loss": round(a, 4),
            "defence_loss": round(d, 4),
        })

    detail.sort(key=lambda p: p["attack_loss"] + p["defence_loss"], reverse=True)
    return {
        "attack_loss": _saturate(attack_loss, MAX_ATTACK_LOSS),
        "defence_loss": _saturate(defence_loss, MAX_DEFENCE_LOSS),
        "attack_loss_raw": round(attack_loss, 4),
        "defence_loss_raw": round(defence_loss, 4),
        "saturated": attack_loss > MAX_ATTACK_LOSS or defence_loss > MAX_DEFENCE_LOSS,
        "players": detail,
    }


def rest_penalty(rest_days: int | None) -> float:
    """Phạt hiệu suất khi không đủ ngày nghỉ. Trả 0 nếu không có dữ liệu."""
    if rest_days is None:
        return 0.0
    deficit = max(0, REST_BASELINE_DAYS - int(rest_days))
    return min(deficit * REST_PENALTY_PER_DAY, MAX_REST_PENALTY)


def motivation_factor(
    position: int | None, total_teams: int | None, season_progress: float | None
) -> float:
    """
    Hệ số động lực trong khoảng [-MOTIVATION_MAX, +MOTIVATION_MAX].

    Logic: đầu mùa mọi đội đều còn mục tiêu nên trả 0. Cuối mùa, đội đang đua
    vô địch hoặc chạy trốn xuống hạng có thêm động lực; đội lửng lơ giữa bảng
    không còn gì để mất thì ngược lại. Hiệu ứng tăng dần theo độ trễ của mùa.
    """
    if position is None or total_teams is None or season_progress is None:
        return 0.0
    if season_progress < LATE_SEASON_THRESHOLD or total_teams < 4:
        return 0.0

    # Mức độ "muộn" của mùa, 0 tại ngưỡng và 1 ở vòng cuối.
    lateness = (season_progress - LATE_SEASON_THRESHOLD) / (1.0 - LATE_SEASON_THRESHOLD)
    lateness = min(max(lateness, 0.0), 1.0)

    relative = (position - 1) / (total_teams - 1)   # 0 đỉnh bảng, 1 đáy bảng

    if relative <= 0.25:
        intensity = 1.0 - (relative / 0.25) * 0.4      # đua vô địch / suất C1
    elif relative >= 0.80:
        intensity = (relative - 0.80) / 0.20            # chạy trốn xuống hạng
    else:
        intensity = -0.5                                 # lửng lơ, hết mục tiêu

    return MOTIVATION_MAX * intensity * lateness


def apply(lam: float, mu: float, context: MatchContext) -> dict:
    """
    Áp toàn bộ hiệu chỉnh lên (λ, μ) và trả kèm phần giải thích.

    λ là bàn thắng kỳ vọng của đội nhà, μ của đội khách. Mất mát phòng ngự
    của một đội làm tăng bàn thắng kỳ vọng của đối phương.
    """
    home_squad = squad_impact(context.home_missing)
    away_squad = squad_impact(context.away_missing)

    home_rest = rest_penalty(context.home_rest_days)
    away_rest = rest_penalty(context.away_rest_days)

    home_motiv = motivation_factor(
        context.home_position, context.total_teams, context.season_progress
    )
    away_motiv = motivation_factor(
        context.away_position, context.total_teams, context.season_progress
    )

    home_multiplier = (
        (1.0 - home_squad["attack_loss"])
        * (1.0 + away_squad["defence_loss"])
        * (1.0 - home_rest)
        * (1.0 + home_motiv)
    )
    away_multiplier = (
        (1.0 - away_squad["attack_loss"])
        * (1.0 + home_squad["defence_loss"])
        * (1.0 - away_rest)
        * (1.0 + away_motiv)
    )

    lam_adj = lam * home_multiplier
    mu_adj = mu * away_multiplier

    # Sân trung lập (chung kết C1): bỏ lợi thế sân nhà đã gộp trong λ.
    # Lợi thế sân nhà điển hình ~1.20 lần, nên chia lại phần đó.
    if context.neutral_venue:
        lam_adj /= 1.20

    return {
        "lambda": lam_adj,
        "mu": mu_adj,
        "lambda_base": lam,
        "mu_base": mu,
        "explain": {
            "home": {
                "squad": home_squad,
                "rest_penalty": round(home_rest, 4),
                "motivation": round(home_motiv, 4),
                "total_multiplier": round(home_multiplier, 4),
            },
            "away": {
                "squad": away_squad,
                "rest_penalty": round(away_rest, 4),
                "motivation": round(away_motiv, 4),
                "total_multiplier": round(away_multiplier, 4),
            },
            "neutral_venue": context.neutral_venue,
        },
    }
