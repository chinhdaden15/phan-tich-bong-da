"""
Thị trường mở rộng: kèo hiệp 1 và người ghi bàn.

VÌ SAO TỰ MÔ HÌNH HOÁ THAY VÌ MUA KÈO:
The Odds API có sẵn cả hai (h2h_h1, totals_h1, player_goal_scorer_anytime)
nhưng chỉ qua endpoint riêng từng trận, tốn 4 lượt quota mỗi trận. Với ~70
trận thì vượt xa trần 500 lượt/tháng. Tự tính thì miễn phí và phủ mọi trận.

Đổi lại, các con số này KÉM CHÍNH XÁC HƠN kèo thật — chúng suy ra từ mô hình
chứ không phải giá thị trường. App phải nói rõ điều đó thay vì trộn lẫn với
phần 1X2 vốn lấy thẳng từ kèo.
"""

from __future__ import annotations

import math

import numpy as np

# Tỷ lệ bàn thắng ghi trong hiệp 1, đo từ dữ liệu thật của từng giải.
# Con số này ổn định đáng kể: 43.7% ở Ngoại hạng Anh trên 1.086 trận,
# 43.9% ở La Liga trên 1.064 trận. Dùng làm mặc định khi giải chưa đủ dữ liệu.
DEFAULT_H1_SHARE = 0.437

# Số trận tối thiểu để tin vào tỷ lệ tính riêng cho một giải.
MIN_MATCHES_FOR_SHARE = 300


def measure_h1_share(payloads) -> tuple[float, int]:
    """
    Đo tỷ lệ bàn hiệp 1 từ dữ liệu openfootball (có trường score.ht).

    Trả về (tỷ_lệ, số_trận_dùng_để_đo). Ít trận quá thì trả mặc định.
    """
    h1 = ft = n = 0
    for payload in payloads:
        for match in payload.get("matches", []):
            score = match.get("score")
            if not isinstance(score, dict):
                continue
            ht, full = score.get("ht"), score.get("ft")
            if not (isinstance(ht, list) and isinstance(full, list)):
                continue
            if len(ht) != 2 or len(full) != 2:
                continue
            try:
                h1 += int(ht[0]) + int(ht[1])
                ft += int(full[0]) + int(full[1])
                n += 1
            except (TypeError, ValueError):
                continue

    if n < MIN_MATCHES_FOR_SHARE or ft <= 0:
        return DEFAULT_H1_SHARE, n
    return h1 / ft, n


def first_half_markets(lam: float, mu: float, share: float, rho: float = 0.0,
                       max_goals: int = 6) -> dict:
    """
    Kèo hiệp 1, suy từ bàn thắng kỳ vọng cả trận nhân với tỷ lệ hiệp 1.

    Giả định: bàn thắng hiệp 1 cũng theo phân phối Poisson với cường độ bằng
    cường độ cả trận nhân `share`. Đây là xấp xỉ — thực tế đội bóng đá khác
    nhau giữa hai hiệp — nhưng độ chệch nhỏ và đã được đối chiếu với kèo thật
    của nhà cái (xem pipeline/validate_h1.py).

    rho để 0 vì hiệu chỉnh tỷ số thấp của Dixon-Coles ước lượng trên tỷ số
    CẢ TRẬN; áp lại cho hiệp 1 sẽ sai vì tỷ số hiệp 1 vốn đã thấp.
    """
    from scipy.stats import poisson

    lam_h1 = max(lam * share, 1e-6)
    mu_h1 = max(mu * share, 1e-6)

    xs = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(xs, lam_h1), poisson.pmf(xs, mu_h1))
    matrix = matrix / matrix.sum()

    totals = np.add.outer(xs, xs)
    return {
        "expected_goals": {"home": round(lam_h1, 3), "away": round(mu_h1, 3)},
        "outcome": {
            "home": round(float(np.tril(matrix, -1).sum()), 4),
            "draw": round(float(np.trace(matrix)), 4),
            "away": round(float(np.triu(matrix, 1).sum()), 4),
        },
        "over_under": {
            str(line): {
                "over": round(float(matrix[totals > line].sum()), 4),
                "under": round(float(matrix[totals <= line].sum()), 4),
            }
            for line in (0.5, 1.5)
        },
        "most_likely_score": _top_score(matrix),
        "share_used": round(share, 4),
    }


def _top_score(matrix: np.ndarray) -> dict:
    idx = int(np.argmax(matrix))
    n = matrix.shape[1]
    return {
        "home": idx // n, "away": idx % n,
        "probability": round(float(matrix.max()), 4),
    }


# --------------------------------------------------------------------------
# Vì sao KHÔNG có dự đoán người ghi bàn
# --------------------------------------------------------------------------
#
# Đã dựng thử và loại bỏ sau khi đối chiếu với kèo thật.
#
# Dữ liệu người ghi bàn miễn phí duy nhất là endpoint scorers của
# football-data.org: 60 cầu thủ toàn giải, tức chỉ những người ĐÃ ghi bàn.
# Nhà cái treo kèo cho 35 cầu thủ mỗi trận, gồm cả người chưa ghi bàn nào.
#
# Hệ quả khi thử:
#   - Aston Villa chưa ghi bàn nào sau 3 vòng -> không có cầu thủ nào để tính
#   - Tammy Abraham (cửa nhà cái đánh giá cao nhất, 35%) không có trong dữ liệu
#   - Sai số so với kèo thật: 8-11 điểm phần trăm, cả thừa lẫn thiếu
#
# Muốn làm đúng cần dữ liệu cấp đội hình: toàn bộ cầu thủ, số phút thi đấu,
# vị trí, số cú sút. Không nguồn miễn phí nào có. Thà không có tính năng còn
# hơn đưa ra những con số trông đáng tin mà lệch cả chục điểm.
#
# Người ghi bàn THỰC TẾ của các trận đã đá thì vẫn có đầy đủ, lấy từ ESPN —
# xem sources/espn.py và phần lịch sử.



# Trọng số của mùa trước so với mùa này khi tính tỷ lệ ghi bàn của cầu thủ.
# Thấp hơn 1 vì phong độ và vai trò trong đội thay đổi giữa các mùa.
PREVIOUS_SEASON_WEIGHT = 0.5


def player_scoring_rates(
    current: list[dict], previous: list[dict] | None = None
) -> dict[str, dict]:
    """
    Số bàn và số trận của từng cầu thủ, gộp mùa này với mùa trước.

    Đầu vào là danh sách từ endpoint scorers của football-data.org. Danh sách
    này CHỈ có những người đã ghi bàn — người chưa ghi bàn nào sẽ không xuất
    hiện, và scorers_from_lineup() bù cho họ bằng mức nền theo vị trí.
    """
    previous = previous or []
    rows: dict[str, dict] = {}

    def add(entries, weight):
        for e in entries:
            player = (e.get("player") or {}).get("name")
            goals = e.get("goals")
            played = e.get("playedMatches")
            if not player or goals is None or not played:
                continue
            row = rows.setdefault(player, {
                "player": player, "goals": 0.0, "matches": 0.0, "goals_current": 0,
            })
            row["goals"] += float(goals) * weight
            row["matches"] += float(played) * weight
            if weight == 1.0:
                row["goals_current"] = int(goals)

    add(current, 1.0)
    add(previous, PREVIOUS_SEASON_WEIGHT)
    return rows


# --------------------------------------------------------------------------
# Người ghi bàn — bản dùng ĐỘI HÌNH THẬT
# --------------------------------------------------------------------------
#
# Lần thử trước thất bại vì không biết ai ra sân: phải suy từ danh sách 60 vua
# phá lưới toàn giải, thiếu hết những người chưa ghi bàn. Có đội hình rồi thì
# biết chính xác 11 người đá chính và vị trí từng người, nên gán được mức nền
# theo vị trí cho người chưa có dữ liệu ghi bàn.
#
# Vẫn là ƯỚC LƯỢNG THÔ. App phải ghi rõ điều đó.

# Bàn/trận điển hình theo vị trí, dùng làm mức nền cho cầu thủ chưa ghi bàn
# mùa này. Lấy từ phân bố bàn thắng theo tuyến trong bóng đá hiện đại.
POSITION_BASE_RATE = {
    "F": 0.30, "ST": 0.30, "CF": 0.30, "S": 0.30,
    "W": 0.20, "LW": 0.20, "RW": 0.20,
    "AM": 0.18, "M": 0.10, "CM": 0.10, "DM": 0.05,
    "LM": 0.12, "RM": 0.12,
    "D": 0.04, "CD": 0.04, "LB": 0.04, "RB": 0.04, "WB": 0.05,
    "G": 0.001,
}
DEFAULT_POSITION_RATE = 0.10

# Trọng số của mức nền vị trí so với dữ liệu thật của cầu thủ. Cao thì an toàn
# hơn nhưng kém phân biệt; 6 "trận ảo" là mức cân bằng.
POSITION_PRIOR_MATCHES = 6.0


def _position_rate(position: str | None) -> float:
    if not position:
        return DEFAULT_POSITION_RATE
    key = str(position).upper().split("-")[0]
    return POSITION_BASE_RATE.get(key, DEFAULT_POSITION_RATE)


def scorers_from_lineup(
    starters: list[dict], rates: dict[str, dict], team_lambda: float, limit: int = 8,
) -> list[dict]:
    """
    Xác suất ghi bàn của từng cầu thủ ĐÁ CHÍNH.

    Với mỗi người, tỷ lệ ghi bàn là trung bình có trọng số giữa số liệu thật
    của họ và mức nền theo vị trí. Sau đó chuẩn hoá để tổng kỳ vọng của cả đội
    khớp đúng bàn thắng kỳ vọng của đội trong trận này — nhờ vậy con số tự
    điều chỉnh theo đối thủ và không bao giờ cộng lại vượt quá λ của đội.
    """
    if not starters or team_lambda <= 0:
        return []

    rows = []
    for player in starters:
        name = player.get("name")
        if not name:
            continue
        base = _position_rate(player.get("position"))
        record = rates.get(name)
        if record:
            goals, matches = record["goals"], record["matches"]
        else:
            goals = matches = 0.0
        # Trung bình có trọng số: dữ liệu thật cộng "trận ảo" ở mức nền vị trí.
        rate = (goals + base * POSITION_PRIOR_MATCHES) / (matches + POSITION_PRIOR_MATCHES)
        rows.append({
            "player": name,
            "position": player.get("position"),
            "rate": rate,
            "goals_this_season": (record or {}).get("goals_current", 0),
        })

    total = sum(r["rate"] for r in rows)
    if total <= 0:
        return []

    out = []
    for row in rows:
        expected = team_lambda * (row["rate"] / total)
        out.append({
            "player": row["player"],
            "position": row["position"],
            "probability": round(1.0 - math.exp(-expected), 4),
            "expected_goals": round(expected, 4),
            "goals_this_season": row["goals_this_season"],
        })

    out.sort(key=lambda r: r["probability"], reverse=True)
    return out[:limit]


def in_form_scorers(
    rates: dict[str, dict], team_players: dict[str, list[str]], team: str,
    limit: int = 4,
) -> list[dict]:
    """
    Danh sách cầu thủ đang ghi bàn tốt nhất của một đội, dùng KHI CHƯA CÓ ĐỘI HÌNH.

    Đây KHÔNG phải xác suất ghi bàn — chỉ là phong độ ghi bàn từ đầu mùa. Đội
    hình chỉ được công bố khoảng một tiếng trước giờ bóng lăn, mà app hiển thị
    lịch tới 14 ngày, nên phần lớn thời gian sẽ không có gì để xem nếu chỉ dựa
    vào đội hình.

    Khi đội hình về, refresh_lineups.py thay danh sách này bằng bản có xác suất
    thật tính trên 11 người đá chính.
    """
    names = team_players.get(team) or []
    rows = []
    for name in names:
        record = rates.get(name)
        if not record or record["matches"] <= 0:
            continue
        rows.append({
            "player": name,
            "goals": round(record["goals"], 1),
            "matches": int(record["matches"]),
            "per_match": round(record["goals"] / record["matches"], 3),
            "goals_this_season": record.get("goals_current", 0),
        })

    rows.sort(key=lambda r: (r["per_match"], r["goals"]), reverse=True)
    return rows[:limit]
