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
