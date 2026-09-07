"""
Kết hợp kèo nhà cái vào dự đoán.

Ba bước, mỗi bước giải một bài toán riêng:

  1. KHỬ BIÊN (de-vig)   Kèo nhà cái luôn cộng thêm biên lợi nhuận, nên tổng
                         nghịch đảo tỷ lệ lớn hơn 1. Phải gỡ phần đó ra mới
                         có xác suất thật mà thị trường đang định giá.

  2. HỢP NHẤT (blend)    Trộn xác suất của mô hình với xác suất thị trường.
                         Gần như luôn tốt hơn dùng riêng một trong hai.

  3. TÌM CHÊNH LỆCH      So mô hình với thị trường để thấy chỗ hai bên bất
                         đồng — đây mới là thứ có ích khi cân nhắc cửa cược.

ĐIỂM QUAN TRỌNG DỄ HIỂU NHẦM:
Bước 2 và bước 3 phục vụ hai mục đích trái ngược nhau. Hợp nhất làm dự đoán
CHÍNH XÁC hơn, nhưng chính vì nó kéo ước lượng về phía thị trường nên nó cũng
XOÁ đi sự bất đồng. Không thể dùng cùng một con số cho cả hai việc: xác suất
đã hợp nhất dùng để dự đoán, xác suất thuần mô hình dùng để so lệch.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .dixon_coles import MAX_GOALS as MAX_GOALS_CAL, RHO_BOUNDS

OUTCOMES = ("home", "draw", "away")


# --------------------------------------------------------------------------
# 1. Khử biên nhà cái
# --------------------------------------------------------------------------

def overround(odds: dict[str, float]) -> float:
    """Tổng nghịch đảo tỷ lệ. Bằng 1 là kèo công bằng; 1.05 là biên 5%."""
    return float(sum(1.0 / odds[k] for k in OUTCOMES))


def devig_multiplicative(odds: dict[str, float]) -> dict[str, float]:
    """
    Khử biên bằng cách chia đều theo tỷ lệ: p_i = (1/o_i) / tổng.

    Đơn giản và nhanh, nhưng có thiên lệch đã biết: nó giả định biên lợi nhuận
    rải đều theo tỷ lệ xác suất, trong khi thực tế nhà cái đặt biên dày hơn ở
    các cửa ít khả năng. Hệ quả là phương pháp này thổi phồng cửa cửa dưới và
    hạ thấp cửa trên (favourite-longshot bias).
    """
    raw = np.array([1.0 / odds[k] for k in OUTCOMES])
    return dict(zip(OUTCOMES, (raw / raw.sum()).tolist()))


def devig_shin(odds: dict[str, float], max_z: float = 0.35) -> dict[str, float]:
    """
    Khử biên bằng phương pháp Shin (1993).

    Mô hình hoá biên lợi nhuận như hệ quả của việc nhà cái tự bảo vệ trước một
    tỷ lệ z người đặt cược có thông tin nội bộ. Giải z sao cho xác suất cộng
    lại đúng bằng 1:

        p_i = [sqrt(z^2 + 4(1-z)·pi_i^2 / PI) - z] / (2(1-z))

    Cho kết quả sát thực tế hơn cách chia đều tỷ lệ, đặc biệt ở các trận
    chênh lệch lớn. Đây là chuẩn trong nghiên cứu định giá kèo bóng đá.
    """
    pi = np.array([1.0 / odds[k] for k in OUTCOMES])
    total = pi.sum()

    if total <= 1.0:
        # Kèo đã công bằng hoặc tốt hơn — không có gì để khử.
        return dict(zip(OUTCOMES, (pi / total).tolist()))

    def implied(z: float) -> np.ndarray:
        return (np.sqrt(z ** 2 + 4.0 * (1.0 - z) * pi ** 2 / total) - z) / (2.0 * (1.0 - z))

    def gap(z: float) -> float:
        return implied(z).sum() - 1.0

    try:
        # gap(0) > 0 khi có biên; gap tăng z thì giảm. Brent tìm nghiệm.
        z = brentq(gap, 0.0, max_z, xtol=1e-10)
    except ValueError:
        # Không tìm được nghiệm trong khoảng — lùi về cách chia đều.
        return devig_multiplicative(odds)

    p = implied(z)
    p = np.clip(p, 1e-9, None)
    return dict(zip(OUTCOMES, (p / p.sum()).tolist()))


DEVIG_METHODS = {
    "multiplicative": devig_multiplicative,
    "shin": devig_shin,
}


def market_probabilities(
    odds_home: float, odds_draw: float, odds_away: float, method: str = "shin"
) -> dict[str, float] | None:
    """Đổi ba tỷ lệ thành xác suất thị trường. Trả None nếu kèo không hợp lệ."""
    values = (odds_home, odds_draw, odds_away)
    if any(v is None or not np.isfinite(v) or v <= 1.0 for v in values):
        return None
    odds = dict(zip(OUTCOMES, (float(v) for v in values)))
    return DEVIG_METHODS[method](odds)


# --------------------------------------------------------------------------
# 2. Hợp nhất mô hình với thị trường
# --------------------------------------------------------------------------

def blend_linear(model: dict, market: dict, weight: float) -> dict[str, float]:
    """
    Trộn tuyến tính: p = (1-w)·mô_hình + w·thị_trường.

    Dễ hiểu, nhưng khi hai nguồn bất đồng mạnh nó tạo ra phân phối bè rộng,
    tức là kém tự tin hơn cả hai nguồn gốc.
    """
    w = float(np.clip(weight, 0.0, 1.0))
    return {k: (1.0 - w) * model[k] + w * market[k] for k in OUTCOMES}


def blend_log(model: dict, market: dict, weight: float, eps: float = 1e-12) -> dict[str, float]:
    """
    Trộn theo trung bình nhân có trọng số: p ∝ mô_hình^(1-w) · thị_trường^w.

    Thường tốt hơn trộn tuyến tính cho dự báo xác suất: khi hai nguồn cùng
    loại trừ một khả năng thì kết quả cũng loại trừ nó, thay vì giữ lại một
    phần như trộn tuyến tính.
    """
    w = float(np.clip(weight, 0.0, 1.0))
    m = np.array([max(model[k], eps) for k in OUTCOMES])
    q = np.array([max(market[k], eps) for k in OUTCOMES])
    p = np.exp((1.0 - w) * np.log(m) + w * np.log(q))
    return dict(zip(OUTCOMES, (p / p.sum()).tolist()))


BLEND_METHODS = {
    "linear": blend_linear,
    "log": blend_log,
}


# --------------------------------------------------------------------------
# 3. So lệch mô hình với thị trường
# --------------------------------------------------------------------------

def edges(
    model: dict[str, float],
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    devig: str = "multiplicative",
) -> dict:
    """
    So xác suất thuần mô hình với kèo thực tế nhà cái đang treo.

    "edge" là lợi nhuận kỳ vọng trên mỗi đồng đặt cược nếu xác suất của mô
    hình đúng: edge = p · tỷ_lệ - 1. Dương nghĩa là mô hình cho rằng cửa đó
    đang bị định giá thấp.

    Dùng xác suất THUẦN mô hình ở đây, không dùng bản đã hợp nhất — hợp nhất
    kéo ước lượng về phía thị trường nên sẽ tự triệt tiêu chính chênh lệch
    mà ta đang muốn tìm.

    Cảnh báo thẳng: kèo đóng cửa là dự báo tốt nhất tồn tại công khai. Edge
    dương gần như luôn có nghĩa là mô hình thiếu thông tin nào đó (chấn
    thương, xoay tua đội hình, thời tiết) chứ không phải nhà cái sai.

    devig phải trùng với cách dùng ở nơi gọi. Nếu không, cùng một cửa sẽ hiện
    hai con số thị trường khác nhau trên cùng một màn hình — bảng so sánh ghi
    50.4% còn ghi chú ngay dưới ghi 50.8%.
    """
    offered = dict(zip(OUTCOMES, (odds_home, odds_draw, odds_away)))
    market = market_probabilities(odds_home, odds_draw, odds_away, method=devig)
    if market is None:
        return {"available": False}

    rows = {}
    for k in OUTCOMES:
        rows[k] = {
            "model_prob": round(model[k], 4),
            "market_prob": round(market[k], 4),
            "offered_odds": round(float(offered[k]), 3),
            "edge": round(model[k] * float(offered[k]) - 1.0, 4),
            "disagreement": round(model[k] - market[k], 4),
        }

    best = max(rows.items(), key=lambda kv: kv[1]["edge"])
    return {
        "available": True,
        "devig": devig,
        "overround": round(overround({k: float(offered[k]) for k in OUTCOMES}), 4),
        "outcomes": rows,
        "best_edge": {"outcome": best[0], **best[1]},
        "max_disagreement": round(
            max(abs(rows[k]["disagreement"]) for k in OUTCOMES), 4
        ),
    }


# --------------------------------------------------------------------------
# 4. Ép mô hình khớp với 1X2 của thị trường
# --------------------------------------------------------------------------

def calibrate_to_market(
    target: dict[str, float],
    rho: float,
    lam_start: float = 1.4,
    mu_start: float = 1.2,
    max_goals: int = 10,
) -> tuple[float, float, bool]:
    """
    Tìm (λ, μ) sao cho ma trận tỷ số Dixon-Coles cho ra đúng 1X2 của thị trường.

    Vì sao cần bước này: kèo chỉ cho ba con số 1X2, trong khi app còn hiển thị
    tỷ số chính xác, tài/xỉu và cả hai đội ghi bàn. Nếu lấy 1X2 từ thị trường
    nhưng các thị trường còn lại từ mô hình, các con số sẽ mâu thuẫn nhau —
    ví dụ tổng xác suất các tỷ số đội nhà thắng không bằng xác suất thắng nhà.

    Cách làm: giữ nguyên rho, giải hai ẩn (λ, μ) khớp hai ràng buộc độc lập
    (xác suất thắng nhà và hoà; xác suất thắng khách suy ra từ tổng bằng 1).
    Bài toán vuông nên thường có nghiệm chính xác.

    Trả về (λ, μ, hội_tụ).
    """
    from scipy.optimize import least_squares
    from scipy.stats import poisson

    xs = np.arange(max_goals + 1)
    grid_x = xs[:, None]
    grid_y = xs[None, :]

    def outcome_probs(log_params):
        lam, mu = np.exp(log_params)
        mat = np.outer(poisson.pmf(xs, lam), poisson.pmf(xs, mu))

        # Hiệu chỉnh Dixon-Coles cho bốn tỷ số thấp, giống hệt trong mô hình.
        tau = np.ones_like(mat)
        tau[0, 0] = 1.0 - lam * mu * rho
        tau[0, 1] = 1.0 + lam * rho
        tau[1, 0] = 1.0 + mu * rho
        tau[1, 1] = 1.0 - rho
        mat = np.clip(mat * tau, 0.0, None)

        total = mat.sum()
        if total <= 0:
            return np.array([1 / 3, 1 / 3])
        mat = mat / total
        return np.array([np.tril(mat, -1).sum(), np.trace(mat)])

    def residual(log_params):
        return outcome_probs(log_params) - np.array([target["home"], target["draw"]])

    result = least_squares(
        residual,
        x0=np.log([max(lam_start, 0.05), max(mu_start, 0.05)]),
        bounds=(np.log([0.05, 0.05]), np.log([6.0, 6.0])),
        xtol=1e-12, ftol=1e-12,
    )
    lam, mu = np.exp(result.x)
    converged = bool(result.success) and float(np.abs(result.fun).max()) < 1e-4
    return float(lam), float(mu), converged


def devig_two_way(odds_over: float, odds_under: float) -> float:
    """Xác suất cửa tài sau khi khử biên, cho thị trường hai cửa."""
    a, b = 1.0 / float(odds_over), 1.0 / float(odds_under)
    return float(a / (a + b))


def calibrate_to_market_full(
    target_1x2: dict[str, float],
    target_over: float,
    line: float = 2.5,
    rho_start: float = -0.05,
    lam_start: float = 1.4,
    mu_start: float = 1.2,
    max_goals: int = MAX_GOALS_CAL,
) -> tuple[float, float, float, bool, float]:
    """
    Giải (λ, μ, rho) sao cho ma trận tỷ số khớp CẢ 1X2 lẫn tổng bàn của thị trường.

    Vì sao cần ba ẩn thay vì hai: 1X2 chỉ ràng buộc HIỆU SỐ bàn thắng. Hai cặp
    (λ, μ) rất khác nhau như (2.4, 0.9) và (1.6, 0.4) có thể cho cùng xác suất
    thắng/hoà/thua, nhưng phân phối tỷ số thì khác hẳn — cặp đầu cho 3-1, 2-1,
    cặp sau cho 2-0, 1-0. Thiếu ràng buộc tổng bàn, phần tỷ số chính xác gần
    như hoàn toàn do mô hình tự đoán, và các tỷ số hàng đầu ra na ná nhau.

    Thêm kèo tài/xỉu là ràng buộc thứ ba, ghim nốt chiều tổng bàn. rho làm ẩn
    thứ ba vì nó điều chỉnh riêng nhóm tỷ số thấp, tương đối độc lập với λ, μ.

    Trả về (λ, μ, rho, hội_tụ, sai_số_lớn_nhất).
    """
    from scipy.optimize import least_squares
    from scipy.stats import poisson

    xs = np.arange(max_goals + 1)
    totals_grid = np.add.outer(xs, xs)

    def predict(params):
        lam, mu = np.exp(params[0]), np.exp(params[1])
        rho = params[2]
        mat = np.outer(poisson.pmf(xs, lam), poisson.pmf(xs, mu))

        tau = np.ones_like(mat)
        tau[0, 0] = 1.0 - lam * mu * rho
        tau[0, 1] = 1.0 + lam * rho
        tau[1, 0] = 1.0 + mu * rho
        tau[1, 1] = 1.0 - rho
        mat = np.clip(mat * tau, 0.0, None)

        total = mat.sum()
        if total <= 0:
            return np.array([1 / 3, 1 / 3, 0.5])
        mat = mat / total
        return np.array([
            np.tril(mat, -1).sum(),          # thắng nhà
            np.trace(mat),                   # hoà
            mat[totals_grid > line].sum(),   # tài
        ])

    goal = np.array([target_1x2["home"], target_1x2["draw"], float(target_over)])

    result = least_squares(
        lambda p: predict(p) - goal,
        x0=[math.log(max(lam_start, 0.05)), math.log(max(mu_start, 0.05)),
            float(np.clip(rho_start, *RHO_BOUNDS))],
        bounds=(
            [math.log(0.05), math.log(0.05), RHO_BOUNDS[0]],
            [math.log(6.0), math.log(6.0), RHO_BOUNDS[1]],
        ),
        xtol=1e-12, ftol=1e-12,
    )

    lam, mu = float(np.exp(result.x[0])), float(np.exp(result.x[1]))
    rho = float(result.x[2])
    max_error = float(np.abs(result.fun).max())
    return lam, mu, rho, bool(result.success) and max_error < 1e-4, max_error
