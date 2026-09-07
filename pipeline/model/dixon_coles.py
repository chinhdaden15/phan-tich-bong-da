"""
Mô hình Dixon-Coles (1997) — Poisson hai biến có hiệu chỉnh tỷ số thấp
và suy giảm theo thời gian.

Với trận đấu giữa đội nhà i và đội khách j:

    λ = exp(intercept + atk[i] + def[j] + gamma)      # bàn thắng kỳ vọng đội nhà
    μ = exp(intercept + atk[j] + def[i])              # bàn thắng kỳ vọng đội khách

Xác suất tỷ số (x, y):

    P(x, y) = tau(x, y, λ, μ) · Poisson(x; λ) · Poisson(y; μ)

trong đó tau là hiệu chỉnh Dixon-Coles cho các tỷ số thấp (0-0, 0-1, 1-0, 1-1),
nơi mô hình Poisson độc lập lệch rõ rệt so với thực tế.

Tham chiếu: Dixon, M.J. & Coles, S.G. (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market", Applied Statistics 46(2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

# Giới hạn an toàn cho rho: ngoài khoảng này tau có thể âm với lambda/mu thực tế.
RHO_BOUNDS = (-0.25, 0.25)

# Số bàn tối đa khi dựng ma trận tỷ số. 10 là quá đủ:
# P(ghi > 10 bàn) < 1e-6 với mọi lambda thực tế trong bóng đá.
MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    """Hiệu chỉnh Dixon-Coles cho bốn tỷ số thấp. Vector hoá theo mảng numpy."""
    x = np.asarray(x)
    y = np.asarray(y)
    out = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    lam_b, mu_b = np.broadcast_arrays(lam, mu)
    lam_b = np.asarray(lam_b, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)

    out = np.where(m00, 1.0 - lam_b * mu_b * rho, out)
    out = np.where(m01, 1.0 + lam_b * rho, out)
    out = np.where(m10, 1.0 + mu_b * rho, out)
    out = np.where(m11, 1.0 - rho, out)
    return out


@dataclass
class DixonColesFit:
    """Kết quả khớp mô hình — đủ để dự đoán bất kỳ cặp đấu nào."""

    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    intercept: float
    n_matches: int
    converged: bool
    log_likelihood: float
    xi: float
    ridge: float = 0.0
    meta: dict = field(default_factory=dict)

    def expected_goals(
        self, home: str, away: str, allow_unknown: bool = False
    ) -> tuple[float, float]:
        """
        Trả về (λ, μ) — bàn thắng kỳ vọng của đội nhà và đội khách.

        allow_unknown=True coi đội chưa từng xuất hiện trong dữ liệu huấn luyện
        là đội trung bình giải (atk=def=0). Cần thiết khi backtest gặp trận đầu
        tiên của một đội mới thăng hạng, và khi dự đoán vòng mở màn mùa mới.
        """
        if not allow_unknown:
            if home not in self.attack:
                raise KeyError(f"Không có dữ liệu cho đội nhà: {home!r}")
            if away not in self.attack:
                raise KeyError(f"Không có dữ liệu cho đội khách: {away!r}")

        atk_h = self.attack.get(home, 0.0)
        atk_a = self.attack.get(away, 0.0)
        def_h = self.defence.get(home, 0.0)
        def_a = self.defence.get(away, 0.0)

        lam = math.exp(self.intercept + atk_h + def_a + self.home_advantage)
        mu = math.exp(self.intercept + atk_a + def_h)
        return lam, mu

    def confidence(self, home: str, away: str) -> dict:
        """
        Mức tin cậy của dự đoán, dựa trên lượng dữ liệu thực tế của hai đội.

        App phải hiển thị con số này. Dự đoán cho một đội mới thăng hạng đá
        2 trận không đáng tin ngang dự đoán cho đội đã có 190 trận, và giấu
        điều đó đi là cách nhanh nhất khiến người dùng tin nhầm.
        """
        mpt = self.meta.get("matches_per_team", {})
        n_home = mpt.get(home, 0)
        n_away = mpt.get(away, 0)
        weakest = min(n_home, n_away)
        if weakest >= 40:
            level = "cao"
        elif weakest >= 15:
            level = "trung bình"
        else:
            level = "thấp"
        return {
            "level": level,
            "home_matches": n_home,
            "away_matches": n_away,
            "limiting_team": home if n_home <= n_away else away,
        }

    def score_matrix(
        self,
        home: str,
        away: str,
        lam: float | None = None,
        mu: float | None = None,
        max_goals: int = MAX_GOALS,
    ) -> np.ndarray:
        """
        Ma trận (max_goals+1) x (max_goals+1) với M[x][y] = P(đội nhà x - đội khách y).

        Cho phép truyền λ/μ đã hiệu chỉnh sẵn (chấn thương, động lực...) để
        tái dùng phần hiệu chỉnh tỷ số thấp mà không phải khớp lại mô hình.
        """
        if lam is None or mu is None:
            lam, mu = self.expected_goals(home, away)

        xs = np.arange(max_goals + 1)
        px = poisson.pmf(xs, lam)
        py = poisson.pmf(xs, mu)
        mat = np.outer(px, py)

        grid_x = xs[:, None]
        grid_y = xs[None, :]
        mat = mat * _tau(grid_x, grid_y, lam, mu, self.rho)

        mat = np.clip(mat, 0.0, None)
        total = mat.sum()
        if total <= 0:
            raise ValueError("Ma trận tỷ số suy biến — kiểm tra lại tham số mô hình")
        return mat / total


def _negative_log_likelihood(params, home_idx, away_idx, hg, ag, weights, n_teams, ridge):
    """
    Hàm mục tiêu: log-likelihood có trọng số thời gian cộng phạt ridge,
    lấy dấu âm để tối thiểu hoá.

    Phạt ridge tương đương đặt tiên nghiệm Gauss quanh mức trung bình giải cho
    sức mạnh mỗi đội. Đội đá nhiều trận gần như không bị ảnh hưởng vì phần
    likelihood áp đảo; đội mới thăng hạng chỉ có vài trận sẽ bị kéo mạnh về
    trung bình thay vì nhận tham số cực đoan. Không có phần này, một đội thắng
    2 trận đầu mùa sẽ bị mô hình coi là gần như không thể thủng lưới.
    """
    intercept = params[0]
    gamma = params[1]
    rho = params[2]
    atk = params[3 : 3 + n_teams]
    dfn = params[3 + n_teams : 3 + 2 * n_teams]

    # Ràng buộc định danh: trung bình tấn công và phòng ngự bằng 0.
    # Áp ngay trong hàm mục tiêu để tránh phải xử lý ràng buộc ngoài.
    atk = atk - atk.mean()
    dfn = dfn - dfn.mean()

    log_lam = intercept + atk[home_idx] + dfn[away_idx] + gamma
    log_mu = intercept + atk[away_idx] + dfn[home_idx]

    # Chặn tràn số: exp(5) ~ 148 bàn, vượt xa mọi giá trị hợp lý.
    log_lam = np.clip(log_lam, -5.0, 5.0)
    log_mu = np.clip(log_mu, -5.0, 5.0)
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    tau = _tau(hg, ag, lam, mu, rho)
    if np.any(tau <= 0):
        return 1e10

    # log P = log tau + log Poisson(hg; lam) + log Poisson(ag; mu)
    ll = (
        np.log(tau)
        + hg * log_lam - lam - _LOG_FACT[hg]
        + ag * log_mu - mu - _LOG_FACT[ag]
    )
    penalty = ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))
    return -float(np.sum(weights * ll)) + float(penalty)


# Bảng tra log(n!) — nhanh hơn nhiều so với gọi gammaln trong mỗi vòng lặp tối ưu.
_LOG_FACT = np.array([math.lgamma(n + 1) for n in range(64)])


def fit_dixon_coles(
    home_teams,
    away_teams,
    home_goals,
    away_goals,
    match_dates=None,
    reference_date=None,
    xi: float = 0.0018,
    ridge: float = 2.0,
    max_iter: int = 400,
    warm_start: 'DixonColesFit | None' = None,
) -> DixonColesFit:
    """
    Khớp mô hình Dixon-Coles bằng hợp lý cực đại có trọng số thời gian.

    xi điều khiển tốc độ suy giảm: w = exp(-xi * số_ngày_trước).
    xi = 0.0018 tương ứng chu kỳ bán rã ~385 ngày (khoảng một mùa giải),
    gần với giá trị Dixon-Coles tìm được và ổn định trên nhiều giải.
    xi = 0 nghĩa là mọi trận có trọng số bằng nhau.

    ridge điều khiển mức co ngót về trung bình giải. Giá trị mặc định chọn
    bằng dò lưới và kiểm chứng trên giai đoạn tách biệt, không phải áng chừng
    — xem pipeline/backtest.py --grid.
    """
    home_teams = list(home_teams)
    away_teams = list(away_teams)
    hg = np.asarray(home_goals, dtype=np.int64)
    ag = np.asarray(away_goals, dtype=np.int64)

    if not (len(home_teams) == len(away_teams) == len(hg) == len(ag)):
        raise ValueError("Các mảng đầu vào phải có cùng độ dài")
    if len(hg) == 0:
        raise ValueError("Không có trận đấu nào để khớp mô hình")
    if hg.max() >= len(_LOG_FACT) or ag.max() >= len(_LOG_FACT):
        raise ValueError("Tỷ số vượt quá bảng tra giai thừa — dữ liệu có thể bị lỗi")

    teams = sorted(set(home_teams) | set(away_teams))
    index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = np.array([index[t] for t in home_teams])
    away_idx = np.array([index[t] for t in away_teams])

    # Trọng số suy giảm theo thời gian
    if match_dates is not None and xi > 0:
        dates = np.asarray(match_dates, dtype="datetime64[D]")
        ref = (
            np.datetime64(reference_date, "D")
            if reference_date is not None
            else dates.max()
        )
        days_ago = (ref - dates).astype(float)
        days_ago = np.clip(days_ago, 0.0, None)
        weights = np.exp(-xi * days_ago)
    else:
        weights = np.ones(len(hg), dtype=float)

    # Khởi tạo: cường độ trung bình từ chính dữ liệu, đội bắt đầu ở mức trung bình.
    mean_goals = float(np.mean(np.concatenate([hg, ag])))
    init = np.zeros(3 + 2 * n_teams)
    init[0] = math.log(max(mean_goals, 0.1))   # intercept
    init[1] = 0.25                              # lợi thế sân nhà
    init[2] = -0.03                             # rho

    # Khởi động ấm từ lần khớp trước. Trong backtest cuốn chiếu, hai lần khớp
    # liên tiếp gần như giống nhau nên cách này cắt phần lớn số vòng lặp.
    # Đội mới xuất hiện giữ giá trị 0 (mức trung bình giải).
    if warm_start is not None:
        init[0] = warm_start.intercept
        init[1] = warm_start.home_advantage
        init[2] = float(np.clip(warm_start.rho, *RHO_BOUNDS))
        for team, i in index.items():
            init[3 + i] = warm_start.attack.get(team, 0.0)
            init[3 + n_teams + i] = warm_start.defence.get(team, 0.0)

    bounds = (
        [(-2.0, 2.0), (-1.0, 1.0), RHO_BOUNDS]
        + [(-3.0, 3.0)] * n_teams
        + [(-3.0, 3.0)] * n_teams
    )

    result = minimize(
        _negative_log_likelihood,
        init,
        args=(home_idx, away_idx, hg, ag, weights, n_teams, ridge),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10},
    )

    p = result.x
    atk = p[3 : 3 + n_teams] - p[3 : 3 + n_teams].mean()
    dfn = p[3 + n_teams : 3 + 2 * n_teams] - p[3 + n_teams : 3 + 2 * n_teams].mean()

    return DixonColesFit(
        teams=teams,
        attack={t: float(atk[i]) for t, i in index.items()},
        defence={t: float(dfn[i]) for t, i in index.items()},
        home_advantage=float(p[1]),
        rho=float(p[2]),
        intercept=float(p[0]),
        n_matches=len(hg),
        converged=bool(result.success),
        log_likelihood=-float(result.fun),
        xi=xi,
        ridge=ridge,
        meta={
            "effective_sample_size": float(weights.sum()),
            "ridge": ridge,
            "matches_per_team": {
                t: int(np.sum(home_idx == i) + np.sum(away_idx == i))
                for t, i in index.items()
            },
        },
    )
