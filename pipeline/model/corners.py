"""
Mô hình phạt góc: Poisson có sức mạnh riêng từng đội, cho kèo tài/xỉu góc.

VÌ SAO TÁCH RIÊNG KHỎI MÔ HÌNH BÀN THẮNG: góc và bàn thắng là hai đại lượng
khác hẳn nhau. Một đội ép sân liên tục có thể kiếm 9 góc mà không ghi bàn nào,
còn một đội phòng ngự phản công thắng 2-0 có khi chỉ 2 góc. Dùng chung tham số
tấn công/phòng ngự sẽ sai cả hai phía.

KHÁC BIỆT SO VỚI DIXON-COLES:

  - Không có hiệu chỉnh tau cho tỷ số thấp. Hiệu chỉnh đó sinh ra để xử lý
    hiện tượng dồn ở 0-0, 1-0, 0-1, 1-1 của bàn thắng. Góc trung bình gần 10
    quả mỗi trận nên vùng tỷ số thấp không có ý nghĩa gì.

  - Tổng góc dùng phân phối nhị thức âm chứ không phải Poisson. Đo trên 7.600
    trận: trung bình 9,90 nhưng phương sai 11,36, tức phân tán hơn Poisson
    khoảng 15%. Dùng thẳng Poisson sẽ đánh giá thấp xác suất các trận nhiều
    góc bất thường, đúng chỗ mà kèo tài/xỉu quan tâm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

# Ước lượng trên 7.600 trận Ngoại hạng Anh và La Liga (2016-17 đến 2025-26).
# Phương sai / trung bình của tổng góc mỗi trận. Cập nhật lại khi đổi tập dữ
# liệu, nhưng giá trị này rất ổn định giữa các giải.
DISPERSION = 1.15


@dataclass
class CornerFit:
    teams: list[str]
    attack: np.ndarray          # xu hướng KIẾM được góc
    defence: np.ndarray         # xu hướng ĐỂ đối thủ hưởng góc
    home_advantage: float
    intercept: float
    converged: bool
    matches_per_team: dict[str, int] = field(default_factory=dict)

    def _index(self, team: str) -> int | None:
        try:
            return self.teams.index(team)
        except ValueError:
            return None

    def expected_corners(self, home: str, away: str) -> tuple[float, float]:
        """Số góc kỳ vọng của mỗi đội. Đội lạ dùng mức trung bình giải."""
        h, a = self._index(home), self._index(away)
        atk_h = self.attack[h] if h is not None else 0.0
        def_h = self.defence[h] if h is not None else 0.0
        atk_a = self.attack[a] if a is not None else 0.0
        def_a = self.defence[a] if a is not None else 0.0
        lam_home = np.exp(self.intercept + atk_h + def_a + self.home_advantage)
        lam_away = np.exp(self.intercept + atk_a + def_h)
        return float(lam_home), float(lam_away)

    def total_distribution(self, home: str, away: str, max_corners: int = 30) -> np.ndarray:
        """
        Phân phối xác suất của TỔNG số góc cả trận.

        Nhị thức âm khớp với trung bình lam và phương sai DISPERSION * lam.
        Tham số hoá theo quy ước của scipy: n lần thất bại, xác suất p.
        """
        lam_home, lam_away = self.expected_corners(home, away)
        lam = lam_home + lam_away
        if DISPERSION <= 1.0:      # không phân tán thêm thì quay về Poisson
            from scipy.stats import poisson
            return poisson.pmf(np.arange(max_corners + 1), lam)
        p = 1.0 / DISPERSION
        n = lam * p / (1.0 - p)
        return nbinom.pmf(np.arange(max_corners + 1), n, p)

    def over_under(self, home: str, away: str, line: float) -> dict[str, float]:
        """Xác suất tài/xỉu tại một mốc. Mốc .5 nên không có trường hợp hoà."""
        pmf = self.total_distribution(home, away)
        counts = np.arange(len(pmf))
        over = float(pmf[counts > line].sum())
        return {"over": over, "under": 1.0 - over}

    def confidence(self, home: str, away: str) -> str:
        n = min(self.matches_per_team.get(home, 0), self.matches_per_team.get(away, 0))
        if n >= 30:
            return "cao"
        if n >= 12:
            return "trung bình"
        return "thấp"


def _negative_log_likelihood(params, home_idx, away_idx, hc, ac, weights, n_teams, ridge):
    atk = params[:n_teams]
    dfn = params[n_teams:2 * n_teams]
    home_adv, intercept = params[-2], params[-1]

    log_lam_h = intercept + atk[home_idx] + dfn[away_idx] + home_adv
    log_lam_a = intercept + atk[away_idx] + dfn[home_idx]
    lam_h = np.exp(np.clip(log_lam_h, -8, 8))
    lam_a = np.exp(np.clip(log_lam_a, -8, 8))

    # Bỏ hằng số log(k!) vì nó không phụ thuộc tham số.
    ll = hc * log_lam_h - lam_h + ac * log_lam_a - lam_a
    penalty = ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))
    return -float(np.sum(weights * ll)) + float(penalty)


def fit_corners(
    home_teams,
    away_teams,
    home_corners,
    away_corners,
    match_dates=None,
    reference_date=None,
    xi: float = 0.0018,
    ridge: float = 2.0,
    max_iter: int = 400,
) -> CornerFit:
    """Khớp mô hình góc bằng hợp lý cực đại có trọng số thời gian."""
    home_teams = np.asarray(home_teams, dtype=object)
    away_teams = np.asarray(away_teams, dtype=object)
    hc = np.asarray(home_corners, dtype=float)
    ac = np.asarray(away_corners, dtype=float)

    keep = ~(np.isnan(hc) | np.isnan(ac))
    home_teams, away_teams, hc, ac = home_teams[keep], away_teams[keep], hc[keep], ac[keep]

    teams = sorted(set(home_teams) | set(away_teams))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    home_idx = np.array([index[t] for t in home_teams])
    away_idx = np.array([index[t] for t in away_teams])

    if match_dates is not None:
        dates = np.asarray(match_dates, dtype="datetime64[D]")[keep]
        ref = np.datetime64(reference_date) if reference_date is not None else dates.max()
        days = (ref - dates).astype(float)
        weights = np.exp(-xi * np.clip(days, 0, None))
    else:
        weights = np.ones(len(hc))

    start = np.zeros(2 * n + 2)
    start[-2] = 0.1                                  # lợi thế sân nhà
    start[-1] = np.log(max(np.mean(hc + ac) / 2.0, 0.5))

    result = minimize(
        _negative_log_likelihood,
        start,
        args=(home_idx, away_idx, hc, ac, weights, n, ridge),
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    atk = result.x[:n]
    dfn = result.x[n:2 * n]
    # Ràng buộc nhận dạng: mô hình chỉ xác định được atk và dfn sai khác một
    # hằng số, nên đưa cả hai về trung bình 0 và dồn phần bù vào intercept.
    intercept = float(result.x[-1]) + float(atk.mean()) + float(dfn.mean())
    atk = atk - atk.mean()
    dfn = dfn - dfn.mean()

    counts: dict[str, int] = {}
    for t in np.concatenate([home_teams, away_teams]):
        counts[t] = counts.get(t, 0) + 1

    return CornerFit(
        teams=teams,
        attack=atk,
        defence=dfn,
        home_advantage=float(result.x[-2]),
        intercept=intercept,
        converged=bool(result.success),
        matches_per_team=counts,
    )
