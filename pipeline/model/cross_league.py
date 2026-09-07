"""
Mô hình Dixon-Coles phân tầng cho dự đoán liên giải (Champions League).

BÀI TOÁN
Gộp thẳng kết quả nhiều giải vào một mô hình duy nhất cho ra kết quả sai.
Celtic đá ~38 trận mỗi mùa ở Scotland và thắng đậm, nhưng chỉ ~8 trận C1.
Hàm hợp lý bị phần quốc nội áp đảo, nên mô hình xếp Celtic trên cả Man City —
nó không có cách nào biết rằng giải Scotland yếu hơn Ngoại hạng Anh.

CÁCH GIẢI
Tách sức mạnh thành hai tầng:

    sức_tấn_công_thật[i] = atk[i] + league_atk[giải_của_i]

  - atk[i] là sức mạnh TRONG giải, chuẩn hoá tổng bằng 0 ở từng giải.
    Được xác định rất chắc nhờ hàng trăm trận quốc nội.
  - league_atk[L] là mặt bằng chung của cả giải. Các trận trong cùng một giải
    hoàn toàn KHÔNG mang thông tin về đại lượng này (nó bị triệt tiêu giữa hai
    đội cùng giải), nên nó chỉ được xác định bởi các trận liên giải — tức
    những trận Champions League đóng vai trò cầu nối.

Nhờ vậy 38 trận nội địa của Celtic nói lên "Celtic mạnh nhất Scotland", còn
8 trận C1 nói lên "Scotland yếu hơn Anh bao nhiêu". Hai câu hỏi khác nhau
được trả lời bằng hai nguồn dữ liệu khác nhau, thay vì trộn lẫn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .dixon_coles import _LOG_FACT, _tau, MAX_GOALS, RHO_BOUNDS


@dataclass
class CrossLeagueFit:
    """Kết quả khớp mô hình phân tầng."""

    teams: list[str]
    team_league: dict[str, str]
    leagues: list[str]
    attack: dict[str, float]           # trong giải
    defence: dict[str, float]          # trong giải
    league_attack: dict[str, float]    # mặt bằng giải
    league_defence: dict[str, float]
    home_advantage: float
    rho: float
    intercept: float
    n_matches: int
    n_bridge_matches: int
    converged: bool
    log_likelihood: float
    xi: float
    ridge: float
    meta: dict = field(default_factory=dict)

    def effective_attack(self, team: str) -> float:
        """Sức tấn công đã cộng mặt bằng giải — so sánh được giữa các giải."""
        lg = self.team_league.get(team)
        return self.attack.get(team, 0.0) + self.league_attack.get(lg, 0.0)

    def effective_defence(self, team: str) -> float:
        lg = self.team_league.get(team)
        return self.defence.get(team, 0.0) + self.league_defence.get(lg, 0.0)

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        """(λ, μ) cho một cặp đấu bất kỳ, kể cả khác giải."""
        gamma = 0.0 if neutral else self.home_advantage
        lam = math.exp(
            self.intercept + self.effective_attack(home)
            + self.effective_defence(away) + gamma
        )
        mu = math.exp(
            self.intercept + self.effective_attack(away) + self.effective_defence(home)
        )
        return lam, mu

    def score_matrix(self, home: str, away: str, lam=None, mu=None,
                     neutral: bool = False, max_goals: int = MAX_GOALS) -> np.ndarray:
        """Ma trận xác suất tỷ số, cùng công thức với mô hình một giải."""
        from scipy.stats import poisson

        if lam is None or mu is None:
            lam, mu = self.expected_goals(home, away, neutral=neutral)

        xs = np.arange(max_goals + 1)
        mat = np.outer(poisson.pmf(xs, lam), poisson.pmf(xs, mu))
        mat = mat * _tau(xs[:, None], xs[None, :], lam, mu, self.rho)
        mat = np.clip(mat, 0.0, None)
        total = mat.sum()
        if total <= 0:
            raise ValueError("Ma trận tỷ số suy biến")
        return mat / total

    def _team_confidence(self, team: str) -> tuple[str, int, int, int]:
        """Mức tin cậy cho một đội: (mức, trận, trận C1, trận C1 của giải)."""
        n = self.meta.get("matches_per_team", {}).get(team, 0)
        b = self.meta.get("bridge_per_team", {}).get(team, 0)
        league = self.team_league.get(team, "unknown")
        league_bridge = self.meta.get("bridge_per_league", {}).get(league, 0)

        # Gần như không có dữ liệu gì: ước lượng chỉ là mức trung bình chung.
        if n < 15:
            return "thấp", n, b, league_bridge

        # Có trận liên giải của chính đội -> vị trí trên thang châu Âu được
        # xác định trực tiếp.
        if b >= 6:
            return "cao", n, b, league_bridge
        if b > 0:
            return "trung bình", n, b, league_bridge

        # Đội không dự cúp châu Âu gần đây (ví dụ Porto) vẫn định vị được nếu
        # giải của họ đã hiệu chỉnh tốt nhờ các CLB khác cùng giải. Xếp "thấp"
        # trong trường hợp này là sai — dữ liệu về giải đó vẫn dồi dào.
        if league_bridge >= 30:
            return "trung bình", n, b, league_bridge
        return "thấp", n, b, league_bridge

    def confidence(self, home: str, away: str) -> dict:
        """
        Mức tin cậy cho một cặp đấu, lấy theo bên yếu dữ liệu hơn.

        Phân biệt hai tình huống rất khác nhau mà cách tính ngây thơ gộp làm
        một: đội hoàn toàn không có dữ liệu (Viking FK), và đội nhiều dữ liệu
        quốc nội nhưng vắng mặt ở cúp châu Âu gần đây (Porto). Trường hợp sau
        vẫn định vị được qua mặt bằng giải.
        """
        order = {"thấp": 0, "trung bình": 1, "cao": 2}
        lh, nh, bh, lgh = self._team_confidence(home)
        la, na, ba, lga = self._team_confidence(away)
        level = lh if order[lh] <= order[la] else la

        return {
            "level": level,
            "home_matches": nh, "away_matches": na,
            "home_bridge_matches": bh, "away_bridge_matches": ba,
            "home_league_bridge": lgh, "away_league_bridge": lga,
            "limiting_team": home if order[lh] <= order[la] else away,
        }


def _nll(params, home_idx, away_idx, hg, ag, weights, n_teams,
         team_lg_idx, n_leagues, league_masks, ridge, ridge_league):
    """Hàm mục tiêu của mô hình phân tầng."""
    intercept, gamma, rho = params[0], params[1], params[2]
    o = 3
    atk = params[o : o + n_teams].copy()
    dfn = params[o + n_teams : o + 2 * n_teams].copy()
    o2 = o + 2 * n_teams
    lg_atk = params[o2 : o2 + n_leagues].copy()
    lg_dfn = params[o2 + n_leagues : o2 + 2 * n_leagues].copy()

    # Chuẩn hoá sức mạnh đội theo TỪNG giải. Đây là bước then chốt: nó buộc
    # atk[i] chỉ mang nghĩa "so với các đội cùng giải", nhường toàn bộ thông
    # tin về mặt bằng chung cho lg_atk — vốn chỉ xác định được nhờ trận liên giải.
    for mask in league_masks:
        atk[mask] -= atk[mask].mean()
        dfn[mask] -= dfn[mask].mean()

    lg_atk = lg_atk - lg_atk.mean()
    lg_dfn = lg_dfn - lg_dfn.mean()

    eff_atk = atk + lg_atk[team_lg_idx]
    eff_dfn = dfn + lg_dfn[team_lg_idx]

    log_lam = np.clip(intercept + eff_atk[home_idx] + eff_dfn[away_idx] + gamma, -5.0, 5.0)
    log_mu = np.clip(intercept + eff_atk[away_idx] + eff_dfn[home_idx], -5.0, 5.0)
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    tau = _tau(hg, ag, lam, mu, rho)
    if np.any(tau <= 0):
        return 1e10

    ll = (np.log(tau)
          + hg * log_lam - lam - _LOG_FACT[hg]
          + ag * log_mu - mu - _LOG_FACT[ag])

    penalty = ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))
    # Phạt nhẹ hơn cho tham số giải: chúng chỉ có vài trăm trận liên giải để
    # ước lượng, phạt nặng sẽ ép mọi giải về cùng một mặt bằng — đúng thứ
    # mô hình này sinh ra để tránh.
    penalty += ridge_league * (np.sum(lg_atk ** 2) + np.sum(lg_dfn ** 2))

    return -float(np.sum(weights * ll)) + float(penalty)


def fit_cross_league(
    home_teams, away_teams, home_goals, away_goals, team_league: dict[str, str],
    match_dates=None, reference_date=None, is_bridge=None,
    xi: float = 0.0018, ridge: float = 2.0, ridge_league: float = 0.5,
    max_iter: int = 8000,
) -> CrossLeagueFit:
    """
    Khớp mô hình phân tầng trên dữ liệu gộp nhiều giải.

    team_league ánh xạ mỗi đội sang giải quốc nội của nó. Trận liên giải
    (Champions League) là thứ duy nhất xác định được mặt bằng từng giải,
    nên phải có đủ số trận đó thì mô hình mới có ý nghĩa.

    max_iter mặc định 8000 là con số đo được, không phải áng chừng: với ~580
    tham số và đạo hàm tính bằng sai phân, mỗi bước gradient tốn gần 580 lần
    gọi hàm. Ở 2000 vòng lặp nghiệm đã đúng nhưng L-BFGS-B chưa kịp thoả điều
    kiện dừng và báo hội tụ=False; ở 8000 thì báo hội tụ thật, cùng một
    log-likelihood và không đổi một vị trí nào trong top 50.
    """
    home_teams, away_teams = list(home_teams), list(away_teams)
    hg = np.asarray(home_goals, dtype=np.int64)
    ag = np.asarray(away_goals, dtype=np.int64)

    teams = sorted(set(home_teams) | set(away_teams))
    index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Đội không rõ giải được gom vào một nhóm riêng thay vì bị loại — chúng
    # vẫn đóng góp thông tin qua các trận đã đá.
    leagues = sorted({team_league.get(t, "unknown") for t in teams})
    lg_index = {l: i for i, l in enumerate(leagues)}
    n_leagues = len(leagues)
    team_lg_idx = np.array([lg_index[team_league.get(t, "unknown")] for t in teams])
    league_masks = [team_lg_idx == i for i in range(n_leagues)]

    home_idx = np.array([index[t] for t in home_teams])
    away_idx = np.array([index[t] for t in away_teams])

    if match_dates is not None and xi > 0:
        dates = np.asarray(match_dates, dtype="datetime64[D]")
        ref = np.datetime64(reference_date, "D") if reference_date is not None else dates.max()
        weights = np.exp(-xi * np.clip((ref - dates).astype(float), 0.0, None))
    else:
        weights = np.ones(len(hg), dtype=float)

    mean_goals = float(np.mean(np.concatenate([hg, ag])))
    n_params = 3 + 2 * n_teams + 2 * n_leagues
    init = np.zeros(n_params)
    init[0] = math.log(max(mean_goals, 0.1))
    init[1] = 0.25
    init[2] = -0.03

    bounds = ([(-2.0, 2.0), (-1.0, 1.0), RHO_BOUNDS]
              + [(-3.0, 3.0)] * (2 * n_teams)
              + [(-2.0, 2.0)] * (2 * n_leagues))

    result = minimize(
        _nll, init,
        args=(home_idx, away_idx, hg, ag, weights, n_teams,
              team_lg_idx, n_leagues, league_masks, ridge, ridge_league),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter, "maxfun": max_iter * 40, "ftol": 1e-11},
    )

    p = result.x
    atk = p[3 : 3 + n_teams].copy()
    dfn = p[3 + n_teams : 3 + 2 * n_teams].copy()
    for mask in league_masks:
        atk[mask] -= atk[mask].mean()
        dfn[mask] -= dfn[mask].mean()
    o2 = 3 + 2 * n_teams
    lg_atk = p[o2 : o2 + n_leagues] - p[o2 : o2 + n_leagues].mean()
    lg_dfn = p[o2 + n_leagues :] - p[o2 + n_leagues :].mean()

    bridge_per_team, bridge_per_league = {}, {}
    if is_bridge is not None:
        flags = np.asarray(is_bridge, dtype=bool)
        for t, i in index.items():
            bridge_per_team[t] = int(
                np.sum(flags & (home_idx == i)) + np.sum(flags & (away_idx == i))
            )
        # Số trận liên giải của cả giải — dùng để biết mặt bằng giải đó được
        # hiệu chỉnh chắc đến đâu, kể cả với đội không tự dự cúp châu Âu.
        for t, count in bridge_per_team.items():
            lg = team_league.get(t, "unknown")
            bridge_per_league[lg] = bridge_per_league.get(lg, 0) + count

    return CrossLeagueFit(
        teams=teams,
        team_league={t: team_league.get(t, "unknown") for t in teams},
        leagues=leagues,
        attack={t: float(atk[i]) for t, i in index.items()},
        defence={t: float(dfn[i]) for t, i in index.items()},
        league_attack={l: float(lg_atk[i]) for l, i in lg_index.items()},
        league_defence={l: float(lg_dfn[i]) for l, i in lg_index.items()},
        home_advantage=float(p[1]), rho=float(p[2]), intercept=float(p[0]),
        n_matches=len(hg),
        n_bridge_matches=int(np.sum(is_bridge)) if is_bridge is not None else 0,
        converged=bool(result.success), log_likelihood=-float(result.fun),
        xi=xi, ridge=ridge,
        meta={
            "matches_per_team": {
                t: int(np.sum(home_idx == i) + np.sum(away_idx == i))
                for t, i in index.items()
            },
            "bridge_per_team": bridge_per_team,
            "bridge_per_league": bridge_per_league,
            "termination": str(result.message),
        },
    )
