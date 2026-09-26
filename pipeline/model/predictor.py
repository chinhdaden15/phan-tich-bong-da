"""
Gộp sức mạnh dài hạn, bối cảnh trận đấu và các thị trường thành một
dự đoán hoàn chỉnh, sẵn sàng ghi ra JSON cho app.

Luồng xử lý:

    Dixon-Coles  ->  (λ, μ) nền
                     |
    Bối cảnh     ->  hiệu chỉnh chấn thương / nghỉ ngơi / động lực
                     |
                     v
                 ma trận tỷ số  ->  1X2, tài xỉu, BTTS, tỷ số chính xác
                     |
                     +-> mức tin cậy + phần giải thích từng yếu tố
"""

from __future__ import annotations

from . import adjustments, markets
from .dixon_coles import DixonColesFit


def predict_match(
    fit: DixonColesFit,
    home: str,
    away: str,
    context: adjustments.MatchContext | None = None,
    allow_unknown: bool = True,
) -> dict:
    """
    Dự đoán một trận. Trả về dict tự mô tả đầy đủ, không cần tra cứu thêm.

    Phần "explain" quan trọng ngang phần "prediction": người dùng cần thấy
    vì sao mô hình nghiêng về một bên, chứ không chỉ thấy con số cuối cùng.
    """
    context = context or adjustments.MatchContext()

    lam_base, mu_base = fit.expected_goals(home, away, allow_unknown=allow_unknown)
    adjusted = adjustments.apply(lam_base, mu_base, context)

    matrix = fit.score_matrix(
        home, away, lam=adjusted["lambda"], mu=adjusted["mu"]
    )
    summary = markets.summarise(matrix)
    confidence = fit.confidence(home, away)

    return {
        "home_team": home,
        "away_team": away,
        "expected_goals": {
            "home": round(adjusted["lambda"], 3),
            "away": round(adjusted["mu"], 3),
            "home_before_context": round(lam_base, 3),
            "away_before_context": round(mu_base, 3),
        },
        "prediction": {
            "outcome": {k: round(v, 4) for k, v in summary["outcome"].items()},
            "fair_odds": summary["fair_odds"],
            "most_likely_score": summary["most_likely_score"],
            "correct_score": summary["correct_score"],
            "over_under": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in summary["over_under"].items()
            },
            "btts": {k: round(v, 4) for k, v in summary["btts"].items()},
            "expected_total_goals": round(summary["expected_total_goals"], 3),
        },
        "confidence": confidence,
        "explain": adjusted["explain"],
        "model": {
            "type": "dixon-coles",
            "xi": fit.xi,
            "ridge": fit.ridge,
            "trained_on_matches": fit.n_matches,
            "home_advantage": round(fit.home_advantage, 4),
            "rho": round(fit.rho, 4),
        },
    }


def team_strength_table(fit: DixonColesFit) -> list[dict]:
    """
    Bảng sức mạnh đội bóng để hiển thị trong app.

    Quy đổi hệ số log về đơn vị dễ hiểu: "ghi nhiều hơn/ít hơn trung bình
    giải bao nhiêu phần trăm". Người dùng đọc +18% dễ hơn nhiều so với +0.166.
    """
    import math

    mpt = fit.meta.get("matches_per_team", {})
    rows = []
    for team in fit.teams:
        atk = fit.attack[team]
        dfn = fit.defence[team]
        rows.append({
            "team": team,
            "attack": round(atk, 4),
            "defence": round(dfn, 4),
            "attack_pct": round((math.exp(atk) - 1) * 100, 1),
            # Phòng ngự âm là tốt, nên đảo dấu để "cao = tốt" cho dễ đọc.
            "defence_pct": round((1 - math.exp(dfn)) * 100, 1),
            "overall": round(atk - dfn, 4),
            "matches": mpt.get(team, 0),
        })
    rows.sort(key=lambda r: r["overall"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def predict_with_market(
    fit: DixonColesFit,
    base: dict,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    odds_source: str = "unknown",
    devig: str = "multiplicative",
    total_line: dict | None = None,
    handicap: dict | None = None,
) -> dict:
    """
    Bổ sung góc nhìn thị trường vào một dự đoán đã có từ predict_match().

    Vì sao thị trường được ưu tiên làm con số chính: kiểm định trên 2.270 trận
    Ngoại hạng Anh và La Liga cho thấy kèo đóng cửa đạt RPS 0.1915 so với
    0.1991 của mô hình, và hợp nhất hai bên KHÔNG tốt hơn dùng riêng thị
    trường (chênh lệch có ý nghĩa thống kê). Xem pipeline/blend_backtest.py.

    Mô hình vẫn giữ vai trò riêng: nó giải thích được vì sao (bàn thắng kỳ
    vọng, sức mạnh đội hình, ca vắng mặt), còn kèo chỉ đưa ra một con số trần
    trụi. Và khi hai bên bất đồng, đó là thông tin đáng xem.

    devig mặc định là "multiplicative" vì nó cho RPS thấp hơn Shin một chút
    trên dữ liệu kiểm định, dù chênh lệch nằm trong khoảng nhiễu.
    """
    from . import markets, odds as odds_mod

    market_probs = odds_mod.market_probabilities(
        odds_home, odds_draw, odds_away, method=devig
    )
    if market_probs is None:
        return {"available": False, "reason": "kèo không hợp lệ"}

    # Ép mô hình khớp thị trường. Có kèo tài/xỉu thì giải ba ẩn (λ, μ, rho)
    # để khớp cả 1X2 lẫn tổng bàn; không có thì lùi về hai ẩn khớp mỗi 1X2.
    #
    # Khác biệt không hề nhỏ ở các trận chênh lệch: với Chelsea gặp Hull, bản
    # hai ẩn cho tài 2.5 là 56.3% trong khi thị trường định giá 66.6%, kéo theo
    # ba tỷ số hàng đầu thành 2-0/1-0/3-0 — toàn tỷ số sạch lưới và tổng bàn
    # thấp giả tạo. Bản ba ẩn ra 2-0/3-0/2-1, sát thực tế hơn hẳn.
    market_over = None
    if total_line and total_line.get("over") and total_line.get("under"):
        market_over = odds_mod.devig_two_way(total_line["over"], total_line["under"])

    # Có kèo chấp thì dùng CẶP chấp + tài/xỉu thay cho 1X2 + tài/xỉu.
    #
    # VÌ SAO: hai thị trường này ràng buộc trực tiếp đúng hai đại lượng cần
    # biết — mốc chấp ghim HIỆU (λ - μ), tài/xỉu ghim TỔNG (λ + μ). Kèo chấp
    # cũng có biên lợi nhuận thấp nhất trong các thị trường bóng đá (2-3% so
    # với 5-8% của 1X2), nên khử biên ít làm méo con số hơn.
    #
    # ĐO TRÊN 9.758 TRẬN (Ngoại hạng Anh, La Liga, Serie A, Bundesliga, mùa
    # 2019-20 đến 2025-26, giá đóng cửa):
    #
    #   trúng tỷ số   1X2+tài/xỉu 13,36%  |  chấp+tài/xỉu 13,72%
    #                 chênh +0,36 điểm, khoảng tin cậy 95% [+0,09; +0,63]
    #
    # Con số tổng trông nhỏ vì hai cách chọn CÙNG một tỷ số ở 93% số trận.
    # Xét riêng 7% số trận chúng bất đồng: kèo chấp đúng 16,0% còn 1X2 chỉ
    # 10,9% — chênh +5,1 điểm, KTC [+1,5; +8,9]. Đó mới là chỗ nó tạo khác biệt.
    #
    # Gộp cả ba thị trường thì KÉM HƠN dùng riêng kèo chấp (13,40% so với
    # 13,72%, KTC [+0,04; +0,58]): ràng buộc 1X2 kéo lời giải về lại gần bản
    # cũ. RPS của cửa 1X2 không đổi (0,1939 cả hai), nên đổi sang đây không
    # đánh mất gì.
    calibration_error = None
    handicap_line = None
    if handicap and handicap.get("line") is not None:
        try:
            handicap_line = float(handicap["line"])
            cover = odds_mod.devig_two_way(
                float(handicap["home"]), float(handicap["away"]))
        except (TypeError, ValueError, ZeroDivisionError):
            handicap_line = None

    if handicap_line is not None and market_over is not None:
        lam, mu, rho, converged, calibration_error = odds_mod.calibrate_to_handicap(
            handicap_line, cover, market_over,
            line=float(total_line.get("point", 2.5)),
            rho=fit.rho,
        )
        calibrated_on = [f"chấp {handicap_line:+g}",
                         f"tài/xỉu {total_line.get('point', 2.5)}"]
        # Lời giải hỏng thì lùi về đường cũ thay vì đưa ra số bậy.
        if not converged or (calibration_error or 0) > 1e-3:
            lam, mu, rho, converged, calibration_error = odds_mod.calibrate_to_market_full(
                market_probs, market_over,
                line=float(total_line.get("point", 2.5)),
                rho_start=fit.rho,
                lam_start=base["expected_goals"]["home"],
                mu_start=base["expected_goals"]["away"],
            )
            calibrated_on = ["1x2", f"tài/xỉu {total_line.get('point', 2.5)}"]
    elif market_over is not None:
        lam, mu, rho, converged, calibration_error = odds_mod.calibrate_to_market_full(
            market_probs, market_over,
            line=float(total_line.get("point", 2.5)),
            rho_start=fit.rho,
            lam_start=base["expected_goals"]["home"],
            mu_start=base["expected_goals"]["away"],
        )
        calibrated_on = ["1x2", f"tài/xỉu {total_line.get('point', 2.5)}"]
    else:
        lam, mu, converged = odds_mod.calibrate_to_market(
            market_probs, rho=fit.rho,
            lam_start=base["expected_goals"]["home"],
            mu_start=base["expected_goals"]["away"],
        )
        rho = fit.rho
        calibrated_on = ["1x2"]

    # rho có thể khác rho của mô hình nền, nên dựng ma trận bằng chính rho
    # vừa giải ra thay vì gọi fit.score_matrix() vốn dùng rho cũ.
    matrix = _matrix_with_rho(lam, mu, rho)
    summary = markets.summarise(matrix)

    model_probs = base["prediction"]["outcome"]
    # Truyền cùng cách khử biên đã dùng ở trên, để mọi con số thị trường
    # trong một trận đều nhất quán với nhau.
    disagreement = odds_mod.edges(
        model_probs, odds_home, odds_draw, odds_away, devig=devig
    )

    return {
        "available": True,
        "source": odds_source,
        "devig": devig,
        "offered_odds": {
            "home": round(float(odds_home), 3),
            "draw": round(float(odds_draw), 3),
            "away": round(float(odds_away), 3),
        },
        "expected_goals": {"home": round(lam, 3), "away": round(mu, 3)},
        "calibration_converged": converged,
        "calibrated_on": calibrated_on,
        "calibration_error": (
            None if calibration_error is None else round(calibration_error, 8)
        ),
        "rho": round(rho, 4),
        "total_line": total_line,
        "market_over": None if market_over is None else round(market_over, 4),
        "prediction": {
            "outcome": {k: round(v, 4) for k, v in summary["outcome"].items()},
            "most_likely_score": summary["most_likely_score"],
            "correct_score": summary["correct_score"],
            "over_under": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in summary["over_under"].items()
            },
            "btts": {k: round(v, 4) for k, v in summary["btts"].items()},
            "expected_total_goals": round(summary["expected_total_goals"], 3),
        },
        "disagreement": disagreement,
    }


def _matrix_with_rho(lam: float, mu: float, rho: float, max_goals: int = 10):
    """
    Ma trận tỷ số với rho chỉ định, độc lập với rho của mô hình nền.

    Cần thiết vì phép hiệu chỉnh ba ẩn giải ra một rho riêng cho từng trận,
    trong khi fit.score_matrix() luôn dùng rho chung của cả mô hình.
    """
    import numpy as np
    from scipy.stats import poisson

    xs = np.arange(max_goals + 1)
    mat = np.outer(poisson.pmf(xs, lam), poisson.pmf(xs, mu))
    tau = np.ones_like(mat)
    tau[0, 0] = 1.0 - lam * mu * rho
    tau[0, 1] = 1.0 + lam * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    mat = np.clip(mat * tau, 0.0, None)
    total = mat.sum()
    if total <= 0:
        raise ValueError("Ma trận tỷ số suy biến sau hiệu chỉnh thị trường")
    return mat / total


def predict_european_match(
    fit,
    home: str,
    away: str,
    context=None,
    neutral: bool = False,
) -> dict:
    """
    Dự đoán một trận liên giải bằng mô hình phân tầng CrossLeagueFit.

    Tách riêng khỏi predict_match() vì hai mô hình có cấu trúc tham số khác
    nhau: mô hình một giải chỉ có atk/def, còn mô hình liên giải cộng thêm
    mặt bằng giải. Phần giải thích trả về cũng phải nêu rõ mặt bằng giải,
    vì với các cặp đấu chênh lệch trình độ thì chính nó chi phối kết quả.
    """
    from . import adjustments, markets

    context = context or adjustments.MatchContext()

    lam_base, mu_base = fit.expected_goals(home, away, neutral=neutral)
    adjusted = adjustments.apply(lam_base, mu_base, context)

    matrix = fit.score_matrix(
        home, away, lam=adjusted["lambda"], mu=adjusted["mu"], neutral=neutral
    )
    summary = markets.summarise(matrix)

    home_lg = fit.team_league.get(home, "unknown")
    away_lg = fit.team_league.get(away, "unknown")

    return {
        "home_team": home,
        "away_team": away,
        "expected_goals": {
            "home": round(adjusted["lambda"], 3),
            "away": round(adjusted["mu"], 3),
            "home_before_context": round(lam_base, 3),
            "away_before_context": round(mu_base, 3),
        },
        "prediction": {
            "outcome": {k: round(v, 4) for k, v in summary["outcome"].items()},
            "fair_odds": summary["fair_odds"],
            "most_likely_score": summary["most_likely_score"],
            "correct_score": summary["correct_score"],
            "over_under": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in summary["over_under"].items()
            },
            "btts": {k: round(v, 4) for k, v in summary["btts"].items()},
            "expected_total_goals": round(summary["expected_total_goals"], 3),
        },
        "confidence": fit.confidence(home, away),
        "explain": adjusted["explain"],
        "leagues": {
            "home": {
                "name": home_lg,
                "level": round(
                    fit.league_attack.get(home_lg, 0.0)
                    - fit.league_defence.get(home_lg, 0.0), 3
                ),
            },
            "away": {
                "name": away_lg,
                "level": round(
                    fit.league_attack.get(away_lg, 0.0)
                    - fit.league_defence.get(away_lg, 0.0), 3
                ),
            },
        },
        "neutral_venue": neutral,
        "model": {
            "type": "dixon-coles-phan-tang",
            "xi": fit.xi,
            "ridge": fit.ridge,
            "trained_on_matches": fit.n_matches,
            "bridge_matches": fit.n_bridge_matches,
            "home_advantage": round(fit.home_advantage, 4),
            "rho": round(fit.rho, 4),
            "converged": fit.converged,
        },
    }


def european_strength_table(fit, teams: list[str] | None = None) -> list[dict]:
    """
    Bảng sức mạnh trên thang chung châu Âu.

    Khác với bảng một giải, ở đây dùng sức mạnh HIỆU DỤNG (đã cộng mặt bằng
    giải) để Celtic và Manchester City nằm trên cùng một thang đo.
    """
    import math

    mpt = fit.meta.get("matches_per_team", {})
    bridge = fit.meta.get("bridge_per_team", {})
    pool = teams if teams is not None else fit.teams

    rows = []
    for team in pool:
        if team not in fit.attack:
            continue
        atk = fit.effective_attack(team)
        dfn = fit.effective_defence(team)
        rows.append({
            "team": team,
            "league": fit.team_league.get(team, "unknown"),
            "attack": round(atk, 4),
            "defence": round(dfn, 4),
            "attack_pct": round((math.exp(atk) - 1) * 100, 1),
            "defence_pct": round((1 - math.exp(dfn)) * 100, 1),
            "overall": round(atk - dfn, 4),
            "matches": mpt.get(team, 0),
            "bridge_matches": bridge.get(team, 0),
        })

    rows.sort(key=lambda r: r["overall"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


# Tần suất nền của từng cửa, tính từ chính lịch sử của giải.
#
# Dùng mốc ngẫu nhiên (1/3 và 1/2) là SAI và tôi đã mắc đúng lỗi đó: "Tài 1.5"
# xảy ra khoảng 75% ở mọi trận bóng đá, nên so với mốc 50% thì nó luôn trồi
# lên đầu bảng và chiếm chỗ ở gần như mọi trận — trong khi nó chẳng nói lên
# điều gì riêng về trận đấu đó.
#
# So với tần suất nền thực tế mới trả lời đúng câu hỏi cần trả lời: cửa này
# khác thường đến mức nào SO VỚI một trận bóng đá điển hình.
DEFAULT_BASE_RATES = {
    "home": 0.44, "draw": 0.25, "away": 0.31,
    "over_1.5": 0.75, "over_2.5": 0.52, "over_3.5": 0.30,
    "under_1.5": 0.25, "under_2.5": 0.48, "under_3.5": 0.70,
    "btts_yes": 0.51, "btts_no": 0.49,
}

# Ngưỡng phân loại, tính trên phần vượt tần suất nền. Đặt thấp hơn ngưỡng
# cũ vì thước đo giờ khắt khe hơn nhiều: vượt nền 15 điểm phần trăm đã là
# một trận rất khác thường.
# Tên gọi quen thuộc của từng mốc chấp, theo cách người Việt hay nói.
HANDICAP_NAMES = {
    0.0: "đồng banh",
    0.25: "chấp 1/4 (0 - 1/2)",
    0.5: "chấp nửa trái",
    0.75: "chấp 1/2 - 1",
    1.0: "chấp một trái",
    1.25: "chấp 1 - 1/4",
    1.5: "chấp một rưỡi",
    1.75: "chấp 1 1/2 - 2",
    2.0: "chấp hai trái",
    2.25: "chấp 2 - 1/4",
    2.5: "chấp hai rưỡi",
}


def describe_handicap(line: float, home: str, away: str) -> dict:
    """
    Diễn giải mốc chấp thành câu tiếng Việt và mức cách biệt cần thiết.

    line theo góc nhìn ĐỘI NHÀ: âm là đội nhà chấp. Mốc phần tư là hai kèo gộp
    nên có thể ăn/thua NỬA tiền — chỗ này nói rõ ra, vì đó đúng là điều người
    đặt kèo cần biết trước trận.
    """
    size = abs(line)
    # Kèo đồng banh không có bên nào chấp, nhưng vẫn cần một mốc quy chiếu cho
    # nhãn ăn/thua. Lấy đội nhà, khớp với câu mô tả ngay bên dưới.
    giver, taker = (away, home) if line > 0 else (home, away)
    name = HANDICAP_NAMES.get(round(size, 2), f"chấp {size:g} trái")

    if size == 0:
        need = f"{home} thắng là ăn; hoà thì hoàn tiền."
    else:
        frac = round(size % 1, 2)
        base = int(size // 1)
        if frac == 0.0:
            need = (f"{giver} phải thắng cách biệt từ {base + 1} bàn; "
                    f"thắng đúng {base} bàn thì hoàn tiền.")
        elif frac == 0.5:
            need = f"{giver} phải thắng cách biệt từ {base + 1} bàn."
        elif frac == 0.25:
            need = (f"{giver} phải thắng cách biệt từ {base + 1} bàn; "
                    f"thắng đúng {base} bàn thì thua nửa tiền."
                    if base else f"{giver} phải thắng; hoà thì thua nửa tiền.")
        else:   # 0.75
            need = (f"{giver} phải thắng cách biệt từ {base + 2} bàn; "
                    f"thắng đúng {base + 1} bàn thì ăn nửa tiền.")

    return {
        "line": round(line, 2),
        "size": round(size, 2),
        "name": name,
        "giver": giver,
        "taker": taker,
        "requirement": need,
    }


def _handicap_outcome(margin: int, line: float) -> str:
    """
    Một tỷ số cụ thể thì ĐỘI CHẤP ăn hay thua kèo.

    Mốc phần tư là hai kèo gộp, mỗi bên nửa tiền, nên có thêm hai kết cục
    "ăn nửa" và "thua nửa" — bỏ qua chúng là nói sai với người đặt kèo.
    """
    # Quy về góc nhìn đội chấp: cách biệt dương nghĩa là đội chấp đang dẫn.
    supremacy = margin if line < 0 else -margin
    size = abs(line)

    if abs(line) < 1e-9:
        return "ăn" if margin > 0 else ("hoà" if margin == 0 else "thua")

    frac = round(size % 1, 2)
    parts = [size - 0.25, size + 0.25] if frac in (0.25, 0.75) else [size]

    score = 0.0
    for part in parts:
        edge = supremacy - part
        score += (1.0 if edge > 1e-9 else (0.0 if abs(edge) <= 1e-9 else -1.0)) / len(parts)

    if score > 0.99:
        return "ăn"
    if score > 0.01:
        return "ăn nửa"
    if abs(score) <= 0.01:
        return "hoà"
    if score > -0.99:
        return "thua nửa"
    return "thua"


def handicap_block(matrix, handicap: dict, home: str, away: str,
                   scores: list[dict] | None = None) -> dict | None:
    """
    Khối kèo chấp cho một trận: mốc, diễn giải, và xác suất ăn cửa trên.

    CỐ Ý KHÔNG đưa kèo chấp vào danh sách "cửa đáng tin nhất trận". Nhà cái đặt
    mốc chấp đúng chỗ để hai cửa gần 50/50, nên nó gần như không bao giờ vượt
    được sàn 55% — và khi vượt thì chỉ nghĩa là giá ngắn, chứ không phải có lợi
    thế. Việc của phần này là GIẢI THÍCH, không phải gợi ý.

    Đo trên 9.758 trận: dùng kèo chấp thay cho 1X2 khi dựng ma trận tỷ số KHÔNG
    làm dự đoán chuẩn hơn (RPS 0,19394 so với 0,19386; log-loss tỷ số 2,88101 so
    với 2,88089 — khoảng tin cậy đều chứa 0). Lý do: nhà cái ra giá cả hai thị
    trường từ cùng một mô hình, nên sau khi khử biên chúng nói y hệt nhau.
    """
    if not handicap or handicap.get("line") is None:
        return None

    from .odds import devig_two_way, handicap_two_way

    line = float(handicap["line"])
    info = describe_handicap(line, home, away)

    try:
        market_cover = devig_two_way(float(handicap["home"]), float(handicap["away"]))
    except (TypeError, ValueError, ZeroDivisionError):
        market_cover = None

    model_cover = handicap_two_way(matrix, line)

    # Tỷ số nào ăn được kèo? Đây chính là mối nối giữa dự đoán tỷ số và kèo
    # chấp: xem tỷ số khả dĩ nhất có vượt được mốc chấp hay không.
    #
    # Chấm theo góc nhìn ĐỘI CHẤP, đúng bên mà câu "cần thắng cách biệt mấy
    # bàn" ở trên đang nói tới. Nếu chấm theo đội nhà thì khi đội khách là bên
    # chấp, hai phần sẽ nói ngược nhau và người đọc chắc chắn hiểu nhầm.
    covers = []
    for sc in (scores or [])[:3]:
        covers.append({
            "home": sc["home"], "away": sc["away"],
            "probability": round(float(sc["probability"]), 4),
            "result": _handicap_outcome(
                int(sc["home"]) - int(sc["away"]), line),
        })

    return {
        **info,
        "odds": {"home": handicap.get("home"), "away": handicap.get("away")},
        "source": handicap.get("source"),
        "market_cover": round(market_cover, 4) if market_cover is not None else None,
        "model_cover": round(model_cover, 4),
        # Ghi rõ nhãn ăn/thua đang nói về bên nào, để giao diện không đoán mò.
        "outcome_for": info["giver"],
        "top_scores": covers,
    }


CLARITY_STRONG = 0.18
CLARITY_MODERATE = 0.10

# Mô hình và thị trường lệch quá mức này thì tín hiệu bị hạ một bậc: hai
# nguồn bất đồng nghĩa là có yếu tố mà ít nhất một bên chưa nắm được.
DISAGREEMENT_LIMIT = 0.08

# Sàn để một cửa được phép đứng tên "cửa đáng tin nhất trận".
#
# ĐO ĐƯỢC TRÊN 4.560 TRẬN có kèo thật của nhà cái (Ngoại hạng Anh và La Liga,
# mùa 2020-21 đến 2025-26). Mỗi trận được dựng lại ĐÚNG bộ 11 cửa mà app đưa
# ra: ma trận tỷ số ép khớp đồng thời 1X2 và tài/xỉu 2.5 của nhà cái, rồi suy
# ra các mốc 1.5/3.5 và cửa cả hai đội ghi bàn.
#
#   - Luật cũ "vượt nền cao nhất, không có sàn": trúng 59,7%.
#   - Thêm hai sàn dưới đây: trúng 65,8%. Ghép cặp trên cùng 3.451 trận mà cả
#     hai luật đều nêu cửa: +2,5 điểm, khoảng tin cậy 95% [+1,6; +3,5] —
#     không phải nhiễu.
#
#   - Quan trọng hơn: ở 24% số trận KHÔNG có cửa nào đạt chuẩn, luật cũ vẫn
#     nêu một cửa và chỉ trúng 48,8% — tệ hơn tung đồng xu. Đó chính là những
#     trận app từng khuyên "Hoà" ở mức 27%. Nay các trận đó được nói thẳng là
#     không có cửa đáng tin.
#
# Vì sao không nâng sàn lên 60%: mức đó cho 72,4% nhưng im lặng ở 35% số trận.
# 55% là điểm cân bằng — vẫn nói được ở ba phần tư số trận.
PICK_PROBABILITY_FLOOR = 0.55
PICK_CLARITY_FLOOR = 0.05


def compute_base_rates(results) -> dict[str, float]:
    """
    Tần suất nền thực tế của từng cửa, tính từ kết quả lịch sử của giải.

    Mỗi giải mỗi khác: La Liga hoà nhiều hơn Ngoại hạng Anh, Bundesliga nhiều
    bàn hơn Serie A. Dùng số của chính giải đó thay vì một hằng số chung.
    """
    import pandas as pd

    if results is None or len(results) == 0:
        return dict(DEFAULT_BASE_RATES)

    hg = pd.to_numeric(results["home_goals"], errors="coerce")
    ag = pd.to_numeric(results["away_goals"], errors="coerce")
    ok = hg.notna() & ag.notna()
    hg, ag = hg[ok], ag[ok]
    if len(hg) < 200:
        return dict(DEFAULT_BASE_RATES)

    total = hg + ag
    rates = {
        "home": float((hg > ag).mean()),
        "draw": float((hg == ag).mean()),
        "away": float((hg < ag).mean()),
        "btts_yes": float(((hg > 0) & (ag > 0)).mean()),
    }
    rates["btts_no"] = 1.0 - rates["btts_yes"]
    for line in (1.5, 2.5, 3.5):
        over = float((total > line).mean())
        rates[f"over_{line}"] = over
        rates[f"under_{line}"] = 1.0 - over
    return rates


def summarise_signal(pred: dict, base_rates: dict[str, float] | None = None) -> dict:
    """
    Xếp hạng mọi cửa của một trận theo mức độ RÕ RÀNG, không phải theo lợi nhuận.

    Trả lời đúng câu hỏi "nhìn bảng này thì tin được cái gì": liệt kê các cửa
    kèm mức vượt trên ngẫu nhiên, rồi hạ bậc những cửa mà mô hình và thị
    trường bất đồng hoặc dữ liệu mỏng.

    Đây KHÔNG phải gợi ý đặt cược. Cửa rõ nhất vẫn có thể sai — 65% nghĩa là
    cứ ba lần thì hỏng một, và app luôn ghi rõ điều đó.
    """
    market = pred.get("market") or {}
    source = market.get("prediction") if market.get("available") else pred["prediction"]
    from_market = bool(market.get("available"))

    rates = base_rates or DEFAULT_BASE_RATES
    outcome = source["outcome"]
    options = [
        {"market": "1X2", "pick": "Thắng nhà", "probability": outcome["home"], "key": "home"},
        {"market": "1X2", "pick": "Hoà", "probability": outcome["draw"], "key": "draw"},
        {"market": "1X2", "pick": "Thắng khách", "probability": outcome["away"], "key": "away"},
    ]

    for line, values in source.get("over_under", {}).items():
        options.append({"market": f"Tài/Xỉu {line}", "pick": f"Tài {line}",
                        "probability": values["over"], "key": f"over_{line}"})
        options.append({"market": f"Tài/Xỉu {line}", "pick": f"Xỉu {line}",
                        "probability": values["under"], "key": f"under_{line}"})

    btts = source.get("btts", {})
    if btts:
        options.append({"market": "Cả hai đội ghi bàn", "pick": "Có",
                        "probability": btts["yes"], "key": "btts_yes"})
        options.append({"market": "Cả hai đội ghi bàn", "pick": "Không",
                        "probability": btts["no"], "key": "btts_no"})

    for opt in options:
        base = rates.get(opt["key"], DEFAULT_BASE_RATES.get(opt["key"], 0.5))
        opt["base_rate"] = round(base, 4)
        opt["clarity"] = round(opt["probability"] - base, 4)
        opt["probability"] = round(opt["probability"], 4)

    # Một cửa chỉ được đứng tên nếu vừa ĐỦ KHẢ NĂNG XẢY RA vừa KHÁC THƯỜNG.
    # Thiếu vế đầu thì sinh ra lời khuyên kiểu "Hoà 27%" — cửa dễ thua hơn
    # thắng. Thiếu vế sau thì luôn rơi vào "Tài 1.5 78%", đúng nhiều nhưng
    # không nói lên điều gì vì trận nào cũng vậy.
    qualified = [
        o for o in options
        if o["probability"] >= PICK_PROBABILITY_FLOOR
        and o["clarity"] >= PICK_CLARITY_FLOOR
    ]
    confident = bool(qualified)

    options.sort(
        key=lambda o: (o in qualified, o["clarity"]) if qualified else (False, o["clarity"]),
        reverse=True,
    )

    if confident:
        best = max(qualified, key=lambda o: o["clarity"])
    else:
        # Không có cửa nào đạt chuẩn. Vẫn nêu cửa khả dĩ nhất để người dùng có
        # cái mà nhìn, nhưng ghi rõ đây chỉ là tham khảo.
        best = max(options, key=lambda o: o["probability"])

    # Xếp bậc ban đầu theo độ rõ, rồi hạ bậc theo các tín hiệu cảnh báo.
    if not confident:
        strength = "mơ hồ"
    elif best["clarity"] >= CLARITY_STRONG:
        strength = "rõ"
    elif best["clarity"] >= CLARITY_MODERATE:
        strength = "vừa"
    else:
        strength = "mơ hồ"

    warnings = []
    disagreement = (market.get("disagreement") or {}).get("max_disagreement")
    if disagreement is not None and disagreement >= DISAGREEMENT_LIMIT:
        warnings.append(
            f"mô hình lệch thị trường {disagreement * 100:.0f} điểm phần trăm"
        )
    if pred.get("confidence", {}).get("level") == "thấp":
        warnings.append("dữ liệu đội bóng mỏng")
    if not from_market:
        warnings.append("chưa có kèo, chỉ dựa vào mô hình")

    order = ["mơ hồ", "vừa", "rõ"]
    if warnings and strength != "mơ hồ":
        strength = order[max(0, order.index(strength) - 1)]

    # Tỷ số dự đoán: lấy tỷ số khả dĩ nhất, kèm xác suất gộp của ba tỷ số đầu.
    # Xác suất gộp quan trọng hơn con số đơn lẻ — một tỷ số riêng lẻ hiếm khi
    # vượt 12%, nghe như vô vọng, trong khi "một trong ba tỷ số này" thường
    # rơi vào khoảng 25-30% và mới là con số đáng dùng.
    scores = source.get("correct_score", [])[:3]
    score_block = None
    if scores:
        top = scores[0]
        score_block = {
            "top": {"home": top["home"], "away": top["away"],
                    "probability": round(top["probability"], 4)},
            "top3": [
                {"home": s["home"], "away": s["away"],
                 "probability": round(s["probability"], 4)}
                for s in scores
            ],
            "top3_combined": round(sum(s["probability"] for s in scores), 4),
        }

    # Tỷ số khả dĩ nhất có ăn khớp với cửa 1X2 rõ nhất không? Nếu lệch nhau
    # thì đó là dấu hiệu trận đấu thực sự khó đoán, cần nói ra.
    consistent = None
    if score_block and best["key"] in ("home", "draw", "away"):
        h, a = score_block["top"]["home"], score_block["top"]["away"]
        implied = "home" if h > a else ("away" if h < a else "draw")
        consistent = implied == best["key"]

    return {
        "best": best,
        "confident": confident,
        "ranked": options[:6],
        "strength": strength,
        "warnings": warnings,
        "from_market": from_market,
        "score": score_block,
        "score_matches_pick": consistent,
        # Nói thẳng tỷ lệ hỏng để con số không bị đọc thành chắc chắn.
        "fails_about": round(1 - best["probability"], 3),
    }
