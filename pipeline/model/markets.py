"""
Chuyển ma trận xác suất tỷ số thành các thị trường quen thuộc:
1X2, tài/xỉu, cả hai đội ghi bàn, tỷ số chính xác, chấp châu Á.

Mọi hàm ở đây nhận ma trận đã chuẩn hoá từ DixonColesFit.score_matrix().
"""

from __future__ import annotations

import numpy as np


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Xác suất thắng nhà / hoà / thắng khách."""
    home = float(np.tril(matrix, -1).sum())   # x > y
    draw = float(np.trace(matrix))            # x == y
    away = float(np.triu(matrix, 1).sum())    # x < y
    return {"home": home, "draw": draw, "away": away}


def over_under(matrix: np.ndarray, line: float = 2.5) -> dict[str, float]:
    """
    Xác suất tài/xỉu cho một mốc bàn thắng.

    Chỉ hỗ trợ mốc lẻ (2.5, 3.5...) vì mốc chẵn tạo ra khả năng hoàn tiền,
    cần xử lý riêng ở tầng gọi.
    """
    n = matrix.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    over = float(matrix[totals > line].sum())
    return {"over": over, "under": 1.0 - over, "line": line}


def both_teams_score(matrix: np.ndarray) -> dict[str, float]:
    """Xác suất cả hai đội đều ghi bàn."""
    yes = float(matrix[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def correct_score(matrix: np.ndarray, top_n: int = 8) -> list[dict]:
    """Danh sách top_n tỷ số khả dĩ nhất, sắp giảm dần theo xác suất."""
    n = matrix.shape[0]
    flat = [
        {"home": int(x), "away": int(y), "probability": float(matrix[x, y])}
        for x in range(n)
        for y in range(n)
    ]
    flat.sort(key=lambda d: d["probability"], reverse=True)
    return flat[:top_n]


def asian_handicap(matrix: np.ndarray, line: float) -> dict[str, float]:
    """
    Chấp châu Á cho mốc lẻ hoặc nguyên, dưới góc nhìn đội nhà.

    line âm nghĩa là đội nhà chấp (ví dụ -1.5 = đội nhà phải thắng từ 2 bàn).
    Mốc nguyên có thể hoà (hoàn tiền), trả về ở khoá "push".
    """
    n = matrix.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n)).astype(float)
    adjusted = diff + line

    home = float(matrix[adjusted > 0].sum())
    push = float(matrix[np.isclose(adjusted, 0.0)].sum())
    away = float(matrix[adjusted < 0].sum())
    return {"home": home, "push": push, "away": away, "line": line}


def expected_points(probs: dict[str, float]) -> dict[str, float]:
    """Điểm kỳ vọng thu về từ trận đấu — hữu ích khi phân tích mục tiêu mùa giải."""
    return {
        "home": 3 * probs["home"] + probs["draw"],
        "away": 3 * probs["away"] + probs["draw"],
    }


def probabilities_to_odds(probs: dict[str, float], margin: float = 0.0) -> dict[str, float]:
    """
    Đổi xác suất sang tỷ lệ thập phân, tuỳ chọn thêm biên lợi nhuận.

    margin=0 cho tỷ lệ công bằng — đây mới là con số nên đem so với nhà cái
    để tìm chỗ lệch giá.
    """
    out = {}
    for key, p in probs.items():
        if not isinstance(p, (int, float)) or p <= 0:
            out[key] = None
            continue
        out[key] = round(1.0 / (p * (1.0 + margin)), 3)
    return out


def summarise(matrix: np.ndarray, lines: tuple[float, ...] = (1.5, 2.5, 3.5)) -> dict:
    """Gói toàn bộ thị trường của một trận vào một dict sẵn sàng ghi ra JSON."""
    probs = outcome_probabilities(matrix)
    n = matrix.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    return {
        "outcome": probs,
        "fair_odds": probabilities_to_odds(probs),
        "over_under": {str(l): over_under(matrix, l) for l in lines},
        "btts": both_teams_score(matrix),
        "correct_score": correct_score(matrix),
        "expected_points": expected_points(probs),
        "expected_total_goals": float((matrix * totals).sum()),
        "most_likely_score": correct_score(matrix, 1)[0],
    }
