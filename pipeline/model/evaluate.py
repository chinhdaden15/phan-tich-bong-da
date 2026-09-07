"""
Đo chất lượng dự đoán. Không có phần này thì mô hình chỉ là phỏng đoán
có vỏ toán học.

Ba thước đo chính, đều là "càng thấp càng tốt":

  RPS      Ranked Probability Score — chuẩn vàng cho 1X2 vì nó tính đến
           thứ tự của kết quả (đoán thắng nhà khi thực tế thắng khách
           bị phạt nặng hơn đoán hoà).
  Brier    Sai số bình phương trung bình trên toàn bộ vector xác suất.
  LogLoss  Phạt rất nặng những dự đoán tự tin mà sai.

Mốc tham chiếu thực tế cho giải hàng đầu châu Âu:
  - Đoán bừa đều nhau (1/3 mỗi cửa):  RPS ~ 0.226
  - Mô hình thống kê tốt:              RPS ~ 0.190 - 0.200
  - Kèo đóng cửa của nhà cái:          RPS ~ 0.185 - 0.190

Nếu mô hình của bạn không thắng nổi mốc "đoán bừa", nó đang hỏng.
Nếu nó thắng được kèo đóng cửa trên mẫu lớn, hãy nghi ngờ có lỗi rò rỉ dữ liệu.
"""

from __future__ import annotations

import numpy as np

OUTCOMES = ("home", "draw", "away")


def _to_array(probs) -> np.ndarray:
    """Chuẩn hoá đầu vào về mảng (n, 3) theo thứ tự home/draw/away."""
    if isinstance(probs, dict):
        probs = [probs]
    if isinstance(probs, list) and probs and isinstance(probs[0], dict):
        return np.array([[p[k] for k in OUTCOMES] for p in probs], dtype=float)
    arr = np.asarray(probs, dtype=float)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def _to_onehot(results) -> np.ndarray:
    """Chuyển kết quả thực tế ('H'/'D'/'A' hoặc home/draw/away) thành one-hot."""
    mapping = {
        "H": 0, "home": 0, "1": 0,
        "D": 1, "draw": 1, "X": 1,
        "A": 2, "away": 2, "2": 2,
    }
    out = np.zeros((len(results), 3), dtype=float)
    for i, r in enumerate(results):
        key = r if r in mapping else str(r)
        if key not in mapping:
            raise ValueError(f"Kết quả không hợp lệ: {r!r}")
        out[i, mapping[key]] = 1.0
    return out


def ranked_probability_score(probs, results) -> float:
    """RPS trung bình. Thấp hơn = tốt hơn."""
    p = _to_array(probs)
    o = _to_onehot(results)
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_o = np.cumsum(o, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (p.shape[1] - 1)))


def brier_score(probs, results) -> float:
    """Brier đa lớp. Thấp hơn = tốt hơn."""
    p = _to_array(probs)
    o = _to_onehot(results)
    return float(np.mean(np.sum((p - o) ** 2, axis=1)))


def log_loss(probs, results, eps: float = 1e-15) -> float:
    """Log-loss. Thấp hơn = tốt hơn. Chặn eps để tránh log(0) thành vô cực."""
    p = np.clip(_to_array(probs), eps, 1.0)
    o = _to_onehot(results)
    return float(-np.mean(np.sum(o * np.log(p), axis=1)))


def accuracy(probs, results) -> float:
    """Tỷ lệ cửa có xác suất cao nhất trùng kết quả thật."""
    p = _to_array(probs)
    o = _to_onehot(results)
    return float(np.mean(np.argmax(p, axis=1) == np.argmax(o, axis=1)))


def calibration_curve(probs, results, n_bins: int = 10) -> list[dict]:
    """
    Đường hiệu chuẩn gộp cả ba cửa: trong nhóm các dự đoán "khoảng 60%",
    thực tế có đúng khoảng 60% số lần xảy ra không?

    Mô hình hiệu chuẩn tốt sẽ có observed ≈ predicted ở mọi nhóm. Đây là
    tính chất quan trọng hơn độ chính xác khi bạn dùng số để cân nhắc cửa cược.
    """
    p = _to_array(probs).ravel()
    o = _to_onehot(results).ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append({
            "bin_low": round(float(edges[b]), 3),
            "bin_high": round(float(edges[b + 1]), 3),
            "predicted": round(float(p[mask].mean()), 4),
            "observed": round(float(o[mask].mean()), 4),
            "count": int(mask.sum()),
        })
    return rows


def odds_to_probabilities(odds_home, odds_draw, odds_away) -> dict[str, float]:
    """
    Đổi tỷ lệ nhà cái sang xác suất, đã loại bỏ biên lợi nhuận
    bằng cách chuẩn hoá tổng về 1.

    Đây là mốc so sánh nghiêm túc nhất: kèo đóng cửa gói gọn toàn bộ
    thông tin thị trường, gồm cả tin nội bộ mà mô hình không thể biết.
    """
    raw = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    normalised = raw / raw.sum()
    return dict(zip(OUTCOMES, (float(v) for v in normalised)))


def evaluate(probs, results, label: str = "model") -> dict:
    """Chạy toàn bộ thước đo một lượt."""
    return {
        "label": label,
        "n": len(results),
        "rps": round(ranked_probability_score(probs, results), 5),
        "brier": round(brier_score(probs, results), 5),
        "log_loss": round(log_loss(probs, results), 5),
        "accuracy": round(accuracy(probs, results), 5),
    }


def compare(model_probs, market_probs, results) -> dict:
    """So mô hình với thị trường trên cùng tập trận."""
    model = evaluate(model_probs, results, "model")
    market = evaluate(market_probs, results, "market")
    return {
        "model": model,
        "market": market,
        "rps_gap": round(model["rps"] - market["rps"], 5),
        "verdict": (
            "Mô hình tốt hơn thị trường — hãy kiểm tra rò rỉ dữ liệu trước khi tin"
            if model["rps"] < market["rps"]
            else "Thị trường tốt hơn — bình thường và đúng như kỳ vọng"
        ),
    }
