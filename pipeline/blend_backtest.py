"""
Đo xem kết hợp kèo nhà cái có làm dự đoán tốt hơn không, và tốt hơn bao nhiêu.

So sánh trên cùng một tập trận:
    - mô hình thuần
    - thị trường thuần (hai cách khử biên)
    - hợp nhất hai bên ở nhiều trọng số khác nhau

Dự đoán của mô hình lấy từ backtest cuốn chiếu, nên tại mỗi trận mô hình chỉ
biết những gì đã xảy ra trước đó. Kèo là kèo đóng cửa của chính trận đó.

    python pipeline/blend_backtest.py --leagues premier-league
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import walk_forward
from model.evaluate import calibration_curve, evaluate
from model.odds import BLEND_METHODS, DEVIG_METHODS, market_probabilities
from sources.odds_history import attach_odds, load_season
from sources.openfootball import load

OUTCOMES = ("home", "draw", "away")


def per_match_rps(probs: np.ndarray, results: list[str]) -> np.ndarray:
    """RPS của từng trận, để so sánh theo cặp thay vì chỉ so hai số trung bình."""
    mapping = {"H": 0, "D": 1, "A": 2}
    obs = np.zeros_like(probs)
    for i, r in enumerate(results):
        obs[i, mapping[r]] = 1.0
    cp = np.cumsum(probs, axis=1)[:, :-1]
    co = np.cumsum(obs, axis=1)[:, :-1]
    return np.sum((cp - co) ** 2, axis=1) / (probs.shape[1] - 1)


def paired_test(a: np.ndarray, b: np.ndarray, results: list[str],
                label_a: str, label_b: str, n_boot: int = 20000) -> dict:
    """
    So hai bộ dự đoán trên CÙNG các trận bằng bootstrap theo cặp.

    So hai con số RPS trung bình là không đủ: chênh lệch nhỏ có thể chỉ là
    nhiễu mẫu. Bootstrap theo cặp giữ nguyên ghép đôi từng trận nên loại được
    phần lớn phương sai do độ khó của trận, cho khoảng tin cậy đáng tin hơn.
    """
    ra = per_match_rps(a, results)
    rb = per_match_rps(b, results)
    diff = ra - rb                      # dương nghĩa là b tốt hơn a
    n = len(diff)

    rng = np.random.default_rng(12345)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])

    return {
        "comparison": f"{label_a} vs {label_b}",
        "mean_diff": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0 or hi < 0),
        "better": label_b if diff.mean() > 0 else label_a,
        "n": n,
    }


def build_dataset(leagues: list[str], seasons: list[int], test_start: str,
                  xi: float, ridge: float) -> pd.DataFrame:
    """Chạy mô hình cuốn chiếu rồi gắn kèo đóng cửa vào từng trận."""
    frames = []
    for league in leagues:
        results = load([league], seasons)

        odds_frames = []
        for year in seasons:
            try:
                odds_frames.append(load_season(league, year))
            except Exception as exc:
                print(f"  bỏ qua kèo {league} {year}: {type(exc).__name__}")
        if not odds_frames:
            continue

        print(f"  {league}: chạy mô hình cuốn chiếu…")
        preds = walk_forward(results, test_start, xi=xi, ridge=ridge)

        odds = pd.concat(odds_frames, ignore_index=True)
        merged = attach_odds(preds, odds)
        frames.append(merged)

    if not frames:
        raise RuntimeError("Không dựng được tập dữ liệu nào")

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    print(f"  {len(df)}/{before} trận có đủ kèo đóng cửa\n")
    return df


def market_matrix(df: pd.DataFrame, method: str) -> np.ndarray:
    """Ma trận xác suất thị trường (n, 3) theo một cách khử biên."""
    rows = []
    for _, r in df.iterrows():
        p = market_probabilities(
            float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"]),
            method=method,
        )
        rows.append([p[k] for k in OUTCOMES] if p else [1 / 3, 1 / 3, 1 / 3])
    return np.array(rows)


def blend_matrix(model: np.ndarray, market: np.ndarray,
                 weight: float, method: str) -> np.ndarray:
    """Hợp nhất từng trận theo trọng số cho trước."""
    fn = BLEND_METHODS[method]
    out = []
    for m, q in zip(model, market):
        md = dict(zip(OUTCOMES, m))
        qd = dict(zip(OUTCOMES, q))
        b = fn(md, qd, weight)
        out.append([b[k] for k in OUTCOMES])
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["premier-league"])
    parser.add_argument("--from-season", type=int, default=2018)
    parser.add_argument("--test-start", default="2024-08-01")
    parser.add_argument("--xi", type=float, default=0.0018)
    parser.add_argument("--ridge", type=float, default=2.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seasons = list(range(args.from_season, 2026))
    print(f"Dựng tập dữ liệu cho {args.leagues}…")
    df = build_dataset(args.leagues, seasons, args.test_start, args.xi, args.ridge)

    results = df["result"].tolist()
    model = df[["p_home", "p_draw", "p_away"]].to_numpy()

    print("=" * 62)
    print("BƯỚC 1 — cách khử biên nào đúng hơn?")
    print("=" * 62)
    markets = {}
    for method in DEVIG_METHODS:
        markets[method] = market_matrix(df, method)
        m = evaluate(markets[method], results, f"thị trường ({method})")
        print(f"  {m['label']:>28s}  RPS={m['rps']:.5f}  logloss={m['log_loss']:.5f}  acc={m['accuracy']:.3f}")

    best_devig = min(markets, key=lambda k: evaluate(markets[k], results)["rps"])
    market = markets[best_devig]
    print(f"\n  -> dùng '{best_devig}' cho các bước sau")

    print("\n" + "=" * 62)
    print("BƯỚC 2 — mô hình so với thị trường")
    print("=" * 62)
    m_model = evaluate(model, results, "mô hình thuần")
    m_market = evaluate(market, results, "thị trường thuần")
    for m in (m_model, m_market):
        print(f"  {m['label']:>28s}  RPS={m['rps']:.5f}  logloss={m['log_loss']:.5f}  acc={m['accuracy']:.3f}")
    print(f"\n  Thị trường tốt hơn {(m_model['rps']-m_market['rps'])/m_model['rps']:.1%} theo RPS")

    print("\n" + "=" * 62)
    print("BƯỚC 3 — hợp nhất ở các trọng số (0 = mô hình, 1 = thị trường)")
    print("=" * 62)
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    table = []
    print(f"  {'w':>5} {'log-pool RPS':>14} {'linear RPS':>13} {'log acc':>9}")
    print("  " + "-" * 45)
    for w in weights:
        row = {"weight": w}
        for method in ("log", "linear"):
            b = blend_matrix(model, market, w, method)
            ev = evaluate(b, results)
            row[method] = ev
        table.append(row)
        star = ""
        print(f"  {w:>5.1f} {row['log']['rps']:>14.5f} {row['linear']['rps']:>13.5f} "
              f"{row['log']['accuracy']:>9.3f}{star}")

    best = min(table, key=lambda r: min(r["log"]["rps"], r["linear"]["rps"]))
    best_method = "log" if best["log"]["rps"] <= best["linear"]["rps"] else "linear"
    best_rps = best[best_method]["rps"]

    print("\n" + "=" * 62)
    print(f"TỐT NHẤT: trộn '{best_method}' với w={best['weight']:.1f}  ->  RPS={best_rps:.5f}")
    print(f"  so với mô hình thuần : {(m_model['rps']-best_rps)/m_model['rps']:+.2%}")
    print(f"  so với thị trường    : {(m_market['rps']-best_rps)/m_market['rps']:+.2%}")
    print(f"  n = {len(df)} trận")

    print("\n=== Hiệu chuẩn của bản hợp nhất tốt nhất ===")
    bb = blend_matrix(model, market, best["weight"], best_method)
    print(f"  {'khoảng':>14} {'dự đoán':>9} {'thực tế':>9} {'n':>6}")
    for r in calibration_curve(bb, results, n_bins=10):
        print(f"  {r['bin_low']:>6.2f}-{r['bin_high']:<7.2f} {r['predicted']:>9.3f} "
              f"{r['observed']:>9.3f} {r['count']:>6d}")

    print("\n" + "=" * 62)
    print("BƯỚC 4 — chênh lệch có thật hay chỉ là nhiễu mẫu?")
    print("=" * 62)
    pure_market = blend_matrix(model, market, 1.0, best_method)
    checks = [
        (model, pure_market, "mô hình thuần", "thị trường thuần"),
        (blend_matrix(model, market, 0.8, best_method), pure_market,
         "hợp nhất w=0.8", "thị trường thuần"),
        (blend_matrix(model, market, 0.9, best_method), pure_market,
         "hợp nhất w=0.9", "thị trường thuần"),
    ]
    stats = []
    for a, b, la, lb in checks:
        t = paired_test(a, b, results, la, lb)
        stats.append(t)
        verdict = "CÓ ý nghĩa" if t["significant"] else "KHÔNG phân biệt được"
        print(f"  {t['comparison']:>42s}")
        print(f"      chênh RPS {t['mean_diff']:+.5f}  "
              f"KTC95% [{t['ci95'][0]:+.5f}, {t['ci95'][1]:+.5f}]  -> {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "n": len(df), "devig": best_devig,
            "model": m_model, "market": m_market,
            "best": {"method": best_method, "weight": best["weight"], "rps": best_rps},
            "grid": [{"weight": r["weight"], "log_rps": r["log"]["rps"],
                      "linear_rps": r["linear"]["rps"]} for r in table],
            "significance": stats,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
