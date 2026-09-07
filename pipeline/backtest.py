"""
Kiểm định cuốn chiếu (walk-forward) cho mô hình Dixon-Coles.

Nguyên tắc bất di bất dịch: khi dự đoán một trận, mô hình chỉ được nhìn
những trận đã đá **trước** ngày đó. Mọi vi phạm sẽ tạo ra kết quả đẹp giả tạo.

Dùng script này để chọn xi (tốc độ suy giảm thời gian) và ridge (mức co ngót),
thay vì áng chừng theo cảm tính.

    python pipeline/backtest.py --grid
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.dixon_coles import fit_dixon_coles
from model.evaluate import calibration_curve, evaluate
from model.markets import outcome_probabilities
from sources.openfootball import load

# Refit sau mỗi ngần này ngày. 7 là điểm cân bằng tốt: trong một tuần
# sức mạnh đội bóng gần như không đổi, nên refit dày hơn tốn thời gian
# mà hầu như không cải thiện độ chính xác.
REFIT_INTERVAL_DAYS = 7


def walk_forward(
    df: pd.DataFrame,
    test_start: str,
    xi: float,
    ridge: float,
    refit_days: int = REFIT_INTERVAL_DAYS,
    min_train: int = 200,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Chạy dự đoán cuốn chiếu và trả về bảng kèm xác suất từng cửa.

    min_train đảm bảo không dự đoán khi chưa đủ dữ liệu nền.
    """
    df = df.sort_values("date").reset_index(drop=True)
    cutoff = pd.Timestamp(test_start)
    test_idx = df.index[df["date"] >= cutoff]
    if len(test_idx) == 0:
        raise ValueError(f"Không có trận nào từ {test_start}")

    rows = []
    fit = None
    last_fit_date = None

    for i in test_idx:
        match = df.loc[i]
        match_date = match["date"]

        # Chỉ dùng trận đã đá TRƯỚC ngày thi đấu — chống rò rỉ dữ liệu.
        train = df[df["date"] < match_date]
        if len(train) < min_train:
            continue

        need_refit = (
            fit is None
            or last_fit_date is None
            or (match_date - last_fit_date).days >= refit_days
        )
        if need_refit:
            fit = fit_dixon_coles(
                train["home_team"], train["away_team"],
                train["home_goals"], train["away_goals"],
                match_dates=train["date"].values,
                reference_date=match_date.to_numpy().astype("datetime64[D]"),
                xi=xi, ridge=ridge, warm_start=fit,
            )
            last_fit_date = match_date
            if verbose:
                print(f"  refit @ {match_date.date()} trên {len(train)} trận")

        matrix = fit.score_matrix(
            match["home_team"], match["away_team"],
            *fit.expected_goals(match["home_team"], match["away_team"], allow_unknown=True),
        )
        probs = outcome_probabilities(matrix)
        conf = fit.confidence(match["home_team"], match["away_team"])

        rows.append({
            "date": match_date,
            "league": match["league"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "result": match["result"],
            "p_home": probs["home"],
            "p_draw": probs["draw"],
            "p_away": probs["away"],
            "confidence": conf["level"],
            "min_matches": min(conf["home_matches"], conf["away_matches"]),
        })

    return pd.DataFrame(rows)


def score(preds: pd.DataFrame, label: str) -> dict:
    """Tính bộ thước đo trên kết quả cuốn chiếu."""
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    return evaluate(probs, preds["result"].tolist(), label)


def baseline_uniform(preds: pd.DataFrame) -> dict:
    """Mốc sàn: đoán đều 1/3 mỗi cửa. Mô hình phải thắng mốc này."""
    probs = np.full((len(preds), 3), 1 / 3)
    return evaluate(probs, preds["result"].tolist(), "đoán đều 1/3")


def baseline_home_bias(preds: pd.DataFrame, df: pd.DataFrame) -> dict:
    """
    Mốc thứ hai: dùng tần suất kết quả trong quá khứ, không hề biết đội nào đá.
    Mô hình phải thắng rõ mốc này, nếu không thì nó chưa học được gì về đội bóng.
    """
    hist = df[df["date"] < preds["date"].min()]["result"].value_counts(normalize=True)
    base = np.array([hist.get("H", 0.44), hist.get("D", 0.26), hist.get("A", 0.30)])
    base = base / base.sum()
    probs = np.tile(base, (len(preds), 1))
    return evaluate(probs, preds["result"].tolist(), "tần suất lịch sử")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest mô hình Dixon-Coles")
    parser.add_argument("--leagues", nargs="+", default=["premier-league", "la-liga"])
    parser.add_argument("--from-season", type=int, default=2016)
    parser.add_argument("--test-start", default="2024-08-01")
    parser.add_argument("--xi", type=float, default=0.0018)
    parser.add_argument("--ridge", type=float, default=2.0)
    parser.add_argument("--grid", action="store_true", help="Dò lưới xi × ridge")
    parser.add_argument("--out", default=None, help="Ghi kết quả ra file JSON")
    args = parser.parse_args()

    seasons = list(range(args.from_season, 2027))
    print(f"Nạp {args.leagues} mùa {args.from_season}-2026 ...")
    df = load(args.leagues, seasons)
    print(f"  {len(df)} trận, {df.date.min().date()} -> {df.date.max().date()}\n")

    # Mỗi giải phải khớp mô hình riêng: sức mạnh không so sánh trực tiếp
    # được giữa các giải khi chưa có trận liên giải để hiệu chỉnh.
    def run(xi: float, ridge: float) -> tuple[pd.DataFrame, dict]:
        parts = []
        for lg in args.leagues:
            sub = df[df["league"] == lg]
            parts.append(walk_forward(sub, args.test_start, xi=xi, ridge=ridge))
        preds = pd.concat(parts, ignore_index=True)
        return preds, score(preds, f"xi={xi} ridge={ridge}")

    if args.grid:
        xis = [0.0, 0.001, 0.0018, 0.003, 0.005]
        ridges = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0]
        results = []
        print(f"Dò lưới {len(xis)}×{len(ridges)} = {len(xis)*len(ridges)} cấu hình\n")
        print(f"{'xi':>8} {'ridge':>7} {'RPS':>9} {'logloss':>9} {'acc':>7} {'giây':>6}")
        print("-" * 52)
        for xi, ridge in itertools.product(xis, ridges):
            t0 = time.time()
            preds, m = run(xi, ridge)
            elapsed = time.time() - t0
            results.append({"xi": xi, "ridge": ridge, **m, "seconds": round(elapsed, 1)})
            print(f"{xi:>8.4f} {ridge:>7.1f} {m['rps']:>9.5f} "
                  f"{m['log_loss']:>9.5f} {m['accuracy']:>7.3f} {elapsed:>6.1f}")

        best = min(results, key=lambda r: r["rps"])
        print("\n" + "=" * 52)
        print(f"TỐT NHẤT: xi={best['xi']}  ridge={best['ridge']}  "
              f"RPS={best['rps']:.5f}  n={best['n']}")
        payload = {"grid": results, "best": best}
    else:
        preds, m = run(args.xi, args.ridge)
        print("=== KẾT QUẢ ===")
        for row in (m, baseline_home_bias(preds, df), baseline_uniform(preds)):
            print(f"  {row['label']:>22s}  RPS={row['rps']:.5f}  "
                  f"logloss={row['log_loss']:.5f}  acc={row['accuracy']:.3f}  n={row['n']}")

        print("\n=== Theo mức tin cậy ===")
        for level in ("cao", "trung bình", "thấp"):
            sub = preds[preds["confidence"] == level]
            if len(sub) > 10:
                s = score(sub, level)
                print(f"  {level:>12s}  RPS={s['rps']:.5f}  acc={s['accuracy']:.3f}  n={s['n']}")

        print("\n=== Hiệu chuẩn ===")
        print(f"  {'khoảng dự đoán':>16s} {'dự đoán':>9s} {'thực tế':>9s} {'n':>6s}")
        probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
        for row in calibration_curve(probs, preds["result"].tolist(), n_bins=10):
            print(f"  {row['bin_low']:>7.2f}-{row['bin_high']:<8.2f} "
                  f"{row['predicted']:>9.3f} {row['observed']:>9.3f} {row['count']:>6d}")
        payload = {"metrics": m, "n": len(preds)}

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
