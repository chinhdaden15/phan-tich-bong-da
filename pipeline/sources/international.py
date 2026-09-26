"""
Bóng đá đội tuyển: kết quả lịch sử, lịch thi đấu và bảng đấu Nations League.

VÌ SAO PHẢI CÓ NGUỒN RIÊNG: mọi nguồn còn lại của dự án đều là bóng đá CLB.
football-data.org gói miễn phí không có Nations League, football-data.co.uk chỉ
có giải quốc nội, còn openfootball không có đội tuyển. Ba thứ cần thì nằm ở ba
chỗ khác nhau.

KẾT QUẢ LỊCH SỬ lấy từ bộ dữ liệu cộng đồng martj42/international_results —
mọi trận đấu quốc tế từ 1872, cập nhật sau mỗi đợt tập trung. Vì nó trễ vài
tuần so với thực tế nên kết quả mới được vá thêm từ ECN, y như cách results.py
vá openfootball bằng football-data.org.

LƯU Ý QUAN TRỌNG VỀ SÂN TRUNG LẬP: mô hình có hệ số lợi thế sân nhà, nên các
trận sân trung lập (World Cup, Euro) phải bị loại khỏi tập khớp. Để lẫn vào,
hệ số sân nhà sẽ bị kéo tụt và mọi dự đoán đều lệch.
"""

from __future__ import annotations

import io
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from . import espn

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "international"
CACHE_HOURS = 12.0

# Mã ESPN của Nations League.
ESPN_CODE = "uefa.nations"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _cached_csv(url: str, name: str, cache_hours: float = CACHE_HOURS) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < cache_hours:
        return pd.read_csv(path)

    try:
        response = requests.get(url, timeout=60, headers=HEADERS)
        response.raise_for_status()
        raw = response.content
    except Exception:
        if path.exists():
            return pd.read_csv(path)
        raise
    path.write_bytes(raw)
    return pd.read_csv(io.BytesIO(raw))


def uefa_teams() -> set[str]:
    """Các đội tuyển từng dự Nations League — chính là 55 thành viên UEFA."""
    df = _cached_csv(RESULTS_URL, "results.csv")
    nl = df[df["tournament"] == "UEFA Nations League"]
    return set(nl["home_team"]) | set(nl["away_team"])


def results(since: str = "2010-01-01") -> pd.DataFrame:
    """
    Kết quả đã đá giữa các đội tuyển UEFA, chỉ tính trận có sân nhà - sân khách.

    Gộp mọi loại giải (Nations League, vòng loại World Cup, vòng loại Euro,
    giao hữu) chứ không riêng Nations League: một đội chỉ đá 6 trận Nations
    League mỗi hai năm, quá mỏng để ước lượng sức mạnh. Gộp lại thì trung vị
    lên khoảng 29 trận mỗi ba năm và mọi đội đều đủ dữ liệu.
    """
    df = _cached_csv(RESULTS_URL, "results.csv")
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_goals", "away_goals"])

    teams = uefa_teams()
    keep = (
        df["home_team"].isin(teams)
        & df["away_team"].isin(teams)
        & (~df["neutral"].astype(bool))
        & (df["date"] >= pd.Timestamp(since))
    )
    out = df[keep].copy()
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out["result"] = [
        "H" if h > a else ("A" if h < a else "D")
        for h, a in zip(out["home_goals"], out["away_goals"])
    ]
    out["league"] = "nations-league"
    out["season"] = out["date"].dt.year
    return out.sort_values("date").reset_index(drop=True)


def _scan(days: list[date]) -> list[dict]:
    """Quét lịch Nations League theo từng ngày qua ESPN."""
    out = []
    for day in days:
        payload = espn._cached_get(
            f"{espn.BASE}/{ESPN_CODE}/scoreboard?dates={day.strftime('%Y%m%d')}",
            f"{ESPN_CODE}_{day.strftime('%Y%m%d')}",
            cache_hours=3.0,
        )
        if not payload:
            continue
        for event in payload.get("events", []):
            comps = event.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            sides = comp.get("competitors") or []
            home = next((c for c in sides if c.get("homeAway") == "home"), None)
            away = next((c for c in sides if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            status = ((comp.get("status") or {}).get("type") or {})
            out.append({
                "event_id": event.get("id"),
                "date": day.isoformat(),
                "kickoff_utc": event.get("date"),
                "group": ((comp.get("group") or {}).get("name")),
                "home_team": (home.get("team") or {}).get("displayName"),
                "away_team": (away.get("team") or {}).get("displayName"),
                "home_crest": (home.get("team") or {}).get("logo"),
                "away_crest": (away.get("team") or {}).get("logo"),
                "home_goals": espn._to_int(home.get("score")),
                "away_goals": espn._to_int(away.get("score")),
                "completed": bool(status.get("completed")),
            })
    return out


def upcoming(days_ahead: int = 60) -> pd.DataFrame:
    """
    Lịch Nations League sắp tới.

    Quét 60 ngày vì đội tuyển đá theo ĐỢT TẬP TRUNG cách nhau 1-2 tháng: cửa sổ
    ngắn như giải CLB sẽ để tab trống suốt phần lớn thời gian.
    """
    today = datetime.utcnow().date()
    rows = [r for r in _scan([today + timedelta(days=i) for i in range(days_ahead + 1)])
            if not r["completed"]]
    if not rows:
        return pd.DataFrame(columns=["kickoff_utc", "home_team", "away_team",
                                     "group", "date", "home_crest", "away_crest"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("kickoff_utc").reset_index(drop=True)


def recent(days_back: int = 120) -> pd.DataFrame:
    """
    Kết quả Nations League vừa đá, dùng để vá phần bộ dữ liệu cộng đồng còn thiếu.

    Bộ đó cập nhật sau mỗi đợt tập trung nên có thể trễ vài tuần; thiếu bước này
    thì mô hình không biết gì về đợt đấu gần nhất và lịch sử dự đoán cũng không
    bao giờ chốt được.
    """
    today = datetime.utcnow().date()
    rows = [r for r in _scan([today - timedelta(days=i) for i in range(days_back + 1)])
            if r["completed"] and r["home_goals"] is not None]
    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team",
                                     "home_goals", "away_goals", "result", "group"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["result"] = [
        "H" if h > a else ("A" if h < a else "D")
        for h, a in zip(df["home_goals"], df["away_goals"])
    ]
    return df.sort_values("date").reset_index(drop=True)


def crests() -> dict[str, str]:
    """Bảng tra logo (cờ) theo tên đội, gom từ cả lịch sắp tới lẫn trận vừa đá."""
    out: dict[str, str] = {}
    for df in (upcoming(), recent()):
        for r in df.to_dict("records"):
            for side in ("home", "away"):
                name, logo = r.get(f"{side}_team"), r.get(f"{side}_crest")
                if name and logo:
                    out.setdefault(name, logo)
    return out
