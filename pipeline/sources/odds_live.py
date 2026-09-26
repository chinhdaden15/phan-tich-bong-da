"""
Kèo cho các trận sắp tới, lấy từ The Odds API.

Gói miễn phí cho 500 lượt gọi mỗi tháng. Pipeline chạy 3 lần/ngày cho 3 giải
tốn khoảng 270 lượt/tháng, nên vẫn còn dư.

Đăng ký key miễn phí: https://the-odds-api.com/

Không có key thì mọi hàm ở đây trả về rỗng và pipeline vẫn chạy bình thường,
chỉ là app không có phần so sánh với thị trường.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "premier-league": "soccer_epl",
    "la-liga": "soccer_spain_la_liga",
    "champions-league": "soccer_uefa_champs_league",
    "serie-a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue-1": "soccer_france_ligue_one",
    "nations-league": "soccer_uefa_nations_league",
}

# Nhà cái sắc nhất, dùng làm ưu tiên khi có mặt. Pinnacle nổi tiếng vì biên
# lợi nhuận thấp và điều chỉnh giá nhanh nhất theo thông tin mới.
PREFERRED_BOOKMAKERS = ("pinnacle", "betfair_ex_eu", "marathonbet")

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "odds_live"


class MissingApiKey(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Thiếu ODDS_API_KEY. Đăng ký miễn phí tại https://the-odds-api.com/ "
            "rồi đặt vào .env. Không có key thì app vẫn chạy, chỉ thiếu phần kèo."
        )


def has_key() -> bool:
    return bool(os.environ.get("ODDS_API_KEY", "").strip())


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def fetch_odds(league: str, cache_hours: float = 11.0, timeout: int = 30) -> list[dict]:
    """
    Kèo 1X2 cho các trận sắp tới của một giải.

    Có cache theo thời gian sống vì quota tính theo từng lượt gọi — gọi lại
    trong vòng vài giờ là lãng phí, kèo hiếm khi đổi mạnh ngoài sát giờ bóng lăn.
    """
    if league not in SPORT_KEYS:
        return []
    if not has_key():
        raise MissingApiKey()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{league}.json"

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_hours:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    response = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEYS[league]}/odds",
        params={
            "apiKey": os.environ["ODDS_API_KEY"].strip(),
            "regions": "eu",
            # Lấy thêm kèo tài/xỉu. Chỉ có 1X2 thì khi ép mô hình khớp thị
            # trường ta mới ghim được HIỆU SỐ bàn thắng, còn TỔNG bàn vẫn do
            # mô hình tự đoán — nên các tỷ số dự báo ra na ná nhau.
            # Mỗi thị trường tốn thêm 1 lượt quota. Pipeline chạy 4 lần/ngày
            # để kết quả trận luôn mới, nhưng cache 11 tiếng ở đây khiến kèo
            # chỉ thực sự được gọi khoảng 2 lần/ngày -> 360 lượt/tháng.
            # spreads = kèo chấp châu Á. Đây là kèo được đặt nhiều nhất ở
            # Việt Nam, và trước đây app không hề hiển thị nó. Mỗi thị trường
            # tốn thêm một lượt quota; với cache 11 tiếng thì cả ba thị trường
            # vẫn chỉ khoảng 540 lượt/tháng, dư trong hạn miễn phí.
            "markets": "h2h,totals,spreads",
            "oddsFormat": "decimal",
        },
        timeout=timeout,
    )
    if response.status_code == 401:
        raise MissingApiKey()
    if response.status_code == 429:
        raise RuntimeError("Hết quota The Odds API tháng này")
    response.raise_for_status()

    events = _parse(response.json(), league)
    cache_file.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    return events


def _parse(payload: list, league: str) -> list[dict]:
    """Rút kèo 1X2 từ phản hồi, gộp nhiều nhà cái thành một bộ giá."""
    out = []
    for event in payload:
        home = event.get("home_team")
        away = event.get("away_team")
        if not home or not away:
            continue

        # Gom giá theo từng cửa, giữ riêng giá của các nhà cái ưu tiên.
        collected = {"home": [], "draw": [], "away": []}
        preferred = {}
        totals = {}   # mốc tài/xỉu -> danh sách (giá tài, giá xỉu)
        spreads = {}  # mốc chấp (theo đội nhà) -> danh sách (giá nhà, giá khách)
        spread_preferred = {}

        for book in event.get("bookmakers", []):
            key = book.get("key", "")
            for market in book.get("markets", []):
                if market.get("key") == "spreads":
                    home_price = away_price = point = None
                    for oc in market.get("outcomes", []):
                        if oc.get("price") is None or oc.get("point") is None:
                            continue
                        if oc.get("name") == home:
                            home_price, point = float(oc["price"]), float(oc["point"])
                        elif oc.get("name") == away:
                            away_price = float(oc["price"])
                    if home_price and away_price and point is not None:
                        spreads.setdefault(point, []).append((home_price, away_price))
                        if key in PREFERRED_BOOKMAKERS and key not in spread_preferred:
                            spread_preferred[key] = (point, home_price, away_price)
                    continue

                if market.get("key") == "totals":
                    over = under = point = None
                    for oc in market.get("outcomes", []):
                        if oc.get("price") is None or oc.get("point") is None:
                            continue
                        if oc.get("name") == "Over":
                            over, point = float(oc["price"]), float(oc["point"])
                        elif oc.get("name") == "Under":
                            under = float(oc["price"])
                    if over and under and point is not None:
                        totals.setdefault(point, []).append((over, under))
                    continue

                if market.get("key") != "h2h":
                    continue
                prices = {}
                for oc in market.get("outcomes", []):
                    name = oc.get("name")
                    price = oc.get("price")
                    if price is None:
                        continue
                    if name == home:
                        prices["home"] = float(price)
                    elif name == away:
                        prices["away"] = float(price)
                    elif name == "Draw":
                        prices["draw"] = float(price)
                if len(prices) != 3:
                    continue
                for k, v in prices.items():
                    collected[k].append(v)
                if key in PREFERRED_BOOKMAKERS and key not in preferred:
                    preferred[key] = prices

        if not all(collected.values()):
            continue

        # Ưu tiên nhà cái sắc nhất; nếu không có thì lấy trung vị toàn thị trường.
        # Trung vị chống được giá lỗi của một nhà cái lẻ tốt hơn trung bình.
        chosen, source = None, None
        for name in PREFERRED_BOOKMAKERS:
            if name in preferred:
                chosen, source = preferred[name], name
                break
        if chosen is None:
            chosen = {k: _median(v) for k, v in collected.items()}
            source = f"median_of_{len(collected['home'])}"

        # Mốc tài/xỉu: ưu tiên 2.5 vì đó là mốc chuẩn và thanh khoản nhất;
        # nếu không có thì lấy mốc được nhiều nhà cái treo nhất.
        total_line = None
        if totals:
            point = 2.5 if 2.5 in totals else max(totals, key=lambda k: len(totals[k]))
            prices = totals[point]
            total_line = {
                "point": point,
                "over": _median([p[0] for p in prices]),
                "under": _median([p[1] for p in prices]),
                "bookmaker_count": len(prices),
            }

        # Kèo chấp: ưu tiên nhà cái sắc nhất vì mốc chấp giữa các nhà cái có
        # thể lệch nhau nửa trái, gộp trung vị sẽ ra một mốc không ai treo.
        handicap = None
        for name in PREFERRED_BOOKMAKERS:
            if name in spread_preferred:
                point, hp, ap = spread_preferred[name]
                handicap = {"line": point, "home": hp, "away": ap,
                            "source": name, "bookmaker_count": 1}
                break
        if handicap is None and spreads:
            point = max(spreads, key=lambda k: len(spreads[k]))
            prices = spreads[point]
            handicap = {
                "line": point,
                "home": _median([p[0] for p in prices]),
                "away": _median([p[1] for p in prices]),
                "source": f"median_of_{len(prices)}",
                "bookmaker_count": len(prices),
            }

        out.append({
            "league": league,
            "commence_time": event.get("commence_time"),
            "home_team": home,
            "away_team": away,
            "odds_home": chosen["home"],
            "odds_draw": chosen["draw"],
            "odds_away": chosen["away"],
            "odds_source": source,
            "bookmaker_count": len(collected["home"]),
            "total_line": total_line,
            "handicap": handicap,
        })
    return out
