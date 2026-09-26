"""
Bộ điều phối: nạp dữ liệu, khớp mô hình, sinh dự đoán, ghi JSON tĩnh
cho app đọc.

Đây là toàn bộ phần "máy chủ" của dự án. Nó chạy trên GitHub Actions theo
lịch, ghi kết quả vào public/data/, rồi Cloudflare Pages phục vụ file tĩnh.
Điện thoại không bao giờ gọi API bên thứ ba — nhờ vậy vừa không chạm giới
hạn tốc độ, vừa tải nhanh, vừa xem được offline.

    python pipeline/build.py
    python pipeline/build.py --leagues premier-league --no-remote
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import history as history_mod
import selfcheck
from config import describe_keys, load_env
from model.adjustments import MatchContext
from model.cross_league import fit_cross_league
from model.dixon_coles import fit_dixon_coles
from model.extra_markets import (first_half_markets, measure_h1_share,
                                 player_scoring_rates,
                                 squad_scorer_probabilities)
from model.predictor import (_matrix_with_rho, compute_base_rates,
                            european_strength_table, handicap_block,
                            predict_european_match, predict_match,
                            predict_with_market, summarise_signal,
                            team_strength_table)
from model.corners import fit_corners
from sources import (espn, european, football_data_org, historical, odds_live,
                     openfootball, results as results_src)
from sources.odds_history import build_name_map, normalise

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "public" / "data"

CURRENT_SEASON = 2026
HISTORY_SEASONS = 6          # số mùa dùng để khớp mô hình
FIXTURE_WINDOW_DAYS = 14     # dự đoán các trận trong ngần này ngày tới

# Cúp châu Âu đá theo đợt cách nhau 2-3 tuần, nên cửa sổ 14 ngày để tab
# Champions League trống trơn suốt phần lớn thời gian giữa hai lượt. Nới ra
# 35 ngày thì luôn thấy được lượt kế tiếp. Giải quốc nội đá hằng tuần nên
# không cần, và giữ 14 ngày ở đó cũng giúp file dự đoán nhẹ.
CUP_FIXTURE_WINDOW_DAYS = 35

# Dưới ngần này trận trong cửa sổ thường thì coi như giải đang nghỉ và nới rộng.
MIN_FIXTURES_BEFORE_WIDENING = 6

# Tham số mô hình — chọn bằng dò lưới rồi KIỂM CHỨNG trên giai đoạn tách biệt.
#
# ridge=2.0: tốt nhất ở cả 5 mức xi trong dò lưới, và cũng tốt nhất trên giai
# đoạn giữ riêng 2022-2024. Bằng chứng nhất quán về hướng, nên chọn.
#
# xi=0.0018 (chu kỳ bán rã ~385 ngày): quán quân dò lưới thực ra là xi=0.0030,
# nhưng khi kiểm chứng trên giai đoạn tách biệt thì xi=0.0018 lại tốt hơn —
# dấu hiệu quán quân đã khớp quá mức vào chính tập dùng để chọn nó.
#
# Trung thực mà nói: bootstrap theo cặp cho thấy chênh lệch giữa các cấu hình
# này KHÔNG có ý nghĩa thống kê (khoảng tin cậy 95% đều chứa 0). Việc tinh
# chỉnh không mang lại cải thiện chứng minh được, chỉ là chọn cấu hình có
# bằng chứng nhất quán nhất và không có mặt trái.
MODEL_XI = 0.0018
MODEL_RIDGE = 2.0

# Múi giờ sân vận động của từng giải. Giờ trong dữ liệu openfootball là giờ
# địa phương nơi thi đấu, KHÔNG phải UTC — bỏ qua điều này thì một trận EPL
# 16:30 giờ Anh sẽ bị hiển thị sai 6 tiếng cho người xem ở Việt Nam, và các
# trận đá muộn rơi nhầm sang ngày hôm trước.
LEAGUE_TIMEZONES = {
    "premier-league": "Europe/London",
    "championship": "Europe/London",
    "la-liga": "Europe/Madrid",
    "serie-a": "Europe/Rome",
    "bundesliga": "Europe/Berlin",
    "ligue-1": "Europe/Paris",
    "eredivisie": "Europe/Amsterdam",
    "primeira-liga": "Europe/Lisbon",
    "champions-league": "Europe/Zurich",
}

LEAGUE_NAMES = {
    "premier-league": "Ngoại hạng Anh",
    "la-liga": "La Liga",
    "champions-league": "Champions League",
    "serie-a": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue-1": "Ligue 1",
}


def compute_standings(results: pd.DataFrame, season: int) -> list[dict]:
    """
    Tính bảng xếp hạng trực tiếp từ kết quả trận đấu.

    Tự tính thay vì gọi API vì kết quả đã có sẵn — tiết kiệm request và
    giữ cho app chạy được khi không có API key nào.
    """
    season_results = results[results["season"] == season]
    if season_results.empty:
        return []

    table: dict[str, dict] = {}

    def row(team: str) -> dict:
        return table.setdefault(team, {
            "team": team, "played": 0, "won": 0, "draw": 0, "lost": 0,
            "goals_for": 0, "goals_against": 0, "points": 0, "form": [],
        })

    for _, m in season_results.sort_values("date").iterrows():
        home, away = row(m["home_team"]), row(m["away_team"])
        hg, ag = int(m["home_goals"]), int(m["away_goals"])

        for side, gf, ga in ((home, hg, ag), (away, ag, hg)):
            side["played"] += 1
            side["goals_for"] += gf
            side["goals_against"] += ga

        if hg > ag:
            home["won"] += 1; home["points"] += 3; home["form"].append("W")
            away["lost"] += 1; away["form"].append("L")
        elif hg < ag:
            away["won"] += 1; away["points"] += 3; away["form"].append("W")
            home["lost"] += 1; home["form"].append("L")
        else:
            home["draw"] += 1; home["points"] += 1; home["form"].append("D")
            away["draw"] += 1; away["points"] += 1; away["form"].append("D")

    rows = list(table.values())
    for r in rows:
        r["goal_difference"] = r["goals_for"] - r["goals_against"]
        r["form"] = r["form"][-5:]   # 5 trận gần nhất

    # Thứ tự xếp hạng: điểm, rồi hiệu số, rồi bàn thắng ghi được.
    rows.sort(key=lambda r: (-r["points"], -r["goal_difference"], -r["goals_for"]))
    for i, r in enumerate(rows, 1):
        r["position"] = i
    return rows


def rest_days_map(results: pd.DataFrame, fixtures: pd.DataFrame) -> dict:
    """Số ngày nghỉ trước trận sắp tới, suy ra từ trận gần nhất của mỗi đội."""
    last_played: dict[str, pd.Timestamp] = {}
    for _, m in results.sort_values("date").iterrows():
        last_played[m["home_team"]] = m["date"]
        last_played[m["away_team"]] = m["date"]

    out = {}
    for _, f in fixtures.iterrows():
        for team in (f["home_team"], f["away_team"]):
            if team in last_played:
                out[(team, f["date"])] = max(0, (f["date"] - last_played[team]).days)
    return out


def kickoff_utc(date: pd.Timestamp, time_str: str | None, league: str) -> str | None:
    """
    Ghép ngày + giờ địa phương của sân thành mốc thời gian UTC chuẩn ISO 8601.

    App sẽ đổi mốc này sang múi giờ của thiết bị, nên người xem ở Việt Nam
    thấy đúng giờ Việt Nam mà không cần cấu hình gì.
    """
    # pd.isna bắt được cả None lẫn NaN. Dùng "not time_str" là không đủ:
    # NaN là truthy nên sẽ lọt qua, và phần lớn lịch La Liga thiếu giờ thi đấu.
    if time_str is None or pd.isna(time_str):
        return None
    try:
        hour, minute = (int(x) for x in str(time_str).split(":")[:2])
    except (ValueError, TypeError):
        return None

    tz = ZoneInfo(LEAGUE_TIMEZONES.get(league, "UTC"))
    local = datetime(
        date.year, date.month, date.day, hour, minute, tzinfo=tz
    )
    return local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_league(league: str, no_remote: bool = False) -> dict:
    """Dựng toàn bộ dữ liệu cho một giải."""
    print(f"\n=== {LEAGUE_NAMES.get(league, league)} ===")

    seasons = list(range(CURRENT_SEASON - HISTORY_SEASONS + 1, CURRENT_SEASON + 1))
    # Gộp thêm kết quả mới từ football-data.org: openfootball chỉ cập nhật
    # khoảng một lần mỗi tuần nên bảng xếp hạng sẽ trễ tới bảy ngày nếu
    # chỉ dựa vào nó.
    results = results_src.load_with_recent([league], seasons)
    print(f"  lịch sử: {len(results)} trận ({results.date.min().date()} -> {results.date.max().date()})")

    # Lưu lịch sử TRƯỚC khi predictions.json bị ghi đè ở cuối hàm này.
    # Bỏ qua bước này thì mọi dự đoán đã đưa ra sẽ mất khi trận đá xong.
    league_dir = OUT_DIR / league
    espn_details = espn.details_for_dates(league, espn.recent_days(10))
    archived, newly, mended = history_mod.archive_finished(
        league, league_dir, results, espn_details=espn_details
    )
    if newly:
        print(f"  lịch sử: +{newly} trận vừa đá xong (tổng {len(archived)})")
    if mended:
        print(f"  lịch sử: sửa lại {mended} bản ghi cũ")

    fit = fit_dixon_coles(
        results["home_team"], results["away_team"],
        results["home_goals"], results["away_goals"],
        match_dates=results["date"].values,
        xi=MODEL_XI, ridge=MODEL_RIDGE,
    )
    print(f"  mô hình: hội tụ={fit.converged}, sân nhà={fit.home_advantage:+.3f}, rho={fit.rho:+.3f}")

    # Bảng xếp hạng lấy từ API và bổ sung cột phong độ tự tính.
    #
    # Trước đây build.py tự tính còn refresh_live.py lấy API, hai nguồn không
    # bao giờ khớp nhau nên mỗi lần chạy hằng giờ lại ghi đè và tạo commit vô
    # ích, kéo theo một lần deploy. Một nguồn duy nhất thì so sánh mới có nghĩa.
    computed = compute_standings(results, CURRENT_SEASON)
    form_by_team = {normalise(r["team"]): r["form"] for r in computed}
    try:
        standings = football_data_org.standings(league)
        for row in standings:
            row["form"] = form_by_team.get(normalise(row["team"]), [])
        if not standings:
            raise LookupError("API trả bảng rỗng")
    except Exception as exc:
        # Không có API thì dùng bản tự tính — app vẫn chạy được.
        print(f"  bảng xếp hạng: dùng bản tự tính ({type(exc).__name__})")
        standings = computed

    position = {r["team"]: r["position"] for r in standings}
    total_teams = len(standings)
    print(f"  bảng xếp hạng: {total_teams} đội")

    fixtures = openfootball.upcoming_fixtures(league, CURRENT_SEASON).copy()
    fixtures["kickoff_utc"] = [
        kickoff_utc(row["date"], row.get("time"), league)
        for _, row in fixtures.iterrows()
    ]

    # So sánh theo mốc thời gian thật, không theo ngày. Trận thiếu giờ thi đấu
    # thì lùi về so sánh theo ngày với biên một ngày để không loại nhầm.
    now_utc = datetime.now(timezone.utc)

    def within(days: int):
        horizon = now_utc + timedelta(days=days)

        def in_window(row) -> bool:
            ko_raw = row["kickoff_utc"]
            if ko_raw is not None and not pd.isna(ko_raw):
                ko = datetime.fromisoformat(str(ko_raw).replace("Z", "+00:00"))
                return now_utc <= ko <= horizon
            day = row["date"].date()
            return now_utc.date() <= day <= horizon.date()

        return fixtures[fixtures.apply(in_window, axis=1)] if len(fixtures) else fixtures

    # Nới cửa sổ khi giải đang nghỉ.
    #
    # Dịp nghỉ FIFA kéo dài khoảng hai tuần, đúng bằng cửa sổ thường, nên app
    # từng trắng trơn cả hai giải quốc nội suốt thời gian đó — người dùng mở
    # lên chỉ thấy "không có trận nào" và tưởng app hỏng.
    #
    # Ngưỡng là SỐ TRẬN chứ không phải "rỗng hay không": bản đầu chỉ nới khi
    # không có trận nào, nên hôm La Liga vừa trở lại đúng một trận trong 14
    # ngày thì app hiện mỗi trận đó, trong khi Ngoại hạng Anh có 30. Một vòng
    # đấu bình thường có khoảng 10 trận, nên dưới 6 nghĩa là giải vẫn đang nghỉ.
    days = FIXTURE_WINDOW_DAYS
    window = within(days)
    if len(window) < MIN_FIXTURES_BEFORE_WIDENING:
        days = CUP_FIXTURE_WINDOW_DAYS
        window = within(days)

    upcoming_kickoff = next_kickoff(fixtures["kickoff_utc"], now_utc) if len(fixtures) else None
    print(f"  lịch: {len(window)} trận trong {days} ngày tới")

    # Kèo góc. Nguồn và mô hình hoàn toàn tách khỏi bàn thắng, nên hỏng ở đây
    # cũng không ảnh hưởng phần còn lại.
    corner_fit, corner_names = build_corner_model(league, seasons)

    rests = rest_days_map(results, window)
    base_rates = compute_base_rates(results)

    # Tỷ lệ bàn hiệp 1 đo từ chính giải này, dùng để suy kèo hiệp 1.
    h1_payloads = []
    for year in seasons:
        try:
            h1_payloads.append(openfootball._load_payload(league, year, True, 30))
        except Exception:
            continue
    h1_share, h1_n = measure_h1_share(h1_payloads)
    print(f"  hiệp 1: {h1_share:.1%} tổng bàn (đo trên {h1_n} trận)")

    scorer_rates, _ = load_scorers(league)
    squads = load_squads(league)
    print(f"  cầu thủ: {len(scorer_rates)} có dữ liệu ghi bàn, "
          f"{len(squads)} đội có đội hình đăng ký")

    # Tiến độ mùa giải, dùng cho hệ số động lực.
    played = len(results[results["season"] == CURRENT_SEASON])
    total_expected = total_teams * (total_teams - 1) if total_teams > 1 else 1
    season_progress = min(played / total_expected, 1.0) if total_expected else 0.0

    # Kèo cho trận sắp tới. Không có key thì bỏ qua, app vẫn chạy đủ.
    odds_index = {}
    if not no_remote and odds_live.has_key():
        try:
            events = odds_live.fetch_odds(league)
            odds_index = _index_odds(events, window)
            print(f"  kèo: khớp {len(odds_index)}/{len(window)} trận")
        except Exception as exc:
            print(f"  kèo: bỏ qua ({type(exc).__name__}: {exc})")
    elif not no_remote:
        print("  kèo: chưa có ODDS_API_KEY, bỏ qua phần thị trường")

    predictions = []
    for _, f in window.iterrows():
        context = MatchContext(
            home_rest_days=rests.get((f["home_team"], f["date"])),
            away_rest_days=rests.get((f["away_team"], f["date"])),
            home_position=position.get(f["home_team"]),
            away_position=position.get(f["away_team"]),
            total_teams=total_teams,
            season_progress=season_progress,
            # Không có nguồn dữ liệu chấn thương miễn phí nào cho mùa hiện
            # tại, nên hai danh sách này luôn rỗng. Lớp xử lý trong
            # adjustments.py vẫn giữ nguyên, sẵn sàng dùng nếu sau này có nguồn.
            home_missing=[],
            away_missing=[],
        )
        pred = predict_match(fit, f["home_team"], f["away_team"], context)
        pred["fixture"] = {
            "date": f["date"].strftime("%Y-%m-%d"),
            "time": f.get("time"),
            "kickoff_utc": (
                None if pd.isna(f.get("kickoff_utc")) else f.get("kickoff_utc")
            ),
            "round": f.get("round"),
            "league": league,
        }
        pred["head_to_head"] = head_to_head(results, f["home_team"], f["away_team"])
        pred["corners"] = corner_block(
            corner_fit, corner_names, f["home_team"], f["away_team"])
        # Kèo hiệp 1 suy từ bàn thắng kỳ vọng cả trận. Dùng số đã hiệu chỉnh
        # theo kèo nếu có, vì nó chính xác hơn số thuần mô hình.
        _eg = pred["expected_goals"]
        pred["first_half"] = first_half_markets(_eg["home"], _eg["away"], h1_share)
        pred["scorers"] = {
            "home": squad_scorer_probabilities(
                squads.get(f["home_team"], []), scorer_rates, _eg["home"]),
            "away": squad_scorer_probabilities(
                squads.get(f["away_team"], []), scorer_rates, _eg["away"]),
            "basis": "squad",
        }

        event = odds_index.get((f["home_team"], f["away_team"]))
        pred["market"] = (
            predict_with_market(
                fit, pred,
                event["odds_home"], event["odds_draw"], event["odds_away"],
                odds_source=event.get("odds_source", "unknown"),
                total_line=event.get("total_line"),
                handicap=event.get("handicap"),
            )
            if event else {"available": False, "reason": "chưa có kèo"}
        )
        if (pred.get("market") or {}).get("available"):
            _mg = pred["market"]["expected_goals"]
            pred["first_half"] = first_half_markets(_mg["home"], _mg["away"], h1_share)
            pred["scorers"] = {
                "home": squad_scorer_probabilities(
                    squads.get(f["home_team"], []), scorer_rates, _mg["home"]),
                "away": squad_scorer_probabilities(
                    squads.get(f["away_team"], []), scorer_rates, _mg["away"]),
                "basis": "squad",
            }
        pred["signal"] = summarise_signal(pred, base_rates)
        pred["handicap"] = market_handicap(pred, event, f["home_team"], f["away_team"])
        predictions.append(pred)

    print(f"  đã sinh {len(predictions)} dự đoán")

    # Mô hình được khớp trên 6 mùa nên biết cả những đội đã xuống hạng từ lâu.
    # Bảng sức mạnh hiển thị cho người dùng chỉ nên gồm các đội đang đá mùa này,
    # nếu không Ngoại hạng Anh sẽ hiện 29 đội thay vì 20.
    current_teams = {r["team"] for r in standings}
    if not current_teams:
        current_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])

    strength = [r for r in team_strength_table(fit) if r["team"] in current_teams]
    for i, row in enumerate(strength, 1):
        row["rank"] = i
    print(f"  bảng sức mạnh: {len(strength)} đội đang thi đấu")

    return {
        "league": league,
        "name": LEAGUE_NAMES.get(league, league),
        "season": CURRENT_SEASON,
        "season_progress": round(season_progress, 3),
        "standings": standings,
        "strength": strength,
        "predictions": predictions,
        "next_fixture_utc": upcoming_kickoff,
        "history": archived,
        "accuracy": history_mod.accuracy_summary(archived),
        "model": {
            "xi": MODEL_XI,
            "ridge": MODEL_RIDGE,
            "home_advantage": round(fit.home_advantage, 4),
            "rho": round(fit.rho, 4),
            "trained_on_matches": fit.n_matches,
            "converged": fit.converged,
        },
    }


def build_champions_league() -> dict:
    """
    Dựng dữ liệu Champions League bằng mô hình phân tầng liên giải.

    Không dùng được mô hình một giải như các giải quốc nội: mỗi đội chỉ đá
    khoảng 8 trận vòng bảng mỗi mùa, quá ít để ước lượng sức mạnh. Thay vào
    đó dùng toàn bộ kết quả quốc nội của 12 giải châu Âu, với chính các trận
    C1 làm cầu nối hiệu chỉnh mặt bằng giữa các giải.
    """
    print(f"\n=== {LEAGUE_NAMES['champions-league']} ===")

    pool, team_league, display_map = european.build_dataset(
        domestic_seasons=list(range(CURRENT_SEASON - 3, CURRENT_SEASON + 1)),
        cl_seasons=[CURRENT_SEASON - 2, CURRENT_SEASON - 1],
    )
    bridge = int(pool["bridge"].sum())
    print(f"  gộp: {len(pool)} trận ({bridge} trận C1 làm cầu nối), "
          f"{len(set(pool.home_team) | set(pool.away_team))} đội")

    league_dir = OUT_DIR / "champions-league"

    # Kết quả C1 MÙA NÀY, lấy riêng từ API.
    #
    # Không dùng pool[pool["bridge"]] được: tập gộp chỉ chứa C1 của hai mùa
    # TRƯỚC (dùng làm cầu nối hiệu chỉnh mặt bằng giải), hoàn toàn không có
    # trận mùa hiện tại. Lấy nhầm nguồn đó gây hai lỗi cùng lúc — cột phong độ
    # hiện 5 trận trong khi đội mới đá 1, và lịch sử dự đoán không bao giờ
    # lưu được vì không tìm thấy kết quả để đối chiếu.
    cl_rows = []
    for m in football_data_org.season_matches("champions-league", CURRENT_SEASON):
        if m["home_goals"] is None:
            continue
        cl_rows.append({
            "date": pd.Timestamp(m["date"]),
            "home_team": m["home_team"], "away_team": m["away_team"],
            "home_goals": m["home_goals"], "away_goals": m["away_goals"],
            "result": ("H" if m["home_goals"] > m["away_goals"]
                       else "A" if m["home_goals"] < m["away_goals"] else "D"),
        })
    cl_results = pd.DataFrame(cl_rows, columns=[
        "date", "home_team", "away_team", "home_goals", "away_goals", "result"
    ])
    print(f"  kết quả C1 mùa này: {len(cl_results)} trận đã đá")
    espn_details = espn.details_for_dates("champions-league", espn.recent_days(10))
    archived, newly, mended = history_mod.archive_finished(
        "champions-league", league_dir, cl_results, espn_details=espn_details
    )
    if newly:
        print(f"  lịch sử: +{newly} trận vừa đá xong (tổng {len(archived)})")
    if mended:
        print(f"  lịch sử: sửa lại {mended} bản ghi cũ")

    fit = fit_cross_league(
        pool["home_team"], pool["away_team"],
        pool["home_goals"], pool["away_goals"],
        team_league,
        match_dates=pool["date"].values,
        is_bridge=pool["bridge"].values,
        xi=MODEL_XI, ridge=MODEL_RIDGE,
    )
    print(f"  mô hình: hội tụ={fit.converged}, sân nhà={fit.home_advantage:+.3f}, "
          f"rho={fit.rho:+.3f}")

    levels = sorted(
        fit.leagues, key=lambda l: -(fit.league_attack[l] - fit.league_defence[l])
    )
    print(f"  mặt bằng giải: {levels[0]} mạnh nhất, {levels[-1]} yếu nhất")

    # Tỷ lệ bàn hiệp 1 cho C1: đo trên các giải quốc nội lớn, vì chính các CLB
    # đó đá Champions League. Bản thân C1 chưa đủ trận trong mùa để tự đo.
    h1_payloads = []
    for lg in ("premier-league", "la-liga", "serie-a", "bundesliga", "ligue-1"):
        for year in range(CURRENT_SEASON - 3, CURRENT_SEASON + 1):
            try:
                h1_payloads.append(openfootball._load_payload(lg, year, True, 30))
            except Exception:
                continue
    h1_share, h1_n = measure_h1_share(h1_payloads)
    print(f"  hiệp 1: {h1_share:.1%} tổng bàn (đo trên {h1_n} trận 5 giải lớn)")
    scorer_rates, _ = load_scorers("champions-league")
    squads = load_squads("champions-league")
    print(f"  cầu thủ: {len(scorer_rates)} có dữ liệu ghi bàn, "
          f"{len(squads)} đội có đội hình đăng ký")

    fixtures = european.upcoming_champions_league(CURRENT_SEASON)
    now_utc = datetime.now(timezone.utc)
    horizon = now_utc + timedelta(days=CUP_FIXTURE_WINDOW_DAYS)

    def in_window(ko: str) -> bool:
        try:
            t = datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return False
        return now_utc <= t <= horizon

    window = fixtures[fixtures["kickoff_utc"].map(in_window)] if len(fixtures) else fixtures
    upcoming_kickoff = next_kickoff(fixtures["kickoff_utc"], now_utc) if len(fixtures) else None
    print(f"  lịch: {len(window)} trận trong {CUP_FIXTURE_WINDOW_DAYS} ngày tới")

    # Kèo góc cho C1: gộp dữ liệu góc của mọi giải quốc nội mà nguồn có, rồi
    # tra từng CLB theo tên. CLB nào không nằm trong các giải đó (chủ yếu là
    # Áo, Azerbaijan, Na Uy) sẽ không có kèo góc — app bỏ qua phần đó thay vì
    # bịa ra con số.
    # Dùng đúng khoảng mùa mà mô hình xuyên giải đang dùng cho phần quốc nội.
    corner_fit, corner_names = build_corner_model(
        CORNER_LEAGUES, list(range(CURRENT_SEASON - 3, CURRENT_SEASON + 1)))

    # Quy đổi tên theo chính bảng dữ liệu đã huấn luyện, không dùng lại bảng
    # dựng từ các mùa C1 trước — mùa nào cũng có CLB lần đầu dự giải.
    fixture_names = sorted(set(window["home_team"]) | set(window["away_team"])) if len(window) else []
    resolved, unresolved = european.resolve_against_pool(fixture_names, pool)
    if unresolved:
        print(f"  CẢNH BÁO: {len(unresolved)} CLB không có dữ liệu quốc nội "
              f"({', '.join(unresolved[:4])}) — dự đoán cho họ sẽ bị đánh dấu tin cậy thấp")

    odds_index = {}
    if odds_live.has_key() and len(window):
        try:
            events = odds_live.fetch_odds("champions-league")
            odds_index = _index_odds(events, window)
            print(f"  kèo: khớp {len(odds_index)}/{len(window)} trận")
        except Exception as exc:
            print(f"  kèo: bỏ qua ({type(exc).__name__})")

    base_rates = compute_base_rates(pool)
    predictions = []
    for _, f in window.iterrows():
        home_key = resolved.get(f["home_team"])
        away_key = resolved.get(f["away_team"])
        missing = [t for t, k in ((f["home_team"], home_key), (f["away_team"], away_key)) if k is None]
        home_key = home_key or normalise(f["home_team"])
        away_key = away_key or normalise(f["away_team"])

        pred = predict_european_match(
            fit, home_key, away_key,
            neutral=(str(f.get("stage", "")).upper() == "FINAL"),
        )
        # Hiển thị tên gốc của nguồn, không phải tên đã chuẩn hoá để ghép.
        pred["home_team"] = f["home_team"]
        pred["away_team"] = f["away_team"]
        pred["fixture"] = {
            "date": f["date"].strftime("%Y-%m-%d"),
            "kickoff_utc": f.get("kickoff_utc"),
            "round": f.get("stage"),
            "matchday": f.get("matchday"),
            "league": "champions-league",
        }
        pred["head_to_head"] = head_to_head(pool, home_key, away_key)
        _eg = pred["expected_goals"]
        pred["first_half"] = first_half_markets(_eg["home"], _eg["away"], h1_share)
        if missing:
            # Đội không có dữ liệu nào: mô hình mặc định coi họ là trung bình
            # châu Âu, gần như chắc chắn là đánh giá quá cao. Nói thẳng ra
            # thay vì để con số trông như mọi dự đoán khác.
            pred["confidence"]["level"] = "thấp"
            pred["confidence"]["no_data_teams"] = missing
        pred["market"] = pred.get("market")
        pred["corners"] = corner_block(
            corner_fit, corner_names, f["home_team"], f["away_team"])

        event = odds_index.get((f["home_team"], f["away_team"]))
        pred["market"] = (
            predict_with_market(
                fit, pred,
                event["odds_home"], event["odds_draw"], event["odds_away"],
                odds_source=event.get("odds_source", "unknown"),
                # Nhánh này trước đây bỏ quên hai tham số dưới, nên ma trận tỷ
                # số của C1 chỉ khớp 1X2 và tổng bàn hoàn toàn do mô hình đoán.
                total_line=event.get("total_line"),
                handicap=event.get("handicap"),
            )
            if event else {"available": False, "reason": "chưa có kèo"}
        )
        if (pred.get("market") or {}).get("available"):
            _mg = pred["market"]["expected_goals"]
            pred["first_half"] = first_half_markets(_mg["home"], _mg["away"], h1_share)
            pred["scorers"] = {
                "home": squad_scorer_probabilities(
                    squads.get(f["home_team"], []), scorer_rates, _mg["home"]),
                "away": squad_scorer_probabilities(
                    squads.get(f["away_team"], []), scorer_rates, _mg["away"]),
                "basis": "squad",
            }
        pred["signal"] = summarise_signal(pred, base_rates)
        pred["handicap"] = market_handicap(pred, event, f["home_team"], f["away_team"])
        predictions.append(pred)

    print(f"  đã sinh {len(predictions)} dự đoán")

    # Bảng sức mạnh chỉ gồm các CLB đang dự C1 mùa này.
    all_names = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"])) if len(fixtures) else []
    all_resolved, _ = european.resolve_against_pool(all_names, pool)
    participants = set(all_resolved.values())
    resolved_by_display = dict(all_resolved)
    strength = european_strength_table(fit, sorted(participants))

    try:
        standings = football_data_org.standings("champions-league")
        # API không trả phong độ 5 trận gần nhất, tự tính từ kết quả để bảng
        # C1 có cùng bộ trường với bảng các giải quốc nội. Thiếu trường này
        # từng làm cả tab bảng xếp hạng trắng trơn vì giao diện gọi r.form.map().
        # Cả hai phía đều quy về tên đã chuẩn hoá trước khi tra, nếu không
        # "Manchester City FC" và "manchester city" thành hai đội khác nhau
        # và cột phong độ rỗng trơn.
        form_by_team = {}
        for _, m in cl_results.sort_values("date").iterrows():
            for team, res in (
                (m["home_team"], "W" if m["result"] == "H" else ("L" if m["result"] == "A" else "D")),
                (m["away_team"], "W" if m["result"] == "A" else ("L" if m["result"] == "H" else "D")),
            ):
                form_by_team.setdefault(normalise(team), []).append(res)
        for row in standings:
            row["form"] = form_by_team.get(normalise(row["team"]), [])[-5:]
        print(f"  bảng xếp hạng: {len(standings)} đội")
    except LookupError:
        # Vòng bảng chưa đá trận nào thì chưa có bảng — bình thường đầu mùa.
        print("  bảng xếp hạng: chưa có (mùa giải chưa bắt đầu)")
        standings = []
    except Exception as exc:
        print(f"  bảng xếp hạng: bỏ qua ({type(exc).__name__}: {exc})")
        standings = []

    return {
        "league": "champions-league",
        "name": LEAGUE_NAMES["champions-league"],
        "season": CURRENT_SEASON,
        "season_progress": 0.0,
        "standings": standings,
        "strength": strength,
        "predictions": predictions,
        "next_fixture_utc": upcoming_kickoff,
        "history": archived,
        "accuracy": history_mod.accuracy_summary(archived),
        "model": {
            "xi": MODEL_XI, "ridge": MODEL_RIDGE,
            "home_advantage": round(fit.home_advantage, 4),
            "rho": round(fit.rho, 4),
            "trained_on_matches": fit.n_matches,
            "bridge_matches": fit.n_bridge_matches,
            "converged": fit.converged,
            "league_levels": {
                l: round(fit.league_attack[l] - fit.league_defence[l], 3)
                for l in fit.leagues
            },
        },
    }


SCORER_COMPETITIONS = {
    "premier-league": ["PL"],
    "la-liga": ["PD"],
    # Cầu thủ dự C1 đến từ khắp châu Âu nên phải gộp nhiều giải quốc nội.
    "champions-league": ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL"],
}


def load_squads(league: str) -> dict[str, list[dict]]:
    """
    Đội hình đăng ký của mọi CLB trong giải, khoá theo tên đội.

    Cache một tuần vì danh sách chỉ đổi khi có chuyển nhượng. Lần chạy đầu tốn
    một request mỗi CLB nên hơi lâu, các lần sau đọc thẳng từ cache.
    """
    try:
        teams = football_data_org.competition_teams(league)
    except Exception as exc:
        print(f"  đội hình đăng ký: bỏ qua ({type(exc).__name__})")
        return {}

    squads = {}
    for team in teams:
        try:
            players = football_data_org.squad(team["id"])
        except Exception:
            continue
        if players:
            squads[team["name"]] = players
    return squads


def load_scorers(league: str) -> tuple[dict, dict]:
    """Tỷ lệ ghi bàn của cầu thủ và danh sách cầu thủ theo đội."""
    codes = SCORER_COMPETITIONS.get(league, [])
    current, previous = [], []
    for code in codes:
        for season, bucket in ((None, current), (CURRENT_SEASON - 1, previous)):
            params = {"limit": 100}
            if season is not None:
                params["season"] = season
            try:
                bucket.extend(football_data_org._get(
                    f"competitions/{code}/scorers", params,
                    cache_hours=12.0 if season is None else 24 * 7,
                ).get("scorers", []))
            except Exception:
                continue

    rates = player_scoring_rates(current, previous)

    # Đội hiện tại của cầu thủ lấy theo mùa NÀY, vì cầu thủ đổi CLB giữa các mùa.
    team_players: dict[str, list[str]] = {}
    for entry in current:
        player = (entry.get("player") or {}).get("name")
        team = (entry.get("team") or {}).get("name")
        if player and team:
            team_players.setdefault(team, []).append(player)
    return rates, team_players


def _index_odds(events: list[dict], fixtures: pd.DataFrame) -> dict:
    """
    Ghép sự kiện từ nhà cái với lịch thi đấu theo tên đội đã chuẩn hoá.

    Nguồn kèo dùng tên ngắn ("Manchester City") còn openfootball dùng tên đầy
    đủ ("Manchester City FC"), nên phải đi qua cùng bộ chuẩn hoá đã dùng cho
    kèo lịch sử. Ghép sai đội sẽ gán kèo của trận khác mà không báo lỗi.
    """
    if not events:
        return {}

    fixture_names = set(fixtures["home_team"].map(normalise)) | set(
        fixtures["away_team"].map(normalise)
    )
    odds_names = {normalise(e["home_team"]) for e in events} | {
        normalise(e["away_team"]) for e in events
    }
    name_map = build_name_map(odds_names, fixture_names)

    by_pair = {}
    for e in events:
        home = name_map.get(normalise(e["home_team"]), normalise(e["home_team"]))
        away = name_map.get(normalise(e["away_team"]), normalise(e["away_team"]))
        by_pair[(home, away)] = e

    out = {}
    for _, f in fixtures.iterrows():
        key = (normalise(f["home_team"]), normalise(f["away_team"]))
        if key in by_pair:
            out[(f["home_team"], f["away_team"])] = by_pair[key]
    return out


def head_to_head(results: pd.DataFrame, home: str, away: str, limit: int = 6) -> dict:
    """Lịch sử đối đầu gần nhất giữa hai đội, bất kể sân nào."""
    mask = (
        ((results["home_team"] == home) & (results["away_team"] == away))
        | ((results["home_team"] == away) & (results["away_team"] == home))
    )
    h2h = results[mask].sort_values("date", ascending=False).head(limit)

    matches = [{
        "date": m["date"].strftime("%Y-%m-%d"),
        "home_team": m["home_team"],
        "away_team": m["away_team"],
        "home_goals": int(m["home_goals"]),
        "away_goals": int(m["away_goals"]),
    } for _, m in h2h.iterrows()]

    wins = sum(
        1 for m in matches
        if (m["home_team"] == home and m["home_goals"] > m["away_goals"])
        or (m["away_team"] == home and m["away_goals"] > m["home_goals"])
    )
    draws = sum(1 for m in matches if m["home_goals"] == m["away_goals"])

    return {
        "matches": matches,
        "summary": {
            "played": len(matches),
            "home_wins": wins,
            "draws": draws,
            "away_wins": len(matches) - wins - draws,
        },
    }


def json_safe(value):
    """
    Dọn NaN/NaT thành None trước khi ghi JSON.

    NaN là giá trị hợp lệ trong Python nhưng KHÔNG hợp lệ trong JSON. Nếu để
    lọt, json.dumps vẫn ghi ra chữ "NaN" và file trông như bình thường — nhưng
    JSON.parse của trình duyệt từ chối đọc cả file. Toàn bộ tab La Liga từng
    trắng trơn vì đúng lỗi này: phần lớn trận La Liga không có giờ thi đấu nên
    trường "time" là NaN, trong khi Ngoại hạng Anh trận nào cũng có giờ nên
    không hề lộ ra.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT or (value is not None and value is pd.NA):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dump_json(path: Path, payload, indent: int | None = None) -> None:
    """
    Ghi JSON đã dọn NaN, và bật allow_nan=False để lỗi tương lai kêu to.

    allow_nan=False biến mọi NaN sót lại thành ValueError ngay lúc build,
    thay vì thành một tab trắng mà không ai biết vì sao.
    """
    path.write_text(
        json.dumps(
            json_safe(payload), ensure_ascii=False, allow_nan=False,
            separators=None if indent else (",", ":"), indent=indent,
        ),
        encoding="utf-8",
    )


def market_handicap(pred: dict, event: dict | None, home: str, away: str) -> dict | None:
    """
    Khối kèo chấp cho một trận, dựng từ ma trận tỷ số đã ép khớp thị trường.

    Dùng ma trận đã hiệu chỉnh chứ không phải ma trận thuần mô hình, để xác
    suất ăn kèo nói cùng một thứ tiếng với các con số khác trong app.
    """
    if not event or not (event.get("handicap")):
        return None

    market = pred.get("market") or {}
    available = bool(market.get("available"))
    source = market["prediction"] if available else pred["prediction"]

    # Dựng lại ma trận từ tham số đã ép khớp thị trường khi có, để xác suất ăn
    # kèo chấp nhất quán với mọi con số khác trong app.
    try:
        if available:
            eg, rho = market["expected_goals"], float(market.get("rho") or 0.0)
        else:
            eg, rho = pred["expected_goals"], 0.0
        matrix = _matrix_with_rho(float(eg["home"]), float(eg["away"]), rho)
    except (KeyError, TypeError, ValueError):
        return None

    return handicap_block(matrix, event["handicap"], home, away,
                          scores=source.get("correct_score", []))


# Các mốc kèo góc mà nhà cái hay treo.
CORNER_LINES = (8.5, 9.5, 10.5, 11.5)

# Các giải quốc nội có dữ liệu góc, dùng để dựng mô hình góc cho Champions
# League. Bỏ hạng Nhất Anh vì không CLB nào ở đó dự cúp châu Âu.
CORNER_LEAGUES = [l for l in historical.DIVISIONS if l != "championship"]


def build_corner_model(league: str | list[str], seasons: list[int]):
    """
    Khớp mô hình phạt góc, kèm bảng quy đổi tên đội sang tên mà app dùng.

    Nguồn góc là football-data.co.uk, viết tên đội khác hẳn nguồn lịch thi đấu
    ("Man City" so với "Manchester City FC"), nên phải có bảng quy đổi mới tra
    được. Trả về (fit, mapping) hoặc (None, {}) khi không có dữ liệu — góc chỉ
    là phần thêm, thiếu nó không được làm hỏng cả lần build.

    Nhận NHIỀU giải để dùng cho Champions League: C1 không có dữ liệu góc riêng,
    nên sức mạnh góc của mỗi CLB phải lấy từ giải quốc nội của họ. Gộp chung
    một lần khớp thay vì khớp từng giải rồi ghép, để mọi đội nằm chung một
    thang đo — hệt cách mô hình bàn thắng xuyên giải đang làm.
    """
    leagues = [league] if isinstance(league, str) else list(league)
    try:
        df = historical.load_leagues(leagues, seasons)
    except Exception as exc:
        print(f"  góc: bỏ qua ({type(exc).__name__}: {exc})")
        return None, {}

    df = df.dropna(subset=["home_corners", "away_corners"])
    if len(df) < 500:
        print(f"  góc: chỉ có {len(df)} trận, quá ít để khớp — bỏ qua")
        return None, {}

    # Với bản gộp nhiều giải (hơn 500 tham số), L-BFGS-B báo converged=False.
    # Đã kiểm: chạy 400, 4.000 và 20.000 vòng cho ra tham số GIỐNG HỆT nhau,
    # chênh lệch đúng bằng 0 — nên đó là báo động giả của điều kiện dừng, lời
    # giải đã tối ưu. Không nâng số vòng vì chẳng thay đổi gì.
    fit = fit_corners(
        df["home_team"], df["away_team"], df["home_corners"], df["away_corners"],
        match_dates=df["date"].values, xi=MODEL_XI, ridge=MODEL_RIDGE,
    )
    print(f"  góc: {len(df)} trận, hội tụ={fit.converged}, "
          f"sân nhà={fit.home_advantage:+.3f}")
    return fit, {normalise(t): t for t in fit.teams}


def corner_block(fit, source_names: dict, home: str, away: str) -> dict | None:
    """Khối dự đoán góc cho một trận, hoặc None nếu thiếu dữ liệu một trong hai đội."""
    if fit is None:
        return None

    mapping = build_name_map(set(source_names), {normalise(home), normalise(away)})

    def resolve(name: str) -> str | None:
        key = normalise(name)
        for src_key, original in source_names.items():
            if src_key == key or mapping.get(src_key) == key:
                return original
        return None

    h, a = resolve(home), resolve(away)
    if h is None or a is None:
        return None

    lam_h, lam_a = fit.expected_corners(h, a)
    return {
        "expected": {"home": round(lam_h, 2), "away": round(lam_a, 2),
                     "total": round(lam_h + lam_a, 2)},
        "over_under": {
            str(line): {k: round(v, 4) for k, v in fit.over_under(h, a, line).items()}
            for line in CORNER_LINES
        },
        "confidence": fit.confidence(h, a),
    }


def next_kickoff(kickoffs, now_utc: datetime) -> str | None:
    """
    Giờ bóng lăn của trận sắp tới gần nhất, kể cả ngoài cửa sổ dự đoán.

    App dùng nó để thay câu "không có trận nào" bằng ngày cụ thể của lượt kế
    tiếp — cần nhất ở Champions League, nơi hai lượt cách nhau vài tuần.
    """
    best = None
    for raw in kickoffs:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            ko = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        # Chuỗi thiếu phần múi giờ cho ra datetime "ngây thơ", mà so sánh nó
        # với datetime có múi giờ sẽ ném TypeError và làm sập cả lần build chỉ
        # vì một ô dữ liệu lệch định dạng. Coi như giờ UTC.
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        if ko >= now_utc and (best is None or ko < best):
            best = ko
    return best.isoformat().replace("+00:00", "Z") if best else None


def write_index(out_dir: Path, built: list[dict]) -> None:
    """
    Ghi index.json, GIỮ LẠI các giải đã dựng từ lần chạy trước.

    Trước đây hàm này ghi đè thẳng danh sách giải vừa dựng. Hệ quả: chạy
    `build.py --leagues champions-league` để cập nhật riêng một giải sẽ xoá
    Ngoại hạng Anh và La Liga khỏi index, và app mất luôn hai tab đó dù file
    dữ liệu vẫn còn nguyên trên đĩa.

    Giờ danh sách cũ được gộp vào, chỉ loại những giải mà file dữ liệu đã
    thực sự biến mất.
    """
    index_file = out_dir / "index.json"
    merged: dict[str, dict] = {}

    if index_file.exists():
        try:
            previous = json.loads(index_file.read_text(encoding="utf-8"))
            for entry in previous.get("leagues", []):
                league_id = entry.get("id")
                if league_id and (out_dir / league_id / "predictions.json").exists():
                    merged[league_id] = entry
        except (json.JSONDecodeError, OSError):
            # Index hỏng thì dựng lại từ đầu, không để nó chặn cả lần build.
            merged = {}

    for entry in built:
        merged[entry["id"]] = entry

    # Giữ thứ tự cố định để tab trong app không nhảy chỗ giữa các lần build.
    order = ["premier-league", "la-liga", "champions-league"]
    leagues = sorted(
        merged.values(),
        key=lambda e: (order.index(e["id"]) if e["id"] in order else len(order), e["id"]),
    )

    keys = describe_keys()
    dump_json(index_file, {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Lịch cập nhật, để app tự tính và hiển thị đếm ngược. Phải khớp với
        # cron trong .github/workflows/refresh-live.yml.
        "refresh": {
            "interval_hours": 3,
            "minute_utc": 5,
            # Cron của GitHub Actions gói miễn phí hay trễ; đo trên repo này
            # thấy 6-37 phút. Cộng khoảng đệm để đếm ngược không về 0 rồi treo.
            "typical_delay_minutes": 25,
        },
        "leagues": leagues,
        "model": {"type": "dixon-coles", "xi": MODEL_XI, "ridge": MODEL_RIDGE},
        # Chỉ ghi CÓ hay KHÔNG, tuyệt đối không ghi giá trị key. File này
        # được phục vụ công khai cho mọi người mở app.
        "sources": {
            "odds_api": {"configured": keys["ODDS_API_KEY"], "purpose": "kèo trận sắp tới"},
            "football_data_org": {"configured": keys["FOOTBALL_DATA_ORG_KEY"],
                                  "purpose": "lịch và kết quả Champions League"},
            "openfootball": {"configured": True, "purpose": "kết quả và lịch, không cần key"},
        },
    }, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dựng dữ liệu tĩnh cho app")
    parser.add_argument("--leagues", nargs="+",
                        default=["premier-league", "la-liga", "champions-league"])
    parser.add_argument("--no-remote", action="store_true",
                        help="Chỉ dùng nguồn không cần API key")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    # Nạp .env trước khi bất kỳ nguồn dữ liệu nào kiểm tra key.
    load_env()
    status = describe_keys()
    print("Key khả dụng: " + ", ".join(
        f"{k}={'có' if v else 'không'}" for k, v in status.items()
    ))
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    built = []
    for league in args.leagues:
        try:
            payload = (
                build_champions_league()
                if league == "champions-league"
                else build_league(league, no_remote=args.no_remote)
            )
        except Exception as exc:
            # Một giải hỏng không được làm sập cả lần dựng.
            print(f"  LỖI {league}: {type(exc).__name__}: {exc}")
            continue

        league_dir = out_dir / league
        league_dir.mkdir(parents=True, exist_ok=True)
        for key in ("standings", "strength", "predictions", "history"):
            dump_json(league_dir / f"{key}.json", payload.get(key, []))
        dump_json(league_dir / "model.json", payload["model"], indent=2)
        built.append({
            "id": league,
            "name": payload["name"],
            "season": payload["season"],
            "fixtures": len(payload["predictions"]),
            "teams": len(payload["standings"]),
            "history": len(payload.get("history", [])),
            "accuracy": payload.get("accuracy", {}),
            # Trận gần nhất trong tương lai, kể cả khi nằm NGOÀI cửa sổ dự
            # đoán. Cúp châu Âu nghỉ 2-3 tuần giữa hai lượt, không có trường
            # này thì app chỉ biết nói "không có trận nào" mà không nói được
            # bao giờ mới có lại.
            "next_fixture_utc": payload.get("next_fixture_utc"),
        })

    if not built:
        raise SystemExit("Không dựng được giải nào")

    write_index(out_dir, built)

    total = sum(b["fixtures"] for b in built)
    print(f"\nXong: {len(built)} giải, {total} dự đoán -> {out_dir}")

    # Chặn việc xuất bản dữ liệu hoặc giao diện đã hỏng.
    if selfcheck.run():
        raise SystemExit("Tự kiểm tra không đạt — xem lỗi ở trên")


if __name__ == "__main__":
    main()
