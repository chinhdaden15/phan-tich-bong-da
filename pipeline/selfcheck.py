"""
Kiểm tra sau mỗi lần build: mọi thứ app cần có đủ và đúng định dạng chưa.

VÌ SAO CÓ FILE NÀY: trong quá trình phát triển, hai lần liên tiếp tôi sửa
app.js bằng cách cắt ghép theo vị trí chuỗi và vô tình xoá mất cả hàm —
lần đầu mất matchCard/renderFixtures, lần sau mất marketSection. Không lần
nào bị phát hiện lúc build; cả hai lần đều do người dùng bấm vào app rồi
báo hỏng.

Cắt ghép theo vị trí là thao tác nguy hiểm vì nó không hề báo lỗi khi xoá
nhầm. Những kiểm tra dưới đây rẻ và chạy sau mỗi build, đủ để bắt đúng loại
lỗi đó ngay tại chỗ.
"""

from __future__ import annotations

import json
import re
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"

# Các hàm app không thể chạy nếu thiếu. Danh sách này là hợp đồng: xoá một
# hàm ở đây mà không cập nhật danh sách thì build sẽ báo lỗi.
REQUIRED_JS = [
    "localDate", "dayKey", "formatDay", "formatTime", "titleCase", "shortName",
    "loadJSON", "showFreshness", "nextRefreshAt", "confidenceClass", "signalClass", "signalSection",
    "bestSource", "matchCard", "renderFixtures",
    "scorersSection", "marketSection", "openSheet", "closeSheet", "enableSwipeToClose",
    "renderHistory", "zoneClass", "renderStandings", "renderStrength", "renderAbout",
    "render", "renderLeagueTabs", "init",
    "crestKey", "crestMap", "crestImg", "cornersSection", "handicapSection", "shortenTeams",
]

# Trường bắt buộc trong mỗi bản ghi dự đoán.
REQUIRED_PREDICTION_FIELDS = [
    "home_team", "away_team", "expected_goals", "prediction",
    "confidence", "fixture", "signal",
]

# Trường bắt buộc trong mỗi dòng bảng xếp hạng.
#
# Danh sách này ra đời từ một lỗi thật: bảng Champions League lấy thẳng từ API
# nên thiếu trường "form", trong khi bảng các giải quốc nội tự tính nên có.
# Giao diện gọi r.form.map() và ném lỗi, làm CẢ TAB bảng xếp hạng trắng trơn —
# không có thông báo lỗi nào ở đâu. Hai nguồn cùng đổ vào một giao diện thì
# phải có cùng bộ trường, và đây là chỗ kiểm tra điều đó.
REQUIRED_STANDINGS_FIELDS = [
    "position", "team", "played", "points", "goal_difference", "form",
]


# Trường bắt buộc trong mỗi bản ghi lịch sử.
#
# Danh sách này ra đời từ hai lỗi thật cùng nằm ở đây: tỷ số bị lưu lúc trận
# đang đá (Bayern thắng 5-0 bị ghi thành 0-0 vì đó là tỷ số hiệp 1), và cửa
# đã nêu không được chấm theo đúng loại kèo của nó (trận 2-1 với cửa "Tài 2.5"
# bị đánh dấu sai). Cả hai đều âm thầm, không báo lỗi ở đâu cả.
REQUIRED_HISTORY_FIELDS = ["home_team", "away_team", "fixture", "prediction", "actual", "grade"]
REQUIRED_ACTUAL_FIELDS = ["home_goals", "away_goals", "result"]


class CheckFailed(RuntimeError):
    pass


def check_js(errors: list[str]) -> None:
    """Mọi hàm bắt buộc phải tồn tại, và file phải parse được."""
    app = PUBLIC / "app.js"
    if not app.exists():
        errors.append("thiếu public/app.js")
        return

    source = app.read_text(encoding="utf-8")
    defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)", source))
    defined |= set(re.findall(r"(?:const|let)\s+(\w+)\s*=", source))

    for name in REQUIRED_JS:
        if name not in defined:
            errors.append(f"app.js thiếu hàm {name}()")

    # node có thì dùng để bắt lỗi cú pháp; không có thì bỏ qua, không chặn build.
    try:
        result = subprocess.run(
            ["node", "--check", str(app)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            first = (result.stderr or "").strip().splitlines()
            errors.append(f"app.js lỗi cú pháp: {first[0] if first else 'không rõ'}")
    except (FileNotFoundError, subprocess.SubprocessError):
        pass


def check_json(errors: list[str]) -> None:
    """
    Mọi file JSON phải hợp lệ theo ĐÚNG chuẩn, không chấp nhận NaN/Infinity.

    json.load mặc định của Python vẫn đọc được NaN, nên phải chặn tường minh
    bằng parse_constant — nếu không, file trông vẫn "đọc được" ở đây trong khi
    trình duyệt từ chối cả file. Tab La Liga từng trắng trơn vì đúng lỗi này.
    """
    def reject(constant):
        raise ValueError(f"gặp {constant}")

    data_dir = PUBLIC / "data"
    files = sorted(data_dir.rglob("*.json"))
    if not files:
        errors.append("public/data không có file JSON nào")
        return

    for path in files:
        try:
            json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)} không hợp lệ: {exc}")


def check_predictions(errors: list[str]) -> None:
    """Mỗi dự đoán phải đủ trường và xác suất phải cộng lại bằng 1."""
    index_file = PUBLIC / "data" / "index.json"
    if not index_file.exists():
        errors.append("thiếu public/data/index.json")
        return

    index = json.loads(index_file.read_text(encoding="utf-8"))
    for entry in index.get("leagues", []):
        league = entry["id"]
        path = PUBLIC / "data" / league / "predictions.json"
        if not path.exists():
            errors.append(f"{league}: thiếu predictions.json")
            continue

        preds = json.loads(path.read_text(encoding="utf-8"))
        for i, pred in enumerate(preds):
            for field in REQUIRED_PREDICTION_FIELDS:
                if field not in pred:
                    errors.append(f"{league}[{i}]: thiếu trường {field!r}")

            outcome = (pred.get("prediction") or {}).get("outcome")
            if outcome:
                total = sum(outcome.values())
                if abs(total - 1.0) > 0.01:
                    errors.append(
                        f"{league}[{i}]: xác suất 1X2 cộng lại {total:.4f}, phải bằng 1"
                    )


def check_standings(errors: list[str]) -> None:
    """Mọi dòng bảng xếp hạng phải đủ trường mà giao diện đọc tới."""
    index_file = PUBLIC / "data" / "index.json"
    if not index_file.exists():
        return

    index = json.loads(index_file.read_text(encoding="utf-8"))
    for entry in index.get("leagues", []):
        league = entry["id"]
        path = PUBLIC / "data" / league / "standings.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for i, row in enumerate(rows):
            for field in REQUIRED_STANDINGS_FIELDS:
                if field not in row:
                    errors.append(f"{league}: bảng xếp hạng dòng {i} thiếu {field!r}")
                    break     # một lỗi mỗi dòng là đủ để biết


def check_history(errors: list[str]) -> None:
    """
    Mỗi bản ghi lịch sử phải đủ trường, và điểm phải khớp với tỷ số đã lưu.

    Phần chấm lại tại chỗ là quan trọng nhất: nó bắt được trường hợp cửa đã
    nêu bị đánh đúng/sai theo luật của một loại kèo KHÁC. Đó chính là lỗi đã
    khiến trận 2-1 với cửa "Tài 2.5" hiện dấu sai.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from history import _pick_correct

    index_file = PUBLIC / "data" / "index.json"
    if not index_file.exists():
        return

    index = json.loads(index_file.read_text(encoding="utf-8"))
    for entry in index.get("leagues", []):
        league = entry["id"]
        path = PUBLIC / "data" / league / "history.json"
        if not path.exists():
            continue
        for i, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            missing = [f for f in REQUIRED_HISTORY_FIELDS if f not in row]
            if missing:
                errors.append(f"{league}: lịch sử [{i}] thiếu {missing[0]!r}")
                continue

            actual = row["actual"] or {}
            for field in REQUIRED_ACTUAL_FIELDS:
                if actual.get(field) is None:
                    errors.append(f"{league}: lịch sử [{i}] kết quả thiếu {field!r}")
                    break
            else:
                grade = row["grade"] or {}
                key = ((row["prediction"] or {}).get("signal") or {}).get("best", {}).get("key")
                expected = _pick_correct(key, actual["home_goals"], actual["away_goals"]) if key else None
                if expected is not None and grade.get("signal_correct") != expected:
                    errors.append(
                        f"{league}: lịch sử [{i}] chấm sai cửa {key!r} — "
                        f"{actual['home_goals']}-{actual['away_goals']} phải là {expected}"
                    )


def run(verbose: bool = True) -> list[str]:
    """Chạy toàn bộ kiểm tra. Trả về danh sách lỗi, rỗng nghĩa là đạt."""
    errors: list[str] = []
    check_js(errors)
    check_json(errors)
    check_predictions(errors)
    check_standings(errors)
    check_history(errors)

    if verbose:
        if errors:
            print(f"\nTỰ KIỂM TRA: {len(errors)} lỗi")
            for e in errors[:15]:
                print(f"  ✗ {e}")
            if len(errors) > 15:
                print(f"  ... và {len(errors) - 15} lỗi nữa")
        else:
            print("\nTự kiểm tra: đạt "
                  f"({len(REQUIRED_JS)} hàm, JSON hợp lệ, dự đoán/bảng xếp hạng đủ trường, "
                  "lịch sử chấm điểm đúng)")
    return errors


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
