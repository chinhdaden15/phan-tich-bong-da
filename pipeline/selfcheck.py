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
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"

# Các hàm app không thể chạy nếu thiếu. Danh sách này là hợp đồng: xoá một
# hàm ở đây mà không cập nhật danh sách thì build sẽ báo lỗi.
REQUIRED_JS = [
    "localDate", "dayKey", "formatDay", "formatTime", "titleCase", "shortName",
    "loadJSON", "confidenceClass", "signalClass", "signalSection",
    "bestSource", "matchCard", "renderFixtures",
    "marketSection", "openSheet", "closeSheet",
    "renderHistory", "zoneClass", "renderStandings", "renderStrength", "renderAbout",
    "render", "renderLeagueTabs", "init",
]

# Trường bắt buộc trong mỗi bản ghi dự đoán.
REQUIRED_PREDICTION_FIELDS = [
    "home_team", "away_team", "expected_goals", "prediction",
    "confidence", "fixture", "signal",
]


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


def run(verbose: bool = True) -> list[str]:
    """Chạy toàn bộ kiểm tra. Trả về danh sách lỗi, rỗng nghĩa là đạt."""
    errors: list[str] = []
    check_js(errors)
    check_json(errors)
    check_predictions(errors)

    if verbose:
        if errors:
            print(f"\nTỰ KIỂM TRA: {len(errors)} lỗi")
            for e in errors[:15]:
                print(f"  ✗ {e}")
            if len(errors) > 15:
                print(f"  ... và {len(errors) - 15} lỗi nữa")
        else:
            print("\nTự kiểm tra: đạt "
                  f"({len(REQUIRED_JS)} hàm, JSON hợp lệ, dự đoán đủ trường)")
    return errors


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
