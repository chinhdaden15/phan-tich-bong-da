"""
Tải logo đội và cờ các nước về máy chủ của chính app.

VÌ SAO CẦN: trước đây app trỏ thẳng vào URL của football-data.org và ESPN.
Service worker chỉ cache tài nguyên CÙNG TÊN MIỀN (sw.js bỏ qua mọi request
khác origin), nên 120 ảnh này không bao giờ được cache — mỗi lần mở app là
một lần tải lại từ máy chủ người khác, và riêng tab Nations League có tới 176
ô logo. Kết quả là logo hiện chậm, nhấp nháy, phải vào vài lần mới thấy đủ.

Tải về rồi phục vụ từ cùng origin thì service worker cache được, lần thứ hai
trở đi hiện tức thì và dùng được cả khi mất mạng.

Ảnh được thu về 64px và chuyển sang WebP: hiển thị chỉ 18-24px nên 64px là dư
cho màn hình retina, còn WebP chất lượng 85 nhỏ hơn PNG gốc khoảng năm lần
(1,6 MB xuống 322 KB cho toàn bộ 120 ảnh).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CREST_DIR = ROOT / "public" / "crests"

# Hiển thị lớn nhất là 24px; 64px đủ sắc nét cho màn hình gấp ba lần.
SIZE = 64
QUALITY = 85

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _name_for(url: str) -> str:
    """Tên tệp ổn định theo URL nguồn, nên ảnh giống nhau chỉ tải một lần."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".webp"


def localise(urls: set[str]) -> dict[str, str]:
    """
    Tải về những ảnh chưa có, trả bảng quy đổi URL gốc -> đường dẫn trong app.

    Ảnh đã nằm trên đĩa thì bỏ qua, nên lần chạy sau gần như không tốn gì và
    bot cũng chỉ tải ảnh của đội mới xuất hiện.

    Ảnh nào tải hỏng sẽ KHÔNG có trong bảng trả về — người gọi giữ nguyên URL
    gốc cho ảnh đó, để một ảnh lỗi không làm mất logo của cả giải.
    """
    CREST_DIR.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    downloaded = failed = 0

    for url in sorted(u for u in urls if u):
        name = _name_for(url)
        path = CREST_DIR / name
        if path.exists() and path.stat().st_size > 0:
            mapping[url] = f"crests/{name}"
            continue

        try:
            response = requests.get(url, timeout=25, headers=HEADERS)
            response.raise_for_status()
            path.write_bytes(_shrink(response.content))
        except Exception:
            failed += 1
            continue

        mapping[url] = f"crests/{name}"
        downloaded += 1

    if downloaded or failed:
        print(f"  logo: tải mới {downloaded}, hỏng {failed}, "
              f"tổng {len(mapping)}/{len(urls)} dùng được")
    return mapping


def write_manifest() -> int:
    """
    Ghi danh sách mọi tệp logo ra manifest.json.

    Service worker đọc tệp này lúc cài đặt để tải sẵn TOÀN BỘ logo một lần,
    thay vì đợi người dùng cuộn tới đâu mới tải tới đó. Toàn bộ 120 ảnh chỉ
    khoảng 490 KB, đổi lại là từ lần mở thứ hai trở đi logo hiện tức thì và
    vẫn thấy được khi mất mạng.
    """
    import json

    files = sorted(f.name for f in CREST_DIR.glob("*.webp")) if CREST_DIR.exists() else []
    if not files:
        return 0
    (CREST_DIR / "manifest.json").write_text(
        json.dumps(files, separators=(",", ":")), encoding="utf-8"
    )
    return len(files)


def _shrink(raw: bytes) -> bytes:
    """
    Thu nhỏ và chuyển sang WebP. Không có Pillow thì giữ nguyên ảnh gốc.

    Giữ nguyên vẫn chạy đúng, chỉ là tệp nặng hơn — không đáng để làm hỏng cả
    lần dựng chỉ vì thiếu một thư viện.
    """
    try:
        from PIL import Image
    except ImportError:
        return raw

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
        image.thumbnail((SIZE, SIZE), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "WEBP", quality=QUALITY, method=6)
        return buffer.getvalue()
    except Exception:
        return raw
