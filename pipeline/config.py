"""
Nạp biến môi trường từ file .env.

Viết tay thay vì dùng python-dotenv để giữ danh sách thư viện tối thiểu —
dự án này chỉ cần numpy, scipy, pandas, requests, và càng ít phụ thuộc thì
càng ít thứ có thể hỏng khi chạy trên GitHub Actions sau này.

Biến đã có sẵn trong môi trường luôn được ưu tiên, nên trên GitHub Actions
(nơi key đến từ Secrets và không có file .env) mọi thứ vẫn chạy đúng.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """
    Đọc .env vào os.environ. Trả về các khoá đã nạp (KHÔNG kèm giá trị).

    override=False nghĩa là không ghi đè biến môi trường đã tồn tại — đây là
    hành vi đúng cho CI, nơi giá trị thật đến từ Secrets chứ không từ file.
    """
    target = path or ENV_FILE
    loaded = {}

    if not target.exists():
        return loaded

    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue

        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = "đã nạp"

    return loaded


def describe_keys() -> dict[str, bool]:
    """Cho biết key nào đã có, dùng để in trạng thái mà không lộ giá trị."""
    return {
        name: bool(os.environ.get(name, "").strip())
        for name in ("ODDS_API_KEY", "FOOTBALL_DATA_ORG_KEY")
    }
