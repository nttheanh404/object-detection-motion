#!/usr/bin/env python3
"""Check the minimal runtime and report external assets without loading models."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def main() -> None:
    print(f"python={os.sys.executable}")
    print(f"cwd={Path.cwd()}")
    for name in ("numpy", "cv2", "torch", "onnxruntime"):
        spec = importlib.util.find_spec(name)
        print(f"{name}={'installed' if spec else 'missing'}")
    for path in (
        "/media/ai5070/HDD16T1/disk_360GB/lth/data_10000_2026",
        "/media/ai5070/HDD16T1/disk_360GB/lth/data_test_2025",
        "/media/ai5070/HDD16T1/disk_360GB/lth/AIhub/data_lumi/data_test_t4_2026",
    ):
        print(f"asset={path} exists={Path(path).exists()}")


if __name__ == "__main__":
    main()

