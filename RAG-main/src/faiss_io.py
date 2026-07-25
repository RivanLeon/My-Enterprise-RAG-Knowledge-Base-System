import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Union

import faiss


def faiss_write_index(index, file_path: Union[str, Path]) -> None:
    """写入 faiss 索引。Windows 下含中文等非 ASCII 路径时，faiss C++ 无法直接写文件。"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        faiss.write_index(index, str(file_path))
        return
    fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        faiss.write_index(index, tmp_path)
        shutil.move(tmp_path, str(file_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def faiss_read_index(file_path: Union[str, Path]):
    """读取 faiss 索引，兼容 Windows 中文路径。"""
    file_path = Path(file_path)
    if sys.platform != "win32":
        return faiss.read_index(str(file_path))
    fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        shutil.copy2(str(file_path), tmp_path)
        return faiss.read_index(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
