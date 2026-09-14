# -*- coding: utf-8 -*-
"""内置测试图 —— 方便安装后立刻验证。

用法
----
    from captcha_solver import sample_image, list_samples

    list_samples()                    # ['sample_airplane.png', ...]
    sample_image()                    # 路径: 飞机那张
    sample_image('sample_fish.png')   # 指定某张
"""
from pathlib import Path
from typing import List, Optional

SAMPLES_DIR = Path(__file__).resolve().parent / 'samples'


def list_samples() -> List[str]:
    """列出所有内置测试图的文件名。"""
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLES_DIR.glob('*.png'))


def sample_image(name: Optional[str] = None) -> Path:
    """返回内置测试图的路径。

    参数
    ----
    name : str, 可选
        文件名（如 'sample_fish.png'），或只给物种关键字（如 'fish'）。
        不传则返回第一张。

    返回
    ----
    pathlib.Path

    异常
    ----
    FileNotFoundError  找不到对应的测试图
    """
    if not SAMPLES_DIR.exists():
        raise FileNotFoundError(f'内置测试图目录不存在: {SAMPLES_DIR}')
    if name is None:
        files = sorted(SAMPLES_DIR.glob('*.png'))
        if not files:
            raise FileNotFoundError(f'{SAMPLES_DIR} 下没有 png')
        return files[0]
    p = SAMPLES_DIR / name
    if p.exists():
        return p
    # 只给了关键字，模糊匹配
    hits = sorted(SAMPLES_DIR.glob(f'*{name}*.png'))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f'找不到测试图 {name!r}，可用: {list_samples()}')


def samples_dir() -> Path:
    """返回内置测试图所在目录。"""
    return SAMPLES_DIR
