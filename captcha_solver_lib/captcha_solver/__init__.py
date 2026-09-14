# -*- coding: utf-8 -*-
"""captcha_solver —— 朝向类验证码识别库。

    from captcha_solver import Solver, sample_image

    s = Solver()
    img = sample_image()            # 内置测试图，装了库就有

    s.arrows(img)                   # ['←', '↘', '→']          右上角箭头朝向
    s.objects(img)                  # [Obj, ...]              所有物体的物种/朝向/中心
    s.click_points(img)             # [(x,y), (x,y), (x,y)]   按箭头顺序要点的坐标
    s.solve(img)                    # 完整结果

详细说明见 README.md。
"""
from .solver import Solver, CaptchaResult, Obj, SolverError
from .arrows import detect_arrows
from .samples import sample_image, list_samples, samples_dir
from .config import SYMS, ANGLE, SPECIES, SPECIES_CN, CODE2SYM, N_OBJECTS

__version__ = '1.0.0'
__all__ = [
    'Solver', 'CaptchaResult', 'Obj', 'SolverError', 'detect_arrows',
    'sample_image', 'list_samples', 'samples_dir',
    'SYMS', 'ANGLE', 'SPECIES', 'SPECIES_CN', 'CODE2SYM', 'N_OBJECTS',
]
