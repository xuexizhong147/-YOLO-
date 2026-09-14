# -*- coding: utf-8 -*-
"""跑一遍测试图, 输出真实结果 (用于写 README)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from captcha_solver import Solver

ASSETS = Path(__file__).resolve().parent.parent / 'assets'


def main():
    s = Solver()
    for name in ('sample_airplane.png', 'sample_fish.png',
                 'sample_turtle.png', 'sample_butterfly.png'):
        p = ASSETS / name
        if not p.exists():
            print(f'缺 {name}')
            continue
        r = s.solve(p)
        print(f'--- {name}  (尺寸 {r.image_size[0]}x{r.image_size[1]}) ---')
        print(f'  箭头: {r.arrows}')
        for o in r.objects:
            print(f'    {o.label:<16} center={o.center}  box={o.box}  '
                  f'det={o.det_conf} dir={o.dir_conf}')
        print(f'  点击: {r.click_points}')
        print(f'  detail: {[(x["step"], x["target"], x["click"]) for x in r.sequence]}')
        print()


if __name__ == '__main__':
    main()
