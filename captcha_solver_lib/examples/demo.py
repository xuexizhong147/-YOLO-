# -*- coding: utf-8 -*-
"""captcha_solver 用法示例。

直接运行:  python examples/demo.py [图片路径]
"""
import json
import sys
from pathlib import Path

# 让 demo 可以直接运行 (把上一级目录加入搜索路径)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captcha_solver import Solver

# 默认用库内自带的测试图
DEFAULT_DIR = Path(__file__).resolve().parent.parent / 'assets'


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else None
    if img is None:
        cands = sorted(DEFAULT_DIR.glob('*.png'))
        if not cands:
            print('请传入图片路径: python examples/demo.py <图片>')
            return
        img = str(cands[0])
    print(f'图片: {img}\n')

    s = Solver()          # 加载模型, 约 2 秒

    print('=' * 62)
    print('① s.arrows(img)          —— 右上角箭头朝向')
    print('=' * 62)
    print(' ', s.arrows(img))

    print('\n' + '=' * 62)
    print('② s.objects(img)         —— 所有物体 (完整对象)')
    print('=' * 62)
    for o in s.objects(img):
        print(f'  #{o.index}  {o.label:<16} {o.species_cn:<4} '
              f'中心={o.center}  朝向={o.dir}  '
              f'检测={o.det_conf}  判向={o.dir_conf}')

    print('\n③ 换 syntax 输出不同的东西:')
    for syn in ('dir', 'center', 'species', 'label', 'box'):
        print(f'   objects(img, syntax={syn!r:<10}) -> {s.objects(img, syntax=syn)}')

    print('\n' + '=' * 62)
    print('④ s.click_points(img)    —— 按箭头顺序要点的坐标')
    print('=' * 62)
    print('  points:', s.click_points(img))
    print('  list  :', s.click_points(img, syntax='list'))
    print('  flat  :', s.click_points(img, syntax='flat'))
    print('  detail:')
    for st in s.click_points(img, syntax='detail'):
        print(f'    第{st["step"]}步  朝向 {st["target"]}  ->  点击 {st["click"]}  '
              f'({st["species_cn"]}_{st["dir"]}, 置信度 {st["dir_conf"]})')

    print('\n' + '=' * 62)
    print('⑤ s.solve_dict(img)      —— 完整结果, 可直接 json')
    print('=' * 62)
    print(json.dumps(s.solve_dict(img), ensure_ascii=False, indent=2)[:1200])


if __name__ == '__main__':
    main()
