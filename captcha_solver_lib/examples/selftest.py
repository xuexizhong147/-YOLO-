# -*- coding: utf-8 -*-
"""验证 captcha_solver 的各种输入格式与导入方式。

从任意目录运行:  python examples/selftest.py
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image
from captcha_solver import Solver, detect_arrows

# 用库内自带的测试图
IMG = Path(__file__).resolve().parent.parent / 'assets' / 'sample_airplane.png'
ok = 0
fail = 0


def check(name, cond, extra=''):
    global ok, fail
    if cond:
        ok += 1
        print(f'  ✅ {name} {extra}')
    else:
        fail += 1
        print(f'  ❌ {name} {extra}')


def main():
    print(f'图片: {IMG}')
    print(f'存在: {IMG.exists()}\n')

    s = Solver()

    print('=== 1. 五种输入格式 ===')
    r1 = s.solve(str(IMG))                       # 路径 str
    check('str 路径', len(r1.objects) > 0, f'{len(r1.objects)} 个物体')

    r2 = s.solve(IMG)                            # pathlib.Path
    check('pathlib.Path', r2.click_points == r1.click_points)

    buf = IMG.read_bytes()
    r3 = s.solve(buf)                            # bytes
    check('bytes', r3.click_points == r1.click_points)

    bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    r4 = s.solve(bgr)                            # numpy BGR (cv2 默认)
    check('numpy BGR', r4.click_points == r1.click_points)

    pil = Image.open(IMG)
    r5 = s.solve(pil)                            # PIL
    check('PIL.Image', r5.click_points == r1.click_points)

    print('\n=== 2. 各种输出语法 ===')
    check("arrows()", s.arrows(IMG) == r1.arrows, s.arrows(IMG))
    check("objects(syntax='dir')", s.objects(IMG, 'dir') == [o.dir for o in r1.objects],
          s.objects(IMG, 'dir'))
    check("objects(syntax='center')", len(s.objects(IMG, 'center')) == 4,
          s.objects(IMG, 'center'))
    check("objects(syntax='species')", len(set(s.objects(IMG, 'species'))) == 1,
          s.objects(IMG, 'species'))
    check("objects(syntax='label')", all('_' in x for x in s.objects(IMG, 'label')),
          s.objects(IMG, 'label'))
    check("objects(syntax='box')", len(s.objects(IMG, 'box')) == 4)
    check("objects(syntax='dict')", isinstance(s.objects(IMG, 'dict')[0], dict))
    check("click_points(syntax='points')", len(s.click_points(IMG)) == 3, s.click_points(IMG))
    check("click_points(syntax='list')", isinstance(s.click_points(IMG, 'list')[0], list))
    check("click_points(syntax='flat')", len(s.click_points(IMG, 'flat')) == 6)
    check("click_points(syntax='detail')",
          s.click_points(IMG, 'detail')[0]['step'] == 1)

    print('\n=== 3. 结果对象字段 ===')
    r = s.solve(IMG)
    o = r.objects[0]
    for f in ('species', 'species_cn', 'dir', 'label', 'center', 'box', 'corners',
              'det_conf', 'dir_conf', 'index'):
        check(f'Obj.{f}', hasattr(o, f), getattr(o, f, ''))
    check('Result.species', r.species is not None, r.species)
    check('Result.image_size', r.image_size == (340, 385), r.image_size)
    check('Result.warnings', isinstance(r.warnings, list), r.warnings)
    check('Result.click_points', len(r.click_points) == 3)

    print('\n=== 4. JSON 可序列化 ===')
    try:
        js = json.dumps(s.solve_dict(IMG), ensure_ascii=False)
        check('solve_dict -> json', len(js) > 200, f'{len(js)} 字符')
    except Exception as e:
        check('solve_dict -> json', False, str(e))

    print('\n=== 5. 独立函数 detect_arrows ===')
    ar = detect_arrows(Image.open(IMG))
    check('detect_arrows(PIL)', len(ar) == 3, [a['dir'] for a in ar])
    check('arrows 与 solve 一致', [a['dir'] for a in ar] == r.arrows)

    print('\n=== 6. 批量 ===')
    imgs = sorted(IMG.parent.glob('*.png'))
    cnt = 0
    for p in imgs:
        s.solve(p)
        cnt += 1
    check(f'批量 {len(imgs)} 张', cnt == len(imgs) and cnt > 0, f'{cnt} 张')

    print(f'\n{"="*50}')
    print(f'通过 {ok} 项, 失败 {fail} 项')
    print('=' * 50)
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
