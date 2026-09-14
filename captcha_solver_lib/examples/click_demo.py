# -*- coding: utf-8 -*-
"""坐标换算 + 模拟点击 示例。

重点演示 README 第六章说的那件事:
    库返回的坐标是**以截图左上角为 (0,0)**
    要在屏幕上点击, 必须**加上截图左上角在屏幕上的位置**

运行:  python examples/click_demo.py
（只做换算演示, 不会真的点击鼠标 —— 要真点请把 DRY_RUN 改成 False）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captcha_solver import Solver

# ---------------------------------------------------------------- 配置
ASSETS = Path(__file__).resolve().parent.parent / 'assets'
IMG_PATH = ASSETS / 'sample_airplane.png'

# ★ 验证码图片左上角在「屏幕」上的位置（按你的实际情况填）
IMG_LEFT, IMG_TOP = 820, 360

DRY_RUN = True          # True = 只打印坐标, 不真的点鼠标


def main():
    if not IMG_PATH.exists():
        print(f'找不到测试图: {IMG_PATH}')
        return

    s = Solver()
    r = s.solve(IMG_PATH)

    print('=' * 66)
    print('① 识别结果（坐标以截图左上角为原点）')
    print('=' * 66)
    print(f'  图片尺寸      : {r.image_size[0]} x {r.image_size[1]}')
    print(f'  箭头序列      : {r.arrows}')
    print(f'  全图物种      : {r.species_cn} ({r.species})')
    print('  所有物体:')
    for o in r.objects:
        print(f'    {o.label:<16} 中心={o.center}  框={o.box}')

    print()
    print('=' * 66)
    print('② 按箭头顺序要点击的坐标（仍是截图坐标系）')
    print('=' * 66)
    print(f'  r.click_points          -> {r.click_points}')
    print(f'  s.click_points(img)     -> {s.click_points(IMG_PATH)}')
    print(f'  s.click_points(syntax="list") -> {s.click_points(IMG_PATH, syntax="list")}')
    print()
    for st in r.sequence:                       # r.sequence 是匹配详情
        print(f'  第{st["step"]}步  朝向 {st["target"]}  ->  截图内 {tuple(st["click"])}'
              f'   ({st["species_cn"]}_{st["dir"]}, 判向置信度 {st["dir_conf"]})')

    print()
    print('=' * 66)
    print(f'③ 换算到屏幕坐标  (图片左上角在屏幕 {IMG_LEFT},{IMG_TOP})')
    print('=' * 66)
    print('  公式:  屏幕坐标 = 截图坐标 + 图片左上角在屏幕上的位置')
    print()
    converted = []
    for i, (x, y) in enumerate(r.click_points, 1):
        sx, sy = IMG_LEFT + x, IMG_TOP + y
        converted.append((sx, sy))
        print(f'  第{i}步   ({x:>3}, {y:>3})  +  ({IMG_LEFT}, {IMG_TOP})'
              f'  =  屏幕 ({sx}, {sy})')

    print()
    print('=' * 66)
    print('④ 模拟点击')
    print('=' * 66)
    if DRY_RUN:
        print('  当前是 DRY_RUN 模式, 只打印不点击。')
        print('  改成 DRY_RUN = False 就会真的调用 pyautogui 点击。')
        for i, (sx, sy) in enumerate(converted, 1):
            print(f'  [模拟] pyautogui.click({sx}, {sy})')
        print()
        print('  真实点击的代码就是:')
        print('''
      import pyautogui
      for sx, sy in converted:
          pyautogui.click(sx, sy)
          time.sleep(0.3)
''')
    else:
        try:
            import pyautogui
        except ImportError:
            print('  需要先安装: pip install pyautogui')
            return
        print('  3 秒后开始点击, 请把鼠标移开...')
        time.sleep(3)
        for i, (sx, sy) in enumerate(converted, 1):
            pyautogui.click(sx, sy)
            print(f'  第{i}步 -> 点击屏幕 ({sx}, {sy})')
            time.sleep(0.3)
        print('  完成')


if __name__ == '__main__':
    main()
