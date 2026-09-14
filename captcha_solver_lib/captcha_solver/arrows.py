# -*- coding: utf-8 -*-
"""右上角箭头读取 —— 纯 cv2 规则, 不需要模型。

验证码右上角有 3 个黑色箭头, 按顺序指示"要依次点击哪些朝向的物体"。
"""
import cv2
import numpy as np
from collections import Counter

from .config import SYMS

ARROW_SYM = SYMS          # 索引 = round(angle/45) % 8


def detect_arrows(img_rgb):
    """检测右上角箭头。

    参数
    ----
    img_rgb : numpy.ndarray   RGB 图 (H,W,3)

    返回
    ----
    list[dict]  按从左到右排序: {'dir': '→', 'bbox': [x1,y1,x2,y2], 'confidence': 0.67}

    原理: 对每个箭头连通块用**三重特征交叉验证** ——
         质心偏移方向 / 最远射线反方向 / 质量分布反方向,
         三者取多数票, 票数越多置信度越高。
    """
    a = np.asarray(img_rgb)
    h, w = a.shape[:2]
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    # 箭头行区域: 右上角, 黑箭头在白底上
    x0, y0 = int(w * 0.60), 5
    region = gray[y0:int(h * 0.16), x0:int(w * 0.99)]
    mask = (region < 215).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    # 箭头尺寸筛除 (太小是噪点, 太大是标题文字)
    comps = [i for i in range(1, num)
             if 100 <= stats[i, cv2.CC_STAT_AREA] <= 400
             and 8 <= stats[i, cv2.CC_STAT_WIDTH] <= 34
             and 8 <= stats[i, cv2.CC_STAT_HEIGHT] <= 34]
    comps.sort(key=lambda i: stats[i, cv2.CC_STAT_LEFT])     # 从左到右

    arrows = []
    for ci in comps:
        x, y, bw, bh, _ = stats[ci]
        sub = (labels == ci).astype(np.uint8)
        ys, xs = np.nonzero(sub)
        comx, comy = xs.mean(), ys.mean()
        bcx = (xs.min() + xs.max()) / 2.0
        bcy = (ys.min() + ys.max()) / 2.0
        dx, dy = comx - bcx, comy - bcy

        # 从质心出发的 8 方向累计射线
        rays = np.zeros(8)
        mass = np.zeros(8)
        for px, py in zip(xs, ys):
            vx, vy = px - comx, py - comy
            d = float(np.hypot(vx, vy))
            if d == 0:
                continue
            ang = (np.degrees(np.arctan2(vy, vx)) + 360) % 360
            bi = int(round(ang / 45)) % 8
            rays[bi] += d
            mass[bi] += 1
        dir_com = int(round((np.degrees(np.arctan2(dy, dx)) + 360) % 360 / 45)) % 8
        dir_far = (int(np.argmax(rays)) + 4) % 8     # 最远射线 -> 反方向才是箭头指向
        dir_mass = (int(np.argmax(mass)) + 4) % 8

        cnt = Counter([dir_com, dir_far, dir_mass])
        top_dir, top_n = cnt.most_common(1)[0]
        arrows.append({
            'dir': ARROW_SYM[top_dir],
            'bbox': [int(x + x0), int(y + y0), int(x + bw + x0), int(y + bh + y0)],
            'confidence': round(top_n / 3.0, 2),
        })
    return arrows
