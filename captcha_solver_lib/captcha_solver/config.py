# -*- coding: utf-8 -*-
"""常量与配置。"""

# 右上角箭头 / 物体朝向 的 8 个方向（索引 = round(角度/45) % 8）
SYMS = ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗']
ANGLE = {s: i * 45 for i, s in enumerate(SYMS)}

# 5 个物种（顺序与检测模型输出的类别索引一致，不要改）
SPECIES = ['airplane', 'butterfly', 'car', 'fish', 'turtle']
SPECIES_CN = {'airplane': '飞机', 'butterfly': '蝴蝶', 'car': '车',
              'fish': '鱼', 'turtle': '乌龟'}

# 分类模型文件夹名 -> 方向
CODE2SYM = {'E': '→', 'SE': '↘', 'S': '↓', 'SW': '↙',
            'W': '←', 'NW': '↖', 'N': '↑', 'NE': '↗'}

# 领域规则：一张验证码里恰好 4 个物体，4 个物体同物种
N_OBJECTS = 4
