# captcha-solver

> 朝向类验证码识别库 —— 输入一张验证码截图，输出**箭头朝向序列**、**所有物体的物种/朝向/中心坐标**，以及**按箭头顺序要点击的坐标**。

纯本地推理，不联网；模型已打包在库内，开箱即用。

---

## ⚡ TL;DR（30 秒上手）

**需要装的库只有 4 个**（`pip install` 会自动装，含传递依赖 torch）：

| 库 | 用途 |
|---|---|
| `ultralytics` | YOLO 推理框架（会自动带上 `torch`） |
| `opencv-python` | 图像处理 + 箭头检测 |
| `numpy` | 数组运算 |
| `pillow` | 图片读取 |

```bash
# ① 装依赖
pip install ultralytics opencv-python numpy pillow

# ② 装本库
git clone https://github.com/<你的用户名>/captcha-solver.git
cd captcha-solver
pip install -e .

# ③ 用内置测试图验证（无需自己准备图片）
python -c "from captcha_solver import Solver, sample_image; print(Solver().click_points(sample_image()))"
```

跑出 `[(268, 150), (155, 107), (217, 192)]` 就说明装好了。

---

## 📷 效果演示

输入内置测试图 `sample_airplane.png`：

![sample](assets/sample_airplane.png)

输出：

```
箭头序列:  ← ↘ →
物体:
  airplane_←   center=(268, 150)   box=(245,133,290,166)   det=0.945  dir=1.000
  airplane_↑   center=( 56, 197)   box=( 38,174, 74,219)   det=0.937  dir=1.000
  airplane_→   center=(217, 192)   box=(192,167,241,217)   det=0.926  dir=0.996
  airplane_↘   center=(155, 107)   box=(121, 73,189,141)   det=0.918  dir=0.998
要点击的坐标:  (268,150) → (155,107) → (217,192)
```

即：按右上角箭头 `← ↘ →` 的顺序，依次点击这三个物体。

---

## 🖼️ 测试图片在哪？（装完就能直接用）

本库内置了 **4 张测试图**，**不需要你另外准备图片**：

| 位置 | 说明 |
|---|---|
| `captcha_solver/samples/` | **包内**，`pip install` 后也有，通过 API 访问 |
| `assets/` | 仓库根目录，方便在 GitHub 网页上直接看 |

包内 4 张图分别覆盖 4 个物种：

```
sample_airplane.png    飞机
sample_butterfly.png   蝴蝶
sample_fish.png        鱼
sample_turtle.png      乌龟
```

### 用代码拿到测试图

```python
from captcha_solver import sample_image, list_samples, samples_dir

list_samples()
# ['sample_airplane.png', 'sample_butterfly.png', 'sample_fish.png', 'sample_turtle.png']

samples_dir()
# PosixPath('/.../site-packages/captcha_solver/samples')

sample_image()                    # 默认第一张（飞机）
sample_image('sample_fish.png')   # 指定文件名
sample_image('fish')              # 也可以只给关键字，自动模糊匹配
```

### 直接跑测试

```bash
# 完整自检（34 项，覆盖所有输入格式与输出语法）
python examples/selftest.py

# 用法演示（用内置测试图）
python examples/demo.py
python examples/demo.py assets/sample_fish.png     # 也可以指定自己的图

# 坐标换算 + 模拟点击演示
python examples/click_demo.py
```

### 最简单的验证

```python
from captcha_solver import Solver, sample_image

s = Solver()
print(s.arrows(sample_image()))          # ['←', '↘', '→']
print(s.click_points(sample_image()))    # [(268, 150), (155, 107), (217, 192)]
```

---

## 一、环境要求

| 项 | 要求 |
|---|---|
| **Python** | **3.8 ~ 3.13**（推荐 3.10~3.12） |
| 操作系统 | Windows / Linux / macOS 均可 |
| 显卡 | **可选**。有 NVIDIA GPU 自动加速（单张 ~100ms）；无 GPU 用 CPU 也能跑（单张 ~1s） |
| 磁盘 | 约 10 MB（含模型权重和测试图） |

### 依赖清单（必须）

| 库 | 版本要求 | 用途 |
|---|---|---|
| **ultralytics** | >= 8.3.0 | YOLO 检测/分类推理（会**自动装上 torch**，无需手动装） |
| **opencv-python** | >= 4.8.0 | 图像处理、箭头连通块检测、图片编解码 |
| **numpy** | >= 1.24.0 | 数组运算 |
| **pillow** | >= 10.0.0 | 图片读取、格式转换 |

一行装完：

```bash
pip install ultralytics opencv-python numpy pillow
```

### 可选依赖

| 库 | 用途 |
|---|---|
| `pyautogui` | 想在自己电脑上**模拟鼠标点击**时才需要（见第六章） |
| `selenium` / `playwright` | 做网页自动化时才需要 |

```bash
pip install pyautogui                  # 模拟点击
pip install -e ".[click]"              # 或者用本库的可选依赖组
```

### ✅ 实测过的环境组合

这套代码在**新旧两套依赖上都验证通过**（34 项自检全过）：

| 环境 | Python | ultralytics | opencv | numpy | pillow | torch |
|---|---|---|---|---|---|---|
| 旧 | 3.8.15 | 8.4.24 | 4.13.0 | 1.24.4 | 10.4.0 | 2.4.1+cu124（GPU） |
| 新 | 3.13.1 | 8.4.147 | 5.0.0 | 2.5.3 | 12.3.0 | 2.14.0+cpu（CPU） |

所以你**不用刻意对齐版本**，按正常方式装最新版就行。

### 建议：用虚拟环境隔离

```bash
# ① 创建虚拟环境（Python 3.10 为例）
python -m venv venv

# ② 激活
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

# ③ 升级 pip 并安装依赖
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy pillow
```

---

## 二、安装

### 方式 A：直接安装（推荐）

```bash
git clone https://github.com/<你的用户名>/captcha-solver.git
cd captcha-solver
pip install -e .
```

装完后**在任意目录**都能 `import captcha_solver`。

### 方式 B：只装依赖，不安装包

```bash
git clone https://github.com/<你的用户名>/captcha-solver.git
cd captcha-solver
pip install -r requirements.txt
```

这种方式必须在**项目根目录**下运行代码（或用方式 C）。

### 方式 C：代码里手动加路径

```python
import sys
sys.path.insert(0, '/path/to/captcha-solver')   # 换成你 clone 下来的目录

from captcha_solver import Solver
```

### 如果要 GPU 加速（可选）

`ultralytics` 默认会装 CPU 版 PyTorch。想用 GPU：

```bash
# 先卸掉 CPU 版
pip uninstall -y torch torchvision

# 再按你的 CUDA 版本装 GPU 版（示例为 CUDA 12.4）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

验证是否用上 GPU：

```python
import torch
print(torch.cuda.is_available())    # True 就是用上了
```

---

## 三、验证安装

```bash
python examples/selftest.py
```

预期输出（34 项全通过）：

```
通过 34 项, 失败 0 项
```

也可以跑演示：

```bash
python examples/demo.py assets/sample_airplane.png
```

---

## 四、快速开始

```python
from captcha_solver import Solver

s = Solver()                                   # 加载模型，约 2 秒（只需一次）

# 三种最常用的调用
s.arrows('assets/sample_airplane.png')         # ['←', '↘', '→']
s.objects('assets/sample_airplane.png')        # [Obj, Obj, Obj, Obj]
s.click_points('assets/sample_airplane.png')   # [(268, 150), (155, 107), (217, 192)]
```

---

## 五、API 与语法 ⭐

库的设计是**「换语法 → 换输出」**：同一个数据源，用不同参数拿不同形式的结果。

### 5.1 `s.arrows(img)` —— 只要箭头朝向

```python
s.arrows('a.png')
# ['←', '↘', '→']
```

返回右上角箭头的朝向列表，**按从左到右排序**。这就是"要依次点击的朝向"。

### 5.2 `s.objects(img, syntax=...)` —— 所有物体

**核心参数 `syntax` 决定返回什么**：

| `syntax` | 返回类型 | 示例输出 |
|---|---|---|
| `'full'`（默认） | `list[Obj]` | `[Obj(species='fish', dir='↖', ...), ...]` |
| **`'dir'`** | `list[str]` | `['↖', '←', '↑', '↘']` |
| **`'center'`** | `list[tuple]` | `[(72, 93), (147, 119), (173, 185), (268, 205)]` |
| `'species'` | `list[str]` | `['fish', 'fish', 'fish', 'fish']` |
| `'label'` | `list[str]` | `['fish_↖', 'fish_←', 'fish_↑', 'fish_↘']` |
| `'box'` | `list[tuple]` | `[(36,56,108,129), (120,97,174,140), ...]` |
| `'dict'` | `list[dict]` | `[{'species': 'fish', 'dir': '↖', ...}, ...]` |

```python
s.objects('a.png', syntax='center')   # [(72,93), (147,119), (173,185), (268,205)]
s.objects('a.png', syntax='dir')      # ['↖', '←', '↑', '↘']
s.objects('a.png', syntax='label')    # ['fish_↖', 'fish_←', 'fish_↑', 'fish_↘']
```

**`Obj` 对象的字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `o.species` | str | `'fish'` |
| `o.species_cn` | str | `'鱼'`（中文名） |
| `o.dir` | str | `'↖'` |
| `o.label` | str | `'fish_↖'`（物种_朝向） |
| `o.center` | (int, int) | **物体中心坐标** (x, y) |
| `o.box` | (int,int,int,int) | 外接矩形 (x1, y1, x2, y2) |
| `o.corners` | list | 旋转框四角 `[[x,y]×4]`（环形顺序） |
| `o.det_conf` | float | 检测置信度 |
| `o.dir_conf` | float | 判向置信度 |
| `o.index` | int | 在图中出现的顺序 |
| `o.to_dict()` | dict | 转成纯字典 |

### 5.3 `s.click_points(img, syntax=...)` —— 要点击的坐标 ⭐

**按箭头顺序匹配物体**，直接给出应该点哪里。

| `syntax` | 返回类型 | 示例 |
|---|---|---|
| `'points'`（默认） | `list[tuple]` | `[(147,119), (72,93), (268,205)]` |
| `'list'` | `list[list]` | `[[147,119], [72,93], [268,205]]` |
| `'flat'` | `list[int]` | `[147,119,72,93,268,205]` |
| `'detail'` | `list[dict]` | 见下 |

```python
s.click_points('a.png', syntax='detail')
# [
#   {'step': 1, 'target': '←', 'click': [147, 119],
#    'species': 'fish', 'species_cn': '鱼', 'dir': '←', 'dir_conf': 0.99, 'index': 1},
#   {'step': 2, 'target': '↖', 'click': [72, 93],  ...},
#   {'step': 3, 'target': '↘', 'click': [268, 205], ...},
# ]
```

### 5.4 `s.solve(img)` / `s.solve_dict(img)` —— 完整结果

```python
r = s.solve('a.png')

r.arrows          # ['←', '↖', '↘']
r.objects         # [Obj × 4]
r.sequence        # 按箭头顺序的匹配详情
r.click_points    # [(147,119), (72,93), (268,205)]
r.species         # 'fish'    ← 全图物种（同图同物种）
r.image_size      # (340, 385)
r.warnings        # []  或 ['检测到 3 个物体 (预期 4 个)']
r.to_dict()       # 全部转 dict

s.solve_dict('a.png')     # 直接返回 dict，可 json.dumps
```

### 5.5 底层函数 `detect_arrows`

只要箭头检测、不要物体：

```python
from captcha_solver import detect_arrows
from PIL import Image

detect_arrows(Image.open('a.png'))
# [{'dir': '←', 'bbox': [204,25,223,42], 'confidence': 1.0},
#  {'dir': '↘', 'bbox': [229,25,246,42], 'confidence': 1.0},
#  {'dir': '→', 'bbox': [252,25,269,42], 'confidence': 1.0}]
```

### 5.6 内置测试图 —— `sample_image()`

`pip install` 后**自带 4 张测试图**，不用自己准备：

```python
from captcha_solver import sample_image, list_samples, samples_dir

list_samples()
# ['sample_airplane.png', 'sample_butterfly.png', 'sample_fish.png', 'sample_turtle.png']

samples_dir()                     # 测试图所在目录 (pathlib.Path)
sample_image()                    # 默认第一张（飞机）
sample_image('sample_fish.png')   # 指定文件名
sample_image('fish')              # 只给关键字也行
```

配合 `Solver` 一行验证安装：

```python
from captcha_solver import Solver, sample_image

s = Solver()
s.solve_dict(sample_image())      # 直接用，不需要任何本地图片
```

### 5.7 构造参数（一般不用改）

```python
Solver(
    detector=None,          # 自定义检测模型路径
    classifier=None,        # 自定义判向模型路径
    conf=0.05,              # 检测阈值（**默认 0.05，别调高**，见「常见问题」）
    topk=4,                 # 每图最多保留几个物体（领域规则：恰好 4 个）
    tta180=True,            # 180° TTA，专治「头尾判反」
    arrow_constraint=False, # 箭头约束解码（收益有限）
    device=None,            # 传 'cpu' 可强制 CPU
)
```

---

## 六、坐标系统与模拟点击 ⭐⭐

这是接入实际使用时**最容易搞错**的地方，务必看完。

### 6.1 库返回的坐标以「截图左上角」为原点

```
        x →
   ┌─────────────────────────┐
 y │ (0,0)                   │
 ↓ │   ┌─────────────────┐   │
   │   │                 │   │
   │   │   验证码图片      │   │
   │   │                 │   │
   │   │        ● (268,150)  │   ← 库返回的就是这个坐标
   │   │                 │   │
   │   └─────────────────┘   │
   │      (340, 385)         │
   └─────────────────────────┘
```

- 原点 `(0, 0)` = **截图图片本身的左上角**
- `x` 向右递增，`y` 向下递增
- 单位：**像素**
- 坐标范围：`0 ≤ x < 图片宽`，`0 ≤ y < 图片高`

你可以用 `r.image_size` 拿到图片尺寸。

### 6.2 要模拟点击，必须加上「图片在屏幕上的位置」

库不知道你的验证码显示在屏幕的哪个位置，所以需要你自己做一次**坐标换算**：

```
屏幕坐标 = 截图坐标 + 截图左上角在屏幕上的位置

screen_x = img_left + center_x
screen_y = img_top  + center_y
```

**举例**：验证码图片在屏幕上的位置是 `left=820, top=360`，库返回 `(268, 150)`：

```
screen_x = 820 + 268 = 1088
screen_y = 360 + 150 = 510
→ 实际要点击屏幕上的 (1088, 510)
```

### 6.3 场景一：用 pyautogui 在自己电脑上点击

适用于「手动打开网页 → 截图 → 自动点击」。

```python
import time
import pyautogui
from captcha_solver import Solver

IMG_LEFT, IMG_TOP = 820, 360        # 验证码图片左上角在屏幕上的位置
IMG_PATH = 'captcha.png'

s = Solver()
points = s.click_points(IMG_PATH, syntax='list')     # [[147,119], [72,93], [268,205]]

for i, (x, y) in enumerate(points, 1):
    screen_x = IMG_LEFT + x                          # ★ 关键：加上偏移
    screen_y = IMG_TOP  + y
    pyautogui.click(screen_x, screen_y)
    print(f'第{i}步: 点击截图内 ({x},{y}) -> 屏幕 ({screen_x},{screen_y})')
    time.sleep(0.3)                                  # 每步之间留点时间
```

> 怎么拿到 `IMG_LEFT / IMG_TOP`？
> - 用 `pyautogui.locateOnScreen('captcha.png')` 自动定位
> - 或从浏览器开发者工具读取元素的 `getBoundingClientRect()`
> - 或直接从你的截图逻辑里记录（截图时你选定了区域，那个区域的原点就是它）

### 6.4 场景二：Selenium（网页元素坐标）

```python
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from captcha_solver import Solver

driver = webdriver.Chrome()
# ... 打开页面，定位验证码元素 ...
el = driver.find_element('css selector', '.captcha-img')

# ① 只截这个元素（最简单，坐标换算最干净）
el.screenshot('captcha.png')

# ② 识别
s = Solver()
points = s.click_points('captcha.png', syntax='list')

# ③ 点击
actions = ActionChains(driver)
for x, y in points:
    # move_to_element_with_offset 的原点就是元素左上角，正好对应我们的坐标系
    actions.move_to_element_with_offset(el, x, y).click()
actions.perform()
```

> **为什么要先截图再识别，而不是直接用元素坐标？**
> 因为验证码图片的"素材"可能被 CSS 缩放。
> 截图后识别，得到的是**截图内的像素坐标**，换算逻辑最清晰。

### 6.5 场景三：Playwright

```python
from playwright.sync_api import sync_playwright
from captcha_solver import Solver

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('https://example.com')
    el = page.query_selector('.captcha-img')

    box = el.bounding_box()                 # {'x':.., 'y':.., 'width':.., 'height':..}
    el.screenshot(path='captcha.png')

    s = Solver()
    for x, y in s.click_points('captcha.png', syntax='list'):
        page.mouse.click(box['x'] + x, box['y'] + y)   # ★ 加上元素在页面中的偏移
```

### 6.6 坐标换算速查

| 你想做的事情 | 用哪些坐标 |
|---|---|
| 在**截图图片**上画标记 | 直接用库返回的坐标 |
| 在**屏幕**上点击 | 库坐标 **+ 截图左上角在屏幕的位置** |
| 在 **Selenium 元素**内点击 | 用 `move_to_element_with_offset`，直接用库坐标 |
| 在 **Playwright 页面**内点击 | 库坐标 **+ 元素 bounding_box 的 x/y** |
| 在 **OpenCV 窗口**里高亮 | 直接用库返回的坐标 |

---

## 七、输入格式

`img` 参数支持 **5 种**：

```python
s.solve('captcha.png')                  # ① 文件路径（中文路径也没问题）
s.solve(Path('captcha.png'))            # ② pathlib.Path
s.solve(response.content)               # ③ bytes（网络请求直接喂）
s.solve(cv2.imread('captcha.png'))      # ④ numpy 数组（cv2 默认 BGR）
s.solve(Image.open('captcha.png'))      # ⑤ PIL.Image
```

最常用的是 ③ —— 爬虫里直接把下载到的字节丢进去，不用落盘：

```python
import requests
from captcha_solver import Solver

s = Solver()
img_bytes = requests.get(captcha_url, cookies=cookies).content
result = s.solve_dict(img_bytes)        # 不需要存成文件
print(result['click_points'])
```

### 批量处理

```python
from pathlib import Path
from captcha_solver import Solver

s = Solver()
for p in sorted(Path('shots').glob('*.png')):
    r = s.solve(p)
    print(p.name, r.arrows, r.click_points)
    if r.warnings:
        print('  ⚠️', r.warnings)
```

---

## 八、工作原理

```
                    输入：验证码截图
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
┌───────────┐    ┌──────────────┐    ┌──────────────┐
│ ① 找物体   │    │ ② 判朝向      │    │ ③ 读箭头      │
│ YOLO11n-  │    │ YOLO11n-cls  │    │ cv2 规则      │
│ OBB 5 类   │    │ 8 类          │    │（不需要模型）  │
└─────┬─────┘    └──────┬───────┘    └──────┬───────┘
      │ 4 个旋转框 + 物种 │ 每个物体的朝向     │ 3 个目标朝向
      └─────────────────┴───────────────────┘
                        ▼
              按箭头顺序匹配 → 点击坐标
```

**用了两个模型 ＋ 一段规则代码**：

| 组件 | 文件 | 作用 |
|---|---|---|
| ① 物种检测器 | `models/detector_5species_obb.pt` | 找 4 个物体 + 物种 + 旋转框 |
| ② 朝向分类器 | `models/orientation_8class_cls.pt` | 对每个物体裁剪图判 8 个朝向 |
| ③ 箭头读取 | `arrows.py` | 纯 cv2，读右上角 3 个箭头 |

**为什么拆成两个模型**：40 类（5 物种 × 8 朝向）一次性学，每类样本太少，检测召回会塌。
拆开后物种只分 5 类、朝向只分 8 类，且朝向分类器能在**裁剪图**上做 8 倍旋转增强。

---

## 九、性能与限制

### 实测性能

在 **120 个物体的独立测试集**（完全不参与训练）上：

| 指标 | 准确率 |
|---|---|
| 定位召回 | **100%** |
| 物种判断 | **100%** |
| 朝向判断 | **93.3%** |
| **全任务（全部正确）** | **93.3%** |

- 剩余错误**全部是朝向**，其中一半是 **180° 头尾判反**（这类任务的固有难点：矩形转 180° 与自身重合，模型只能靠"头长什么样"来判断）
- 95% 置信区间约 **87% ~ 97%**
- 速度：单张约 **100~200 ms**（GPU）/ **1~2 s**（CPU）

### ⚠️ 已知限制

1. **置信度不能当质量门**
   实测存在"置信度 1.000 但判错"的情况。如需人工复核，低置信度只适合做**优先级排序**，不能只看低置信度。

2. **强依赖领域规则**
   本库针对的是这种特定验证码：**每图恰好 4 个物体、4 个同物种、右上角 3 个箭头**。
   换成不同布局的验证码需要重新训练。

3. **只支持 5 个物种**
   `airplane / butterfly / car / fish / turtle`。
   加新物种需要补标注并重训检测器（**朝向分类器可复用**，因为它与物种无关）。

---

## 十、常见问题

**Q: 检测不到物体，或物体数量不对？**
检查 `r.warnings`。如果提示"检测到 3 个物体 (预期 4 个)"，通常是该图本身有渲染问题（空白图、半张图），或物体被截断。

**Q: 为什么 `conf` 默认是 0.05 而不是 0.25？**
这个模型的置信度标定偏低。用 0.25 会漏掉真物体（实测召回从 1.00 掉到 0.83）。
默认的 `conf=0.05` 粗筛 + `topk=4` 领域规则（每图恰好 4 个物体）反而更稳。

**Q: 箭头读出来不是 3 个？**
检查 `r.warnings`。极少数图片会有渲染异常（多出几个箭头），此时箭头序列不可信。

**Q: 能识别其他形状的验证码吗？**
不能直接用。需要按同样的流程重新训练（物种检测器要重做，朝向分类器在朝向体系相同时可复用）。

**Q: 中文路径报错？**
不会。库内部用 `np.fromfile + cv2.imdecode` 读图，绕开了 OpenCV 不支持中文路径的问题。

**Q: 一定要 GPU 吗？**
不需要。CPU 也能跑，只是慢一些（约 1~2 秒/张）。生产环境建议用 GPU。

---

## 十一、目录结构

```
captcha-solver/
├── README.md
├── requirements.txt                    # 依赖清单
├── pyproject.toml                      # pip install 配置
├── .gitignore
├── assets/                             # 测试图片（方便在 GitHub 网页上看）
│   ├── sample_airplane.png
│   ├── sample_butterfly.png
│   ├── sample_fish.png
│   └── sample_turtle.png
├── captcha_solver/                     # 包本体（pip install 装的就是这个）
│   ├── __init__.py
│   ├── solver.py                       # 主类 Solver
│   ├── arrows.py                       # 箭头检测（cv2）
│   ├── samples.py                      # sample_image() 等
│   ├── config.py                       # 常量
│   ├── models/                         # ★ 模型权重（随包安装）
│   │   ├── detector_5species_obb.pt    #   5 类物种检测（5.6 MB）
│   │   └── orientation_8class_cls.pt   #   8 类朝向分类（3.1 MB）
│   └── samples/                        # ★ 测试图片（随包安装）
│       ├── sample_airplane.png
│       ├── sample_butterfly.png
│       ├── sample_fish.png
│       └── sample_turtle.png
└── examples/
    ├── demo.py                         # 完整用法演示
    ├── selftest.py                     # 自检（34 项）
    ├── click_demo.py                   # 坐标换算 + 模拟点击示例
    └── dump_samples.py                 # 打印各测试图的识别结果
```

> **为什么测试图放两处？**
> `captcha_solver/samples/` 在包内，`pip install` 后依然有，供代码调用；
> `assets/` 在仓库根目录，方便在 GitHub 网页上直接浏览，也供 README 引用。
> 两者内容相同。

---

## 十二、最小可用示例（复制即用）

```python
from captcha_solver import Solver

s = Solver()
r = s.solve('assets/sample_airplane.png')

print('箭头序列:', r.arrows)
print('要点的坐标:', r.click_points)

# 换算到屏幕坐标（假设验证码图片在屏幕上的位置是 left=820, top=360）
LEFT, TOP = 820, 360
for i, (x, y) in enumerate(r.click_points, 1):
    print(f'第{i}步 -> 屏幕坐标 ({LEFT + x}, {TOP + y})')
```

输出：

```
箭头序列: ['←', '↘', '→']
要点的坐标: [(268, 150), (155, 107), (217, 192)]
第1步 -> 屏幕坐标 (1088, 510)
第2步 -> 屏幕坐标 (975, 467)
第3步 -> 屏幕坐标 (1037, 552)
```

---

## 许可

请根据你的需要添加 LICENSE 文件（如 MIT）。
使用前请确认符合目标网站的服务条款与当地法律法规。
