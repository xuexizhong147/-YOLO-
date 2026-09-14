# -*- coding: utf-8 -*-
"""验证码求解库。

用法
----
    from captcha_solver import Solver

    s = Solver()                       # 加载模型 (首次约 2 秒)
    img = "001.png"                    # 路径 / numpy / PIL / bytes 都可以

    s.arrows(img)                      # -> ['←', '↘', '→']        箭头朝向序列
    s.objects(img)                     # -> [ {物种, 朝向, 中心, 框}, ... ]  所有物体
    s.click_points(img)                # -> [(x,y), (x,y), (x,y)]   按箭头顺序要点的坐标
    s.solve(img)                       # -> 全部结果 (结构化对象)

    s.objects(img, syntax='dir')       # -> ['↓', '↑', '←', '↖']   只要朝向
    s.objects(img, syntax='center')    # -> [(120,90), ...]        只要中心
    s.objects(img, syntax='species')   # -> ['fish', 'fish', ...]  只要物种
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .arrows import detect_arrows
from .config import CODE2SYM, N_OBJECTS, SPECIES, SPECIES_CN, SYMS, ANGLE

__all__ = ['Solver', 'CaptchaResult', 'Obj', 'SolverError']

_HERE = Path(__file__).resolve().parent
DEFAULT_DET = _HERE / 'models' / 'detector_5species_obb.pt'
DEFAULT_CLS = _HERE / 'models' / 'orientation_8class_cls.pt'

ImgLike = Union[str, Path, bytes, bytearray, np.ndarray, Image.Image]


class SolverError(RuntimeError):
    """输入无法解析或模型推理失败。"""


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class Obj:
    """一个检测到的物体。"""
    species: str                     # 'fish'
    dir: str                         # '↖'
    center: Tuple[int, int]          # (x, y) 像素
    box: Tuple[int, int, int, int]   # 外接矩形 x1,y1,x2,y2
    corners: List[List[float]]       # 旋转框四角 (像素, 环形顺序)
    det_conf: float                  # 检测置信度
    dir_conf: float                  # 判向置信度
    index: int = 0                   # 在图中出现的顺序

    @property
    def species_cn(self) -> str:
        return SPECIES_CN.get(self.species, self.species)

    @property
    def label(self) -> str:
        """如 'fish_↖'"""
        return f'{self.species}_{self.dir}'

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['species_cn'] = self.species_cn
        d['label'] = self.label
        return d


@dataclass
class CaptchaResult:
    """一次识别的完整结果。"""
    arrows: List[str] = field(default_factory=list)       # 箭头朝向序列
    objects: List[Obj] = field(default_factory=list)      # 所有物体
    sequence: List[Dict[str, Any]] = field(default_factory=list)  # 按箭头顺序的点击目标
    image_size: Tuple[int, int] = (0, 0)                  # (W, H)
    warnings: List[str] = field(default_factory=list)

    @property
    def species(self) -> Optional[str]:
        """全图物种 (领域规则: 同图同物种)。"""
        return self.objects[0].species if self.objects else None

    @property
    def species_cn(self) -> Optional[str]:
        """全图物种的中文名。"""
        sp = self.species
        return SPECIES_CN.get(sp) if sp else None

    @property
    def click_points(self) -> List[Tuple[int, int]]:
        return [tuple(s['click']) for s in self.sequence]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'species': self.species,
            'species_cn': SPECIES_CN.get(self.species or '', None),
            'arrows': self.arrows,
            'n_objects': len(self.objects),
            'objects': [o.to_dict() for o in self.objects],
            'sequence': self.sequence,
            'click_points': [list(p) for p in self.click_points],
            'image_size': list(self.image_size),
            'warnings': self.warnings,
        }


# --------------------------------------------------------------------------
# 求解器
# --------------------------------------------------------------------------
class Solver:
    """两阶段验证码求解器。

    阶段1: YOLO11n-OBB (5 类) 找 4 个物体 + 物种 + 旋转框
    阶段2: YOLO11n-cls (8 类) 对每个物体裁剪图判朝向
    阶段3: cv2 规则读右上角箭头, 按顺序匹配物体
    """

    def __init__(self,
                 detector: Union[str, Path, None] = None,
                 classifier: Union[str, Path, None] = None,
                 conf: float = 0.05,
                 crop_size: int = 112,
                 pad: float = 0.15,
                 imgsz: int = 640,
                 topk: int = N_OBJECTS,
                 tta180: bool = True,
                 arrow_constraint: bool = False,
                 device: Optional[str] = None,
                 verbose: bool = False):
        """
        参数
        ----
        conf : float
            检测置信度阈值。**默认 0.05**（不是 0.25）——
            模型置信度标定偏低, 阈值太高会漏掉真物体;
            配合 topk=4 的领域规则反而更稳。
        topk : int
            每图最多保留几个物体（领域规则: 恰好 4 个）。
        tta180 : bool
            把裁剪图转 180° 再预测一次, 概率平移后平均。专治"头尾判反"。
        arrow_constraint : bool
            启用"4 个朝向里恰 3 个落在箭头集合内"的约束解码。收益有限, 默认关。
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise SolverError('需要 ultralytics: pip install ultralytics') from e

        self.det_path = Path(detector) if detector else DEFAULT_DET
        self.cls_path = Path(classifier) if classifier else DEFAULT_CLS
        for p in (self.det_path, self.cls_path):
            if not p.exists():
                raise SolverError(f'找不到模型文件: {p}')

        self.det = YOLO(str(self.det_path))
        self.cls = YOLO(str(self.cls_path))
        if device:
            try:
                self.det.to(device)
                self.cls.to(device)
            except Exception:
                pass

        self.conf = conf
        self.crop_size = crop_size
        self.pad = pad
        self.imgsz = imgsz
        self.topk = topk
        self.tta180 = tta180
        self.arrow_constraint = arrow_constraint
        self.verbose = verbose

        names = (list(self.cls.names.values())
                 if isinstance(self.cls.names, dict) else list(self.cls.names))
        self.code = [CODE2SYM.get(str(n).split('_')[-1], str(n)) for n in names]

    # ---------------- 输入处理 ----------------
    @staticmethod
    def _to_rgb(img: ImgLike) -> np.ndarray:
        """统一转成 RGB ndarray。支持 路径/bytes/numpy/PIL。"""
        if isinstance(img, Image.Image):
            return np.asarray(img.convert('RGB'))
        if isinstance(img, np.ndarray):
            a = img
            if a.ndim == 2:
                return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            if a.shape[2] == 4:
                return cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
            return a[:, :, ::-1].copy() if a.dtype == np.uint8 else a
        if isinstance(img, (bytes, bytearray)):
            buf = np.frombuffer(bytes(img), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise SolverError('bytes 解码失败, 不是有效图片')
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # 路径
        p = Path(img)
        if not p.exists():
            raise SolverError(f'图片不存在: {p}')
        # 注意: cv2.imread 无法处理中文路径, 用 np.fromfile
        buf = np.fromfile(str(p), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SolverError(f'图片解码失败: {p}')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ---------------- 内部: 裁剪 ----------------
    def _crop(self, img_rgb: np.ndarray, cs: np.ndarray) -> Optional[np.ndarray]:
        """从旋转框四角抠出正方形小图 (喂给判向分类器)。"""
        H, W = img_rgb.shape[:2]
        x1, y1 = float(cs[:, 0].min()), float(cs[:, 1].min())
        x2, y2 = float(cs[:, 0].max()), float(cs[:, 1].max())
        bw, bh = x2 - x1, y2 - y1
        X1 = int(max(0, x1 - bw * self.pad - 3)); Y1 = int(max(0, y1 - bh * self.pad - 3))
        X2 = int(min(W, x2 + bw * self.pad + 3)); Y2 = int(min(H, y2 + bh * self.pad + 3))
        if X2 - X1 < 6 or Y2 - Y1 < 6:
            return None
        sub = img_rgb[Y1:Y2, X1:X2]
        s = max(sub.shape[:2])
        sq = np.full((s, s, 3), 255, np.uint8)
        oy, ox = (s - sub.shape[0]) // 2, (s - sub.shape[1]) // 2
        sq[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
        out = cv2.resize(sq, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)          # ultralytics 吃 BGR

    # ---------------- 内部: 判向 ----------------
    def _classify(self, crop_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        pr = self.cls.predict(crop_bgr, imgsz=128, verbose=False)[0]
        probs = {self.code[i]: float(x) for i, x in enumerate(pr.probs.data.cpu().numpy())}
        if self.tta180:
            pr2 = self.cls.predict(cv2.rotate(crop_bgr, cv2.ROTATE_180),
                                   imgsz=128, verbose=False)[0]
            p2 = {self.code[i]: float(x) for i, x in enumerate(pr2.probs.data.cpu().numpy())}
            probs = {s: 0.5 * (probs[s] + p2[SYMS[(SYMS.index(s) + 4) % 8]]) for s in SYMS}
        sym = max(probs, key=probs.get)
        return sym, float(probs[sym]), probs

    # ---------------- 内部: 后处理 ----------------
    @staticmethod
    def _iou_box(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _dedup(self, objs: List[Obj], thr: float = 0.35) -> List[Obj]:
        keep: List[Obj] = []
        for o in sorted(objs, key=lambda x: -x.det_conf):
            if all(self._iou_box(o.box, k.box) < thr for k in keep):
                keep.append(o)
        return keep

    @staticmethod
    def _unify_species(objs: List[Obj]) -> List[Obj]:
        """领域规则: 一张图里 4 个物体物种相同 -> 取多数派。"""
        if not objs:
            return objs
        from collections import Counter
        maj = Counter(o.species for o in objs).most_common(1)[0][0]
        for o in objs:
            o.species = maj
        return objs

    def _apply_arrow_constraint(self, objs: List[Obj], probs_list, arrows) -> None:
        """约束解码: 4 个物体里恰好 3 个的朝向落在箭头集合内。"""
        import math
        if len(arrows) != 3 or len(objs) < 2:
            return
        A = set(arrows)
        outside = [s for s in SYMS if s not in A]
        best = None
        for di in range(len(objs)):
            total, assign, ok = 0.0, [], True
            for i, o in enumerate(objs):
                cands = outside if i == di else [s for s in SYMS if s in A]
                if not cands:
                    ok = False; break
                s = max(cands, key=lambda x: probs_list[i].get(x, 0.0))
                assign.append(s)
                total += math.log(max(probs_list[i].get(s, 1e-9), 1e-12))
            if ok and (best is None or total > best[0]):
                best = (total, assign)
        if best:
            for o, s in zip(objs, best[1]):
                o.dir = s

    # ======================================================================
    # 公开 API —— 按"语法"返回不同结果
    # ======================================================================
    def arrows(self, img: ImgLike) -> List[str]:
        """只读右上角箭头。-> ['←', '↘', '→']"""
        return [a['dir'] for a in detect_arrows(self._to_rgb(img))]

    def objects(self, img: ImgLike,
                syntax: str = 'full') -> Union[List[Obj], List[str], List[Tuple[int, int]]]:
        """检测所有物体。

        syntax
        ------
        'full'     -> [Obj, ...]              完整对象 (物种/朝向/中心/框/置信度)
        'dir'      -> ['↖', '→', ...]          只要朝向
        'center'   -> [(x,y), ...]            只要中心坐标
        'species'  -> ['fish', ...]           只要物种
        'label'    -> ['fish_↖', ...]         物种_朝向
        'box'      -> [(x1,y1,x2,y2), ...]    只要外接框
        'dict'     -> [{...}, ...]            纯字典 (可直接 json 序列化)
        """
        r = self._run(img)
        if syntax == 'full':
            return r.objects
        if syntax == 'dir':
            return [o.dir for o in r.objects]
        if syntax == 'center':
            return [tuple(o.center) for o in r.objects]
        if syntax == 'species':
            return [o.species for o in r.objects]
        if syntax == 'label':
            return [o.label for o in r.objects]
        if syntax == 'box':
            return [tuple(o.box) for o in r.objects]
        if syntax == 'dict':
            return [o.to_dict() for o in r.objects]
        raise SolverError(f"未知 syntax: {syntax!r} (可选 full/dir/center/species/label/box/dict)")

    def click_points(self, img: ImgLike, syntax: str = 'points'):
        """按箭头顺序, 输出要点击的坐标。

        syntax
        ------
        'points' -> [(x,y), (x,y), (x,y)]     默认
        'list'   -> [[x,y], [x,y], [x,y]]     列表形式 (好存 json)
        'flat'   -> [x1,y1,x2,y2,x3,y3]       展平
        'detail' -> [{step,target,click,species,dir}, ...]
        """
        r = self._run(img)
        if syntax == 'points':
            return r.click_points
        if syntax == 'list':
            return [[int(x), int(y)] for x, y in r.click_points]
        if syntax == 'flat':
            return [int(v) for p in r.click_points for v in p]
        if syntax == 'detail':
            return r.sequence
        raise SolverError(f"未知 syntax: {syntax!r} (可选 points/list/flat/detail)")

    def solve(self, img: ImgLike) -> CaptchaResult:
        """完整结果 (箭头 + 所有物体 + 点击序列)。"""
        return self._run(img)

    def solve_dict(self, img: ImgLike) -> Dict[str, Any]:
        """完整结果, 纯 dict (可直接 json.dumps)。"""
        return self._run(img).to_dict()

    # ---------------- 核心 ----------------
    def _run(self, img: ImgLike) -> CaptchaResult:
        img_rgb = self._to_rgb(img)
        H, W = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        arrows = [a['dir'] for a in detect_arrows(img_rgb)]

        # ---- 阶段1: 检测 ----
        r = self.det.predict(img_bgr, imgsz=self.imgsz, conf=self.conf,
                             agnostic_nms=True, verbose=self.verbose)[0]
        cand = []
        if r.obb is not None and len(r.obb):
            for xy, c, cf in zip(r.obb.xyxyxyxy.cpu().numpy(),
                                 r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()):
                cand.append((float(cf), int(c), xy))
        cand.sort(key=lambda x: -x[0])
        cand = cand[:self.topk]              # 领域规则: 每图恰好 4 个物体

        objs: List[Obj] = []
        probs_list: List[Dict[str, float]] = []
        for cf, ci, xy in cand:
            cs = np.asarray(xy, np.float64)
            x1 = float(cs[:, 0].min()); y1 = float(cs[:, 1].min())
            x2 = float(cs[:, 0].max()); y2 = float(cs[:, 1].max())
            crop = self._crop(img_rgb, cs)
            if crop is None:
                continue
            sym, p, probs = self._classify(crop)
            objs.append(Obj(
                species=SPECIES[ci], dir=sym,
                center=(int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))),
                box=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                corners=[[round(float(a), 2), round(float(b), 2)] for a, b in cs],
                det_conf=round(cf, 3), dir_conf=round(p, 3)))
            probs_list.append(probs)

        objs = self._dedup(objs)
        objs = self._unify_species(objs)
        if self.arrow_constraint:
            self._apply_arrow_constraint(objs, probs_list, arrows)
        for i, o in enumerate(objs):
            o.index = i

        warnings: List[str] = []
        if len(objs) != N_OBJECTS:
            warnings.append(f'检测到 {len(objs)} 个物体 (预期 {N_OBJECTS} 个)')
        if len(arrows) != 3:
            warnings.append(f'读到 {len(arrows)} 个箭头 (预期 3 个): {arrows}')

        # ---- 阶段2: 按箭头顺序匹配 ----
        used, sequence = set(), []
        for a in arrows:
            best, bi = -1.0, -1
            for i, o in enumerate(objs):
                if i in used or o.dir != a:
                    continue
                if o.dir_conf > best:
                    best, bi = o.dir_conf, i
            if bi >= 0:
                used.add(bi)
                o = objs[bi]
                sequence.append({'step': len(sequence) + 1, 'target': a,
                                 'click': list(o.center), 'species': o.species,
                                 'species_cn': o.species_cn, 'dir': o.dir,
                                 'dir_conf': o.dir_conf, 'index': bi})
        return CaptchaResult(arrows=arrows, objects=objs, sequence=sequence,
                             image_size=(W, H), warnings=warnings)
