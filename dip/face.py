"""功能 4：人臉偵測。

提供兩種偵測器，介面一致：
  * YuNet  — OpenCV 內建的輕量級深度學習偵測器（cv2.FaceDetectorYN, ONNX），
             輸出 0~1 的信心度分數。
  * Haar   — 經典 Haar-Cascade（Viola-Jones）。OpenCV 不輸出機率，
             這裡用 detectMultiScale3 的 levelWeights（最後一級串接分類器的累積權重）
             當作「信心分數」，數值越大越可信，但它不是 0~1 的機率。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
HAAR_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"

BACKENDS = {
    "YuNet（深度學習，輕量）": "yunet",
    "Haar Cascade（經典特徵）": "haar",
}
YUNET_MAX_SIDE = 1280  # 超過則縮小後偵測，再把座標換算回原圖


class FaceDetectionError(RuntimeError):
    pass


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass
class DetectionResult:
    faces: List[Face] = field(default_factory=list)
    annotated: np.ndarray = None  # RGB uint8
    backend: str = ""
    score_note: str = ""
    elapsed_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.faces)


# 注意：Windows 上 OpenCV（C++ 端）無法開啟含中文等非 ASCII 字元的路徑，
# 例如 E:\\高科大\\...\\models\\xxx.onnx 會出現 "Can't read ONNX file"。
# 因此一律由 Python 讀成位元組，再以「記憶體」方式交給 OpenCV，與資料夾名稱無關。
@lru_cache(maxsize=None)
def _read_model_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FaceDetectionError(f"找不到模型檔：{path.name}（請確認 models/ 資料夾完整）")
    return path.read_bytes()


def _make_yunet(size, score_threshold: float, nms_threshold: float):
    buf = np.frombuffer(_read_model_bytes(YUNET_PATH), dtype=np.uint8)
    return cv2.FaceDetectorYN.create(
        "onnx", buf, np.empty(0, dtype=np.uint8), size, float(score_threshold), float(nms_threshold), 5000
    )


def _make_haar() -> "cv2.CascadeClassifier":
    xml = _read_model_bytes(HAAR_PATH).decode("utf-8")
    fs = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    cascade = cv2.CascadeClassifier()
    if not cascade.read(fs.getFirstTopLevelNode()) or cascade.empty():
        raise FaceDetectionError("Haar 模型載入失敗。")
    return cascade


def _detect_yunet(rgb: np.ndarray, score_threshold: float, nms_threshold: float) -> List[Face]:
    h, w = rgb.shape[:2]
    scale = 1.0
    work = rgb
    if max(h, w) > YUNET_MAX_SIDE:
        scale = YUNET_MAX_SIDE / max(h, w)
        work = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    wh, ww = work.shape[:2]
    bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
    detector = _make_yunet((ww, wh), score_threshold, nms_threshold)
    detector.setInputSize((ww, wh))
    _, rows = detector.detect(bgr)
    faces: List[Face] = []
    if rows is None:
        return faces
    for r in rows:
        x, y, bw, bh = (r[0:4] / scale).tolist()
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
        if x1 > x0 and y1 > y0:
            faces.append(Face(x0, y0, x1 - x0, y1 - y0, float(r[-1])))
    return faces


def _detect_haar(rgb: np.ndarray, scale_factor: float, min_neighbors: int, min_size: int) -> List[Face]:
    cascade = _make_haar()  # 每次呼叫建立新物件（CascadeClassifier 非執行緒安全，不共用）
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)  # 直方圖等化，降低光照差異影響
    rects, _, weights = cascade.detectMultiScale3(
        gray,
        scaleFactor=float(scale_factor),
        minNeighbors=int(min_neighbors),
        minSize=(int(min_size), int(min_size)),
        outputRejectLevels=True,
    )
    faces: List[Face] = []
    for (x, y, w, h), wt in zip(rects, np.asarray(weights).reshape(-1)):
        faces.append(Face(int(x), int(y), int(w), int(h), float(wt)))
    return faces


def _draw(rgb: np.ndarray, faces: List[Face], is_prob: bool) -> np.ndarray:
    out = np.ascontiguousarray(rgb.copy())
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 400))
    font_scale = max(0.5, max(h, w) / 1200)
    color = (0, 255, 0)
    for i, f in enumerate(faces, start=1):
        cv2.rectangle(out, (f.x, f.y), (f.x + f.w, f.y + f.h), color, thick)
        label = f"#{i} {f.score:.2f}" if is_prob else f"#{i} w={f.score:.1f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = f.y - 6 if f.y - th - 10 > 0 else f.y + f.h + th + 6
        cv2.rectangle(out, (f.x, ty - th - 4), (f.x + tw + 6, ty + base), color, -1)
        cv2.putText(out, label, (f.x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)
    return out


def detect_faces(
    rgb: np.ndarray,
    backend: str = "yunet",
    *,
    score_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: int = 30,
) -> DetectionResult:
    """偵測人臉並回傳標註後影像、人臉列表（含信心度）與耗時。"""
    t0 = time.perf_counter()
    if backend == "yunet":
        faces = _detect_yunet(rgb, score_threshold, nms_threshold)
        note = "YuNet 信心度為 0~1 的分類機率"
        name = "YuNet"
        is_prob = True
    elif backend == "haar":
        faces = _detect_haar(rgb, scale_factor, min_neighbors, min_size)
        note = "Haar 分數為串接分類器累積權重（非機率，越大越可信）"
        name = "Haar Cascade"
        is_prob = False
    else:
        raise FaceDetectionError(f"未知的偵測器：{backend}")
    faces.sort(key=lambda f: (f.y, f.x))
    return DetectionResult(
        faces=faces,
        annotated=_draw(rgb, faces, is_prob),
        backend=name,
        score_note=note,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
