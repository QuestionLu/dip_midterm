"""核心演算法的自動化測試（pytest）。

執行：  pytest -v
重點：手刻卷積需與「樸素四層迴圈」及 cv2.filter2D 結果一致。
"""
import math
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dip import color, convolution as C, face, io_utils, metrics  # noqa: E402

RNG = np.random.default_rng(0)


def rand_img(h=24, w=31, c=3):
    shape = (h, w, c) if c else (h, w)
    return RNG.integers(0, 256, size=shape, dtype=np.uint8)


# ----------------------------------------------------------------------------
# 卷積
# ----------------------------------------------------------------------------
def naive_convolution(x, k):
    """依定義 y[i,j] = Σ k[u,v] x[i-u, j-v] 的四層迴圈（補零，same 輸出）。"""
    h, w = x.shape
    kh, kw = k.shape
    a, b = kh // 2, kw // 2
    y = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            s = 0.0
            for u in range(-a, a + 1):
                for v in range(-b, b + 1):
                    ii, jj = i - u, j - v
                    if 0 <= ii < h and 0 <= jj < w:
                        s += k[u + a, v + b] * x[ii, jj]
            y[i, j] = s
    return y


@pytest.mark.parametrize("shape", [(3, 3), (5, 5), (3, 5), (1, 1), (7, 3)])
def test_convolution_matches_definition(shape):
    x = rand_img(12, 15, 0)
    k = RNG.normal(size=shape)
    got = C.convolve2d(x, k, flip=True, border="zeros")
    assert np.allclose(got, naive_convolution(x.astype(float), k), atol=1e-9)


def test_identity_kernel_returns_same_image():
    x = rand_img()
    k = C.parse_kernel(C.PRESETS["原圖 (Identity 3×3)"])
    assert np.array_equal(C.apply_filter(x, k), x)


def test_flip_matters_for_asymmetric_kernel():
    x = rand_img(10, 10, 0)
    k = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    conv = C.convolve2d(x, k, flip=True)
    corr = C.convolve2d(x, k, flip=False)
    assert not np.allclose(conv, corr)
    # 卷積 = 相關（kernel 翻轉後）
    assert np.allclose(conv, C.convolve2d(x, k[::-1, ::-1], flip=False))


@pytest.mark.parametrize(
    "border,cv_border",
    [
        ("zeros", cv2.BORDER_CONSTANT),
        ("replicate", cv2.BORDER_REPLICATE),
        ("reflect", cv2.BORDER_REFLECT_101),
    ],
)
@pytest.mark.parametrize("name", ["高斯平滑 (Gaussian 5×5)", "銳化 (Sharpen 3×3)", "浮雕 (Emboss 3×3)"])
def test_matches_opencv_filter2d(border, cv_border, name):
    x = rand_img(40, 50, 3)
    k, _ = C.normalize_kernel(C.parse_kernel(C.PRESETS[name]), "abs")
    mine = C.apply_filter(x, k, flip=False, border=border)  # filter2D 是相關運算
    ref = cv2.filter2D(x, cv2.CV_64F, k, borderType=cv_border)
    ref = np.clip(np.rint(ref), 0, 255).astype(np.uint8)
    assert np.abs(mine.astype(int) - ref.astype(int)).max() <= 1


def test_gaussian_normalized_preserves_mean_brightness():
    x = np.full((20, 20, 3), 100, dtype=np.uint8)
    k, _ = C.normalize_kernel(C.parse_kernel(C.PRESETS["高斯平滑 (Gaussian 3×3)"]), "sum")
    out = C.apply_filter(x, k, border="replicate")
    assert np.all(out == 100)


def test_output_is_clipped_to_uint8_range():
    x = np.full((8, 8), 200, dtype=np.uint8)
    out = C.apply_filter(x, np.full((3, 3), 5.0), border="replicate")
    assert out.dtype == np.uint8 and np.all(out == 255)
    out2 = C.apply_filter(x, np.full((3, 3), -5.0), border="replicate")
    assert np.all(out2 == 0)


def test_offset_shifts_output():
    x = np.zeros((8, 8), dtype=np.uint8)
    out = C.apply_filter(x, np.zeros((3, 3)), offset=128)
    assert np.all(out == 128)


# ----------------------------------------------------------------------------
# 卷積核輸入防呆
# ----------------------------------------------------------------------------
def test_parse_basic_and_separators():
    a = C.parse_kernel("1 2 3\n4 5 6\n7 8 9")
    b = C.parse_kernel("1,2,3;4,5,6;7,8,9")
    c = C.parse_kernel("[[1,2,3],[4,5,6],[7,8,9]]")
    assert np.array_equal(a, b) and np.array_equal(a, c) and a.shape == (3, 3)


def test_parse_fractions_and_decimals():
    k = C.parse_kernel("1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9")
    assert math.isclose(k.sum(), 1.0)
    assert C.parse_kernel("0.5").shape == (1, 1)
    assert C.parse_kernel("1e-1")[0, 0] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "text,keyword",
    [
        ("", "空"),
        ("   \n  ", "空"),
        ("1 2 3\n4 5", "不一致"),
        ("1 2 3\n4 5 6", "非方陣"),
        ("1 2\n3 4", "奇數"),
        ("1 a 3\n4 5 6\n7 8 9", "不是有效數字"),
        ("nan 1 1\n1 1 1\n1 1 1", "不是有效數字"),
        ("inf 1 1\n1 1 1\n1 1 1", "不是有效數字"),
        ("1/0 1 1\n1 1 1\n1 1 1", "分母為 0"),
        ("1e9 1 1\n1 1 1\n1 1 1", "過大"),
    ],
)
def test_parse_rejects_bad_input(text, keyword):
    with pytest.raises(C.KernelError, match=keyword):
        C.parse_kernel(text)


def test_nonsquare_allowed_when_flag_set():
    assert C.parse_kernel("1 2 3\n4 5 6\n7 8 9\n1 2 3\n4 5 6", allow_nonsquare=True).shape == (5, 3)


def test_kernel_too_large():
    big = "\n".join(" ".join(["1"] * 17) for _ in range(17))
    with pytest.raises(C.KernelError, match="過大"):
        C.parse_kernel(big)


def test_kernel_larger_than_image():
    with pytest.raises(C.KernelError, match="大於影像"):
        C.convolve2d(np.zeros((3, 3)), np.ones((5, 5)))


def test_zero_sum_normalization_does_not_divide_by_zero():
    k = C.parse_kernel(C.PRESETS["邊緣偵測 (Laplacian 3×3)"])
    assert k.sum() == 0
    out, warns = C.normalize_kernel(k, "sum")
    assert np.array_equal(out, k) and np.all(np.isfinite(out)) and warns
    out2, warns2 = C.normalize_kernel(k, "abs")
    assert math.isclose(np.abs(out2).sum(), 1.0) and not warns2


def test_all_zero_kernel_normalization_warns():
    out, warns = C.normalize_kernel(np.zeros((3, 3)), "abs")
    assert warns and np.all(out == 0)


# ----------------------------------------------------------------------------
# PSNR
# ----------------------------------------------------------------------------
def test_psnr_identical_is_inf():
    x = rand_img()
    assert math.isinf(metrics.psnr(x, x))


def test_psnr_known_value():
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 110, dtype=np.uint8)  # MSE = 100
    assert metrics.psnr(a, b) == pytest.approx(10 * math.log10(255 ** 2 / 100))
    assert metrics.psnr(a, b) == pytest.approx(28.1308, abs=1e-3)


def test_psnr_matches_opencv():
    a, b = rand_img(30, 30), rand_img(30, 30)
    assert metrics.psnr(a, b) == pytest.approx(cv2.PSNR(a, b), abs=1e-6)


def test_psnr_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.psnr(np.zeros((4, 4)), np.zeros((5, 5)))


def test_psnr_is_symmetric_and_more_noise_means_lower():
    a = rand_img()
    mild = np.clip(a.astype(int) + 2, 0, 255).astype(np.uint8)
    strong = np.clip(a.astype(int) + 40, 0, 255).astype(np.uint8)
    assert metrics.psnr(a, mild) > metrics.psnr(a, strong)
    assert metrics.psnr(a, mild) == pytest.approx(metrics.psnr(mild, a))


# ----------------------------------------------------------------------------
# 色彩空間與直方圖
# ----------------------------------------------------------------------------
def test_yuv_matches_opencv():
    x = rand_img(40, 40)
    mine = color.split_yuv(x)
    ref = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
    for i, name in enumerate("YUV"):
        assert np.abs(mine[name].astype(int) - ref[..., i].astype(int)).max() <= 2, name


def test_yuv_roundtrip():
    x = rand_img(40, 40)
    back = color.yuv_to_rgb(color.rgb_to_yuv(x))
    assert np.abs(back.astype(int) - x.astype(int)).max() <= 1


def test_gray_input_has_neutral_chroma():
    g = np.full((6, 6, 3), 90, dtype=np.uint8)
    yuv = color.split_yuv(g)
    assert np.all(np.abs(yuv["U"].astype(int) - 128) <= 1)
    assert np.all(np.abs(yuv["V"].astype(int) - 128) <= 1)
    assert np.all(np.abs(yuv["Y"].astype(int) - 90) <= 1)


def test_histogram_matches_opencv_and_sums_to_pixel_count():
    x = rand_img(33, 47, 0)
    h = color.histogram(x)
    ref = cv2.calcHist([x], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    assert h.shape == (256,) and np.array_equal(h, ref) and h.sum() == x.size


def test_split_rgb_channels():
    x = rand_img()
    s = color.split_rgb(x)
    assert np.array_equal(s["R"], x[..., 0]) and np.array_equal(s["B"], x[..., 2])


# ----------------------------------------------------------------------------
# 影像載入
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("ext,fmt", [(".png", "PNG"), (".jpg", "JPEG"), (".jpeg", "JPEG"), (".bmp", "BMP")])
def test_load_supported_formats(ext, fmt):
    x = rand_img(20, 30)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t" + ext)
        Image.fromarray(x).save(p)
        arr, info = io_utils.load_image(p)
    assert arr.shape == (20, 30, 3) and arr.dtype == np.uint8
    assert info.fmt == fmt and (info.width, info.height) == (30, 20)
    assert info.channels == 3 and info.file_size_bytes > 0
    if ext in (".png", ".bmp"):
        assert np.array_equal(arr, x)  # 無損格式應完全相同


def test_load_grayscale_and_rgba():
    with tempfile.TemporaryDirectory() as d:
        g = os.path.join(d, "g.png")
        Image.fromarray(rand_img(10, 10, 0), "L").save(g)
        arr, info = io_utils.load_image(g)
        assert arr.shape == (10, 10, 3) and info.mode == "L" and info.channels == 1

        a = os.path.join(d, "a.png")
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)  # 全透明
        Image.fromarray(rgba, "RGBA").save(a)
        arr, info = io_utils.load_image(a)
        assert info.channels == 4 and np.all(arr == 255)  # 透明處以白底合成


def test_load_rejects_bad_files():
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, "x.txt")
        open(txt, "w").write("hi")
        with pytest.raises(io_utils.ImageLoadError, match="不支援"):
            io_utils.load_image(txt)
        fake = os.path.join(d, "fake.png")
        open(fake, "wb").write(b"not an image")
        with pytest.raises(io_utils.ImageLoadError, match="無法解析"):
            io_utils.load_image(fake)
    with pytest.raises(io_utils.ImageLoadError):
        io_utils.load_image("/no/such/file.png")


def test_large_image_is_downscaled_but_info_keeps_original_size():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "big.png")
        Image.fromarray(np.zeros((100, io_utils.MAX_SIDE + 500, 3), dtype=np.uint8)).save(p)
        arr, info = io_utils.load_image(p)
    assert max(arr.shape[:2]) == io_utils.MAX_SIDE
    assert info.width == io_utils.MAX_SIDE + 500 and info.processed_size is not None
    assert "已等比縮小" in info.to_markdown()


# ----------------------------------------------------------------------------
# 人臉偵測
# ----------------------------------------------------------------------------
def _face_image():
    skimage_data = pytest.importorskip("skimage.data")
    return np.ascontiguousarray(skimage_data.astronaut())


@pytest.mark.parametrize("backend", ["yunet", "haar"])
def test_blank_image_has_no_faces(backend):
    res = face.detect_faces(np.full((240, 320, 3), 127, dtype=np.uint8), backend)
    assert res.count == 0 and res.annotated.shape == (240, 320, 3)


@pytest.mark.parametrize("backend", ["yunet", "haar"])
def test_detects_single_face(backend):
    img = _face_image()
    res = face.detect_faces(img, backend)
    assert res.count >= 1
    assert all(0 <= f.x < img.shape[1] and 0 <= f.y < img.shape[0] and f.w > 0 and f.h > 0 for f in res.faces)
    assert not np.array_equal(res.annotated, img)  # 有畫框
    if backend == "yunet":
        assert all(0.0 <= f.score <= 1.0 for f in res.faces)


def test_detects_multiple_faces_yunet():
    img = _face_image()
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2))
    canvas = np.full((h, w + small.shape[1], 3), 255, dtype=np.uint8)
    canvas[:h, :w] = img
    canvas[: small.shape[0], w:] = small
    res = face.detect_faces(canvas, "yunet")
    assert res.count >= 2


def test_threshold_monotonic():
    img = _face_image()
    loose = face.detect_faces(img, "yunet", score_threshold=0.3).count
    strict = face.detect_faces(img, "yunet", score_threshold=0.99).count
    assert loose >= strict


def test_unknown_backend():
    with pytest.raises(face.FaceDetectionError):
        face.detect_faces(np.zeros((10, 10, 3), np.uint8), "nope")
