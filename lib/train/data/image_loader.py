try:
    import jpeg4py
    HAS_JPEG4PY = True
except ImportError:
    HAS_JPEG4PY = False

import cv2 as cv
from PIL import Image
import numpy as np

davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def default_image_loader(path):
    if HAS_JPEG4PY:
        im = jpeg4py_loader(path)
        if im is not None:
            return im
    return opencv_loader(path)


def jpeg4py_loader(path):
    if not HAS_JPEG4PY:
        return opencv_loader(path)
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def opencv_loader(path):
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def jpeg4py_loader_w_failsafe(path):
    if HAS_JPEG4PY:
        try:
            return jpeg4py.JPEG(path).decode()
        except:
            pass
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def opencv_seg_loader(path):
    try:
        return cv.imread(path)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def imread_indexed(filename):
    im = Image.open(filename)
    annotation = np.atleast_3d(im)[..., 0]
    return annotation


def imwrite_indexed(filename, array, color_palette=None):
    if color_palette is None:
        color_palette = davis_palette
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')
