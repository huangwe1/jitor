import numpy as np
import jittor as jt


def generate_heatmap(bboxes, patch_size=320, stride=16):
    gaussian_maps = []
    heatmap_size = patch_size // stride
    for single_patch_bboxes in bboxes:
        bs = single_patch_bboxes.shape[0]
        gt_scoremap = jt.zeros(bs, heatmap_size, heatmap_size)
        classes = jt.arange(bs).int64()
        bbox = single_patch_bboxes * heatmap_size
        wh = bbox[:, 2:]
        centers_int = (bbox[:, :2] + wh / 2).round()
        CenterNetHeatMap.generate_score_map(gt_scoremap, classes, wh, centers_int, 0.7)
        gaussian_maps.append(gt_scoremap)
    return gaussian_maps


class CenterNetHeatMap(object):
    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, min_overlap)
        radius = jt.clamp(radius, min_v=0)
        radius = radius.int32().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = int(gt_class[i].item())
            CenterNetHeatMap.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        box_tensor = box_size
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = jt.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = jt.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = jt.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return jt.minimum(r1, jt.minimum(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        m, n = radius
        y, x = np.ogrid[-m: m + 1, -n: n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetHeatMap.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = jt.array(gaussian).float32()
        x, y = int(center[0].item()), int(center[1].item())
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top: y + bottom, x - left: x + right]
        masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = jt.maximum(masked_fmap, masked_gaussian * k)
            fmap[y - top: y + bottom, x - left: x + right] = masked_fmap


def get_pred(score_map_ctr, size_map, offset_map, feat_size):
    max_score, idx = jt.argmax(score_map_ctr.flatten(1), dim=1, keepdims=True)

    idx = idx.unsqueeze(1).expand([idx.shape[0], 2, 1])
    size = jt.gather(size_map.flatten(2), 2, idx).squeeze(-1)
    offset = jt.gather(offset_map.flatten(2), 2, idx).squeeze(-1)

    return size * feat_size, offset
