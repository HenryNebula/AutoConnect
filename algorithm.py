import logging
from PIL import Image
from skimage.color import rgb2hsv, rgb2gray
from itertools import product
from skimage import measure, feature, morphology, metrics
from scipy import ndimage
import numpy as np

from sklearn.feature_extraction.image import extract_patches
from sklearn.cluster import AgglomerativeClustering


def detect_tile_region(img: Image, area: int, crop_coordinates=None):

    def get_contour_area(contours, contour_label):
        filled = ndimage.binary_fill_holes(contours == contour_label).ravel()
        area = np.bincount(filled)[1]
        return area

    def center_crop(image, old_area, new_area):
        width, height = image.size
        target_width, target_height = tuple(map(lambda n: int(n * np.sqrt(new_area / old_area)), image.size))
        border_width = (width - target_width) // 2 + 1
        border_height = (height - target_height) // 2 + 1
        return image.crop((border_width, border_height, border_width + target_width, border_height + target_height))

    if crop_coordinates is None:
        edges = feature.canny(np.array(img.convert("L")), sigma=7)
        enhanced = morphology.binary_dilation(edges, np.ones((5, 5)))
        enhanced = morphology.binary_erosion(enhanced, np.ones((4, 4)))
        contours = measure.label(enhanced, background=0, connectivity=2)
        contour_labels = np.unique(contours)
        contour_areas = list(map(lambda c: get_contour_area(contours, c), contour_labels))
        contour_areas = np.array(contour_areas)
        target_label_idx = np.argmin(abs(contour_areas - area))
        target_area = contour_areas[target_label_idx]
        assert abs(target_area - area) < 0.1 * area, f"{target_area} too far from {area}"
        target_label = contour_labels[target_label_idx]
        nonzero_idx = np.where((contours == target_label) != 0)
        upper_left = min(nonzero_idx[1]), min(nonzero_idx[0]) # first non-zero col and first non-zero row
        buttom_right = max(nonzero_idx[1]), max(nonzero_idx[0])
        crop_coordinates = upper_left + buttom_right

    left, upper, right, buttom = crop_coordinates
    target_area = (right - left) * (buttom - upper)
    region_with_border = img.crop(crop_coordinates)

    if target_area <= area:
        region = region_with_border
    else:
        region = region_with_border
        # center_crop(region_with_border, old_area=target_area, new_area=area)

    return region, crop_coordinates


def crop_tiles(region, tile_size):
    tiles = extract_patches(np.array(region), (tile_size, tile_size, 3), extraction_step=tile_size)
    tiles = tiles.squeeze()  # shape: (nrows, ncols, tile_size, tile_size, 3)
    return tiles


def detect_edges(patch):
    return feature.canny(rgb2gray(patch))


def find_maximum_similarity(base_tiles, target_tiles, multichannel=True):
    nrow, ncol = base_tiles.shape[:2]
    max_sim = -np.ones(shape=(nrow, ncol))

    for row, col in product(range(nrow), range(ncol)):
        target_tile = target_tiles[row, col] if multichannel else detect_edges(target_tiles[row, col])
        for base_row, base_col in product(range(nrow), range(ncol)):
            base_tile = base_tiles[base_row, base_col] if multichannel else detect_edges(base_tiles[base_row, base_col])
            sim = metrics.structural_similarity(target_tile, base_tile, multichannel=multichannel, win_size=11)
            max_sim[row, col] = max(sim, max_sim[row, col])
    return max_sim


def generate_background_mask(tiles, pad=False, metric="grayscale"):
    # tiles have shape (n_rows, n_cols, tile_size, tile_size, 3)
    # generate a mask where True stands for background
    nrows, ncols = tiles.shape[:2]
    tiles = tiles.reshape(-1, *(tiles.shape[-3:]))

    def get_tile_features(tile):
        if metric == "grayscale":
            return np.array([tile[..., c].std() for c in range(3)])
        elif metric == "lightness":
            return np.std(rgb2hsv(tile)[..., -1])
        else:
            raise ValueError(f"{metric} is not a valid metric.")

    features = list(map(get_tile_features, tiles))
    feature_dim = len(features[0])
    features = np.array(features).reshape(-1, feature_dim)

    cluster = AgglomerativeClustering(n_clusters=2)
    cluster.fit(features)
    labels = cluster.labels_
    centroids = list(map(lambda label: np.median(features[labels == label]), labels))
    background_label = labels[np.argmin(centroids)]
    mask = labels == background_label
    mask = mask.reshape(nrows, ncols)
    if pad:
        mask = np.pad(mask, pad_width=1, constant_values=1)
    return mask


def generate_background_mask_v2(base_tiles, target_tiles, pad=False):
    # more accurate but slower than v1
    max_sim = find_maximum_similarity(base_tiles, target_tiles)
    mask = max_sim < 0.9
    if pad:
        mask = np.pad(mask, pad_width=1, constant_values=1)
    return mask


def check_background_consistency(bg_mask, num_eliminated_pairs, is_padded=True):
    if is_padded:
        hole_cnt = np.sum(bg_mask[1:-1, 1:-1])
    else:
        hole_cnt = np.sum(bg_mask)
    target = num_eliminated_pairs * 2
    is_consistent = hole_cnt == target
    if not is_consistent:
        logging.warning(f"Number of holes does not equal the number of eliminated patches: "
                        f"{hole_cnt} != {target}.")
    return is_consistent


def pattern_matching(screenshot: Image, pattern: Image):
    width, height = pattern.size
    result = feature.match_template(np.array(screenshot), np.array(pattern))
    max_idx = np.argmax(result)
    y, x, _ = np.unravel_index(max_idx, result.shape)
    return (x, y, width, height) if np.max(result) > 0.8 else None


# def detect_points(screenshot: Image, coords):
#     # given x, y, width, height for points_label
#     # coords = (x + width, y, x + width * 2, y + height)
#
#     points_region_thumbnail = screenshot.crop(coords)
#     tb = np.array(points_region_thumbnail.convert("L"))
#     tb = tb > 170 # simple enhancement: binarize the image
#     points_str = pytesseract.image_to_string(np.array(tb), config='--psm 6').strip()
#     try:
#         points = int(points_str)
#     except ValueError:
#         points = 0
#
#     return points

