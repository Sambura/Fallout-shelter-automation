from PIL import ImageGrab, Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# local terms:
#       bounds: instance of Bounds class
#       bbox: array/tuple of numbers [x0, y0, x1, y1]
#       rect: array/tuple of numbers [x0, y0, width, height]

def grab_real_screen(bbox):
    if bbox is not None: bbox = [bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1]
    return np.array(ImageGrab.grab(bbox))

mock_frames = None
mock_index = 0
mock_mode = None

# frames / fixed mode
def set_mock_frames(frames, mode='frames'):
    global mock_frames, mock_index, mock_mode
    mock_frames = frames
    mock_index = 0
    mock_mode = mode

def get_mock_frames(): return mock_frames 

def load_mock_frames(path, mode='frames'):
    frames = []

    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            frames.append(img_array)
    
    print(f'Loaded {len(frames)} mock frames')
    set_mock_frames(frames, mode)

def grab_mock_screen(bbox):
    global mock_index
    index = mock_index
    if mock_mode != 'fixed': mock_index += 1
    return mock_frames[index].copy()

__grab_screen = grab_real_screen

def set_grab_screen(mode='real'):
    global __grab_screen
    if mode == 'real':
        print('Warning: grab function set to real')
        __grab_screen = grab_real_screen
    elif mode == 'mock':
        print('Warning: grab function set to mock')
        __grab_screen = grab_mock_screen
    else:
        raise 'error'

def grab_screen(*args): return __grab_screen(*args)

DEBUG_COLORS = (np.array([plt.get_cmap('rainbow')(i) for i in np.linspace(0, 1, 8)]) * 255).astype(int)

def get_debug_color(value):
    return DEBUG_COLORS[value % len(DEBUG_COLORS)]

class Bounds:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.width = self.x_max - self.x_min + 1
        self.height = self.y_max - self.y_min + 1
        self.x = (x_min + x_max) // 2
        self.y = (y_min + y_max) // 2
        self.area = self.width * self.height
        self.shape = (self.height, self.width)
        self.pos = np.array([self.x, self.y])
        self.low_pos = np.array([self.x_min, self.y_min])

    def from_rect(x, y, width, height):
        return Bounds(x, y, x + width - 1, y + height - 1)

    def get_corners(self, get_8=False):
        corners = [(self.x_min, self.y_min), (self.x_min, self.y_max), (self.x_max, self.y_min), (self.x_max, self.y_max)]
        if get_8:
            return [*corners, (self.x_min, self.y), (self.x_max, self.y), (self.x, self.y_min), (self.x, self.y_max)]
        return corners

    def to_rect(self):
        return [self.x_min, self.y_min, self.width, self.height]

    def from_center(x, y, width, height):
        x_min = x - width // 2
        y_min = y - height // 2
        return Bounds.from_rect(x_min, y_min, width, height)

    def are_smaller_than(self, width, height=None):
        "specify one argument to compare to a square"
        if height is None: height = width
        return self.width < width or self.height < height

    def from_points(*points):
        np_points = np.array(points)
        xs, ys = np_points[:,0], np_points[:,1]
        return Bounds(np.min(xs), np.min(ys), np.max(xs), np.max(ys))

    def get_scaled_from_center(self, scale):
        half_width = int(scale * self.width / 2)
        half_height = int(scale * self.height / 2)
        return Bounds(self.x - half_width, self.y - half_height, self.x + half_width, self.y + half_height)
    
    def contains_bounds(self, bounds):
        if self.x_max < bounds.x_max or bounds.x_min < self.x_min: return False
        if self.y_max < bounds.y_max or bounds.y_min < self.y_min: return False
        return True

    def contains_point(self, x, y):
        return self.contains_bounds(Bounds(x, y, x, y))

    def to_slice(self, offset=None):
        if offset is None: offset = [0, 0]
        ox, oy = offset
        return (slice(self.y_min + oy, self.y_max + 1 + oy), slice(self.x_min + ox, self.x_max + 1 + ox))

    def get_bbox(self): return [self.x_min, self.y_min, self.x_max, self.y_max]

    def offset(self, offset):
        x, y = offset
        return Bounds(self.x_min + x, self.y_min + y, self.x_max + x, self.y_max + y)

    def offset_bounds(self, *offsets):
        if len(offsets) == 1:
            v = offsets[0]
            return Bounds(self.x_min - v, self.y_min - v, self.x_max + v, self.y_max + v)

        raise NotImplementedError('only implemented for 1 parameter for now :(')

    def __str__(self):
        return f'(Bounds) x: {self.x} y: {self.y}; {self.width}x{self.height}'

class Fragment:
    def __init__(self, source_pixels, mask, value, rect):
        self.points = np.array(np.where(mask == value)).T
        self.fragment_value = value
        self.point_count = len(self.points)
        self.bounds = Bounds.from_rect(*rect)
        self.source_pixels = source_pixels

    def compute(self, source_patch=False, patch_mask=False, masked_patch=False):
        if source_patch or masked_patch:
            self.source_patch = self.source_pixels[self.bounds.to_slice()]

        if patch_mask or masked_patch:
            self.patch_mask = np.zeros((self.bounds.height, self.bounds.width), dtype=bool)
            self.patch_mask[self.points[:,0] - self.bounds.y_min, self.points[:,1] - self.bounds.x_min] = True

        if masked_patch:
            self.masked_patch = self.source_patch * self.patch_mask[:,:,np.newaxis]

def detect_fragment(pixels, x, y, mask, value, matching_mask):
    _, _, _, rect = cv2.floodFill(mask, matching_mask, (x, y), value, 255, 255, cv2.FLOODFILL_FIXED_RANGE)
    return Fragment(pixels, mask, value, rect)

def fragment_mask_prepare(mask):
    """Inverts the given array and pads with 1 pixel from each side. Resulting array is of type uint8"""
    return np.pad(np.logical_not(mask), 1, 'constant', constant_values=True).astype(np.uint8)

def detect_fragments(pixels, fragments_mask, **compute_kwargs):
    "returns list of found fragments, list of x and list of y coordinates derived from fragment_mask"
    mask = np.zeros(pixels.shape[:2], dtype=int)
    ys, xs = np.where(fragments_mask)
    fragments_mask = fragment_mask_prepare(fragments_mask)

    fragment_index = 1
    fragments = []
    for x, y in zip(xs, ys):
        if mask[y, x] > 0: continue
        fragment = detect_fragment(pixels, x, y, mask, fragment_index, fragments_mask)
        fragment_index += 1
        fragments.append(fragment)
        if len(compute_kwargs) > 0: fragment.compute(**compute_kwargs)
    
    return fragments, xs, ys, mask

# returns list of lists of fragments (list of groups)
def group_fragments(fragments, radius):
    if len(fragments) == 0: return []
    free_fragments : list = fragments[:]
    groups = []

    while len(free_fragments) > 0:
        current_group = [free_fragments[0]]
        checked_fragments = [False]
        free_fragments.remove(free_fragments[0])

        while not checked_fragments[-1]:
            for i, (group_fragment, is_checked) in enumerate(zip(current_group, checked_fragments)):
                if is_checked: continue
                
                for fragment in free_fragments[:]: # copy list
                    for corner in fragment.bounds.get_corners():
                        if np.linalg.norm(np.array(corner) - group_fragment.bounds.pos) > radius: continue
                        current_group.append(fragment)
                        checked_fragments.append(False)
                        free_fragments.remove(fragment)
                        break

                checked_fragments[i] = True
        groups.append(current_group)

    return groups

def match_color_exact(pixels, color):
    "Simple matcher, matches only pixels that are exactly `color`"
    return np.all(pixels == color, axis=2)

def match_color_fuzzy(pixels, color, max_deviation):
    "Simple fuzzy matcher, matches any pixels that are up to `max_deviation` far from `color`"
    return np.sum(np.abs(pixels - color), axis=2) <= max_deviation

def match_color_tone(pixels, color, min_pixel_value=0, tolerance=0):
    """
    Allows to match any pixels that have the same tone as `color` (e.g. `color` or darker)
    Use `min_pixel_value` to mask all pixels that are too dark. This is the min value of r+g+b of pixel
    Tolerance controls how close the color tone should be to the `color` to match
    """
    pixel_values = np.sum(pixels, axis=2)
    mask = pixel_values >= min_pixel_value
    color_grads = np.mean(pixels.astype(float) / color, axis=2)
    recolor = color_grads.reshape(*color_grads.shape, 1) * color.reshape(1, -1)
    deviations = np.sum(np.abs(pixels - recolor), axis=2)
    return (deviations <= tolerance) * mask

def match_cont_color_values(pixels, color):
    color_grads = np.mean(pixels.astype(float) / color, axis=2)
    recolor = color_grads.reshape(*color_grads.shape, 1) * color.reshape(1, -1)
    return np.sum(np.abs(pixels - recolor), axis=2)

def match_cont_color(pixels, color, tolerance):
    """
    Use when you need to match pixels that have any color you can get by multiplying
    a number from 0 to 1 by the `color` (so any color from black up to `color`)
    """
    return match_cont_color_values(pixels, color) <= tolerance

def detect_blending_pixels(pixels, primary_color, secondary_color, max_blending=1, max_deviation=0):
    """
    Returns mask that matches all the pixels that match any color from primary to secondary (RGB blending)
    max_blending controls actual value of secondary_color: = primary_color * (1 - max_blending) + secondary_color * max_blending
    max_deviation lets match pixels that do not match exactly with the blend
    """
    max_blended_color = np.clip(np.round(primary_color * (1 - max_blending) + secondary_color * max_blending), 0, 255)

    color_direction = max_blended_color - primary_color
    pixel_directions = pixels - primary_color
    pixel_directions_sec = pixels - max_blended_color
    distances = np.linalg.norm(np.cross(pixel_directions, color_direction), axis=2) / np.linalg.norm(color_direction)
    raw_results = distances <= max_deviation # without endpoint checking

    directions_primary = np.dot(pixel_directions, color_direction)
    directions_secondary = np.dot(pixel_directions_sec, -color_direction)
    endpoint_mask = np.logical_and(directions_primary >= 0, directions_secondary >= 0)

    masked_results = np.logical_and(raw_results, endpoint_mask) # everything past endpoints cut off

    close_pixels_1 = np.linalg.norm(pixel_directions, axis=2) <= max_deviation
    close_pixels_2 = np.linalg.norm(pixel_directions_sec, axis=2) <= max_deviation
    close_pixels = np.logical_or(close_pixels_1, close_pixels_2)

    return np.logical_or(masked_results, close_pixels) # restore cut off pixels close to endpoints 

def is_fragment_rectangular(fragment: Fragment, report_fraction=False):
    if fragment.bounds.area == 0: return None
    if not hasattr(fragment, 'patch_mask'): fragment.compute(patch_mask=True)

    rect_mask = np.ones(fragment.bounds.shape, dtype=bool)
    rect_mask[1:-1, 1:-1] = False

    # same as np.sum(rect_mask)
    if rect_mask.shape[0] == 1 or rect_mask.shape[1] == 1:
        mask_ones_count = rect_mask.shape[0] + rect_mask.shape[1] - 1
    else:
        mask_ones_count = 2 * (rect_mask.shape[0] + rect_mask.shape[1] - 2)
    mask_ones_count2 = np.sum(rect_mask)
    if mask_ones_count != mask_ones_count2: raise 'wtf'

    matching_pixels = np.sum(np.logical_and(fragment.patch_mask, rect_mask))
    matching_fraction = matching_pixels / mask_ones_count
 
    if report_fraction: return matching_fraction == 1, matching_fraction
    return matching_fraction == 1

def compute_img_diff_ratio(img1, img2):
    "computes fraction of differing pixels between two images"
    return np.sum(np.any(img1 != img2, axis=2)) / np.prod(img1.shape[:2])
