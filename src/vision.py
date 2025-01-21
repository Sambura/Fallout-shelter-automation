from .debug import result_log

from PIL import ImageGrab, Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import os
import platform

# local terms:
#       bounds: instance of Bounds class
#       bbox: array/tuple of numbers [x0, y0, x1, y1]
#       rect: array/tuple of numbers [x0, y0, width, height]

def pil_screen_capture(bbox):
    if bbox is not None: bbox = [bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1]
    return np.array(ImageGrab.grab(bbox))

_mock_directory = None
_mock_frame_paths = []
_current_mock_path_index = 0

_mock_frames = None
_mock_index = 0
_mock_mode = None

__capture_func = pil_screen_capture
__finish_func = None

def reset_mock_frame_index():
    global _mock_index
    _mock_index = 0

# frames / fixed mode
def set_mock_frames(frames, mode='frames'):
    global _mock_frames, _mock_mode
    _mock_frames = frames
    _mock_mode = mode
    reset_mock_frame_index()

def get_mock_frames(): return _mock_frames 

def load_mock_frames(path, mode='frames'):
    frames = []

    # filenames should have format (\d+)-.*?\.(png|jpg)
    # example: 102-mock-frame.png
    for filename in sorted(os.listdir(path), key=lambda x: int(x.split('-')[0])):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            frames.append(img_array)
    
    print(f'Loaded {len(frames)} mock frames')
    set_mock_frames(frames, mode)

def _is_frame_dir(path):
    return len([x for x in os.listdir(path) if x.endswith(('.jpg', '.png'))]) > 0

def set_mock_frames_path(path, load_frames=True, load_mode='frames'):
    global _mock_directory, _mock_frame_paths, _current_mock_path_index
    has_images = _is_frame_dir(path)
    if has_images:
        _mock_directory = Path(path).parent
        _mock_frame_paths = [path]
    else:
        children = [os.path.join(path, x) for x in os.listdir(path)]
        _mock_frame_paths = [x for x in children if os.path.isdir(x) and _is_frame_dir(x)]

    _current_mock_path_index = 0
    if len(_mock_frame_paths) == 0:
        raise Exception(f'Could not find mock frame directories in {path}')

    if load_frames:
        load_mock_frames(_mock_frame_paths[_current_mock_path_index], load_mode)

def load_next_mock_path(previous=False):
    global _current_mock_path_index
    _current_mock_path_index = (_current_mock_path_index + (-1 if previous else 1)) % len(_mock_frame_paths)
    load_mock_frames(_mock_frame_paths[_current_mock_path_index], _mock_mode)
    return _mock_frame_paths[_current_mock_path_index]

def get_current_mock_path():
    return _mock_frame_paths[_current_mock_path_index]

def crop_image(image, bbox):
    "Crops image. This is what used for cropping screen grabs"
    if bbox is None: return image
    return image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]

def grab_mock_screen(bbox=None):
    global _mock_index
    index = _mock_index
    if _mock_mode != 'fixed': _mock_index += 1
    return crop_image(_mock_frames[index].copy(), bbox)

def init_screen_capture(mode='real', window_title=None, mock_directory=None, _mock_mode='frames', use_native=True):
    """Performs necessary initializations for screen/window capture.

    Parameters:
        mode (str): either 'real' for capturing actual screen or 'mock' for mocking screen capture
        window_title (str): should be specified if 'real' mode is selected - window title to be captured
        mock_directory (str): may be specified if 'mock' mode is selected - path to directory with mock frames
        _mock_mode (str): 'frames' for mock screen capture to return all found frames sequentially, or 'fixed' to return the first frame every time
        use_native (bool): set to False to use generic capture function even if native is available (only for mode == 'real')
    
    Returns: a tuple (func, bool) with a function to be used for screen capture (with optional bbox parameter); and bool indicating whether 
        capture function has native implementation (True) or generic (False) 
    """
    global __capture_func, __finish_func
    native_capture = False

    if mode == 'real':
        if platform.system() == 'Windows' and use_native:
            from .windows_screen_capture import init_window_capture, do_capture, finish_window_capture
            if window_title is None: raise Exception('Should specify window_title for native windows capture')
            if not init_window_capture(window_title):
                result_log('Critical: could not find a window to capture. Fallback to PIL capture')
                __capture_func = pil_screen_capture
                return __capture_func, False

            __finish_func = finish_window_capture
            __capture_func = lambda bbox=None: crop_image(do_capture(), bbox)
            native_capture = True
        else:
            __capture_func = pil_screen_capture

    elif mode == 'mock':
        print('Warning: capture function set to mock')
        
        __capture_func = grab_mock_screen
        native_capture = False

        if mock_directory is not None:
            set_mock_frames_path(mock_directory, load_mode=_mock_mode)
    else:
        raise Exception('mode should be `real` or `mock`')

    return __capture_func, native_capture

def finish_screen_capture():
    """De-initialize screen capture"""
    if __finish_func is not None:
        __finish_func()

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
        "This is more of shrink / expand function then offset"
        if len(offsets) == 1:
            v = offsets[0]
            return Bounds(self.x_min - v, self.y_min - v, self.x_max + v, self.y_max + v)

        raise NotImplementedError('only implemented for 1 parameter for now :(')

    def get_iou(self, bounds):
        "intersection over union of two bounds"
        x1, y1, x2, y2 = self.get_bbox()
        x3, y3, x4, y4 = bounds.get_bbox()

        # Compute intersection area
        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)

        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

        # Compute union area
        rect1_area = (x2 - x1) * (y2 - y1)
        rect2_area = (x4 - x3) * (y4 - y3)
        union_area = rect1_area + rect2_area - intersection_area

        return intersection_area / union_area if union_area != 0 else 0

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
    "returns list of found fragments, list of x and list of y coordinates derived from fragment_mask, and the numeric mask (fragments, xs, ys, mask)"
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

def match_color_grades(pixels, color, min_pixel_value=0, tolerance=0):
    """
    Allows to match any pixels that have the same tone as `color` (e.g. `color` or darker)
    NOTE: this function is primarily suited to match RGB multiples of the specified color, rather than the actual color tone.
    consider using `match_color_hue` if you need a more general mather
    Use `min_pixel_value` to mask all pixels that are too dark. This is the min value of r+g+b of pixel
    Tolerance controls how close the color tone should be to the `color` to match
    """
    pixel_values = np.sum(pixels, axis=2)
    mask = pixel_values >= min_pixel_value
    color_grads = np.mean(pixels.astype(float) / color, axis=2)
    recolor = color_grads.reshape(*color_grads.shape, 1) * color.reshape(1, -1)
    deviations = np.sum(np.abs(pixels - recolor), axis=2)
    return (deviations <= tolerance) * mask

def match_color_grades_std(pixels, color, min_pixel_value=0, tolerance=0):
    """
    Allows to match any pixels that have the same tone as `color` (e.g. `color` or darker)
    Use `min_pixel_value` to mask all pixels that are too dark. This is the min value of r+g+b of pixel
    Tolerance controls how close the color tone should be to the `color` to match
    """
    pixel_values = np.sum(pixels, axis=2)
    mask = pixel_values >= min_pixel_value
    deviations = np.std(pixels.astype(float) / color, axis=2)
    return (deviations <= tolerance / 100) * mask

def _match_color_hue(pixels, target_hue, min_pixel_value=0, tolerance=0):
    pixel_values = np.sum(pixels, axis=2)
    mask = pixel_values >= min_pixel_value
    hues = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)[:,:,0]
    return (np.abs(hues - target_hue) <= tolerance) & mask

def match_color_hue(pixels, color, min_pixel_value=0, tolerance=0):
    target_hue = cv2.cvtColor(color.reshape(1, 1, -1), cv2.COLOR_RGB2HSV)[:,:,0]
    return _match_color_hue(pixels, target_hue, min_pixel_value, tolerance)

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

def get_vertical_scanline_fraction(patch_mask, scanline):
    return np.mean(patch_mask[:, scanline])

def get_horizontal_scanline_fraction(patch_mask, scanline):
    return np.mean(patch_mask[scanline, :])

def _is_fragment_rectangular(patch_mask):
    rect_mask = np.ones(patch_mask.shape, dtype=bool)
    rect_mask[1:-1, 1:-1] = False

    # same as np.sum(rect_mask)
    if rect_mask.shape[0] == 1 or rect_mask.shape[1] == 1:
        mask_ones_count = rect_mask.shape[0] + rect_mask.shape[1] - 1
    else:
        mask_ones_count = 2 * (rect_mask.shape[0] + rect_mask.shape[1] - 2)
    mask_ones_count2 = np.sum(rect_mask) # probably can be removed
    if mask_ones_count != mask_ones_count2: raise 'wtf'

    matching_pixels = np.sum(np.logical_and(patch_mask, rect_mask))
 
    return matching_pixels / mask_ones_count

def is_fragment_rectangular(fragment: Fragment, report_fraction=False):
    if fragment.bounds.area == 0: return None
    if not hasattr(fragment, 'patch_mask'): fragment.compute(patch_mask=True)

    matching_fraction = _is_fragment_rectangular(fragment.patch_mask)
 
    if report_fraction: return matching_fraction == 1, matching_fraction
    return matching_fraction == 1

def strip_rectangular_fragment(fragment: Fragment, min_side_fill_fraction):
    """Tries to make a fragment more rectangular, but certainly is not universal,
    do not expect this to work for your usecase"""
    raise Exception('I changed my mind, not gonna implement this')

def compute_img_diff_ratio(img1, img2):
    "computes fraction of differing pixels between two images"
    return np.sum(np.any(img1 != img2, axis=2)) / np.prod(img1.shape[:2])

def compute_motion_diff(old_frame, current_frame, future_frame, diff_threshold=None):
    "Specify integer threshold to ignore too small changes"
    if diff_threshold is None:
        pre_diff = np.any(old_frame != current_frame, axis=2)
        post_diff = np.any(current_frame != future_frame, axis=2)
    else:
        pre_diff = np.sum(np.abs(old_frame.astype(int) - current_frame), axis=2) >= diff_threshold
        post_diff = np.sum(np.abs(current_frame.astype(int) - future_frame), axis=2) >= diff_threshold
   
    neutral = np.logical_and(pre_diff, post_diff)
    positive = post_diff ^ neutral
    negative = pre_diff ^ neutral

    return np.dstack((negative, positive, neutral))

def box_blur(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    if len(image.shape) == 2: image = image.reshape(*image.shape, 1)
    return np.dstack([convolve2d(image[:, :, x], kernel, mode='same') for x in range(image.shape[2])])
