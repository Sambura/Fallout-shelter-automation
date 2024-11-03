from PIL import ImageGrab, Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# local terms:
#       bounds: instance of Bounds class
#       bbox: array/tuple of numbers [x0, y0, x1, y1]
#       rect: array/tuple of numbers [x0, y0, width, height]

def grab_real_screen(rect):
    if rect is not None: rect = [rect[0], rect[1], rect[2] + 1, rect[3] + 1]
    return np.array(ImageGrab.grab(rect))

grab_screen = grab_real_screen

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

    def get_corners(self):
        return [(self.x_min, self.y_min), (self.x_min, self.y_max), (self.x_max, self.y_min), (self.x_max, self.y_max)]

    def to_rect(self):
        return [self.x_min, self.y_min, self.width, self.height]

    def from_center(x, y, width, height):
        x_min = x - width // 2
        y_min = y - height // 2
        return Bounds.from_rect(x_min, y_min, width, height)

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

def detect_fragments(pixels, fragments_mask, min_pixel_count=0, **compute_kwargs):
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
        if fragment.point_count < min_pixel_count: continue
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
