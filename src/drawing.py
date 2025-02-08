from .debug import progress_log
from .vision import Bounds
from .util import rotate_vector_2d

import numpy as np

def draw_border(pixels: np.ndarray, bounds: Bounds, color: np.ndarray, thickness=1, replace=False):
    "Make sure pixels have the same or larger data type than color"
    border = np.ones((*bounds.shape, pixels.shape[2]), dtype=pixels.dtype) * color
    h_thickness = min(thickness, (bounds.width - thickness) // 2)
    v_thickness = min(thickness, (bounds.height - thickness) // 2)
    border[v_thickness:-v_thickness, h_thickness:-h_thickness] = 0
    
    border_bounds = bounds.intersect(Bounds(0, 0, pixels.shape[1] - 1, pixels.shape[0] - 1)).offset(-bounds.low_pos).collapse_negative()

    if replace:
        pixels[bounds.to_slice()] = border[border_bounds.to_slice()]
    else:
        pixels[bounds.to_slice()] += border[border_bounds.to_slice()]

def draw_rect(pixels, bounds, color, replace=False):
    if replace:
        pixels[bounds.to_slice()] = color
    else:
        pixels[bounds.to_slice()] += color

def draw_circle(canvas, pos, radius, color, thickness=1):
    x, y = pos
    yy, xx = np.ogrid[-y:canvas.shape[0] - y, -x:canvas.shape[1] - x]
    r1 = radius - thickness / 2
    r2 = radius + thickness / 2
    mask = (xx**2 + yy**2 <= r2**2) & (xx**2 + yy**2 > r1**2)
    canvas[mask] = color

def draw_disk(canvas, pos, radius, color):
    x, y = pos
    yy, xx = np.ogrid[-y:canvas.shape[0] - y, -x:canvas.shape[1] - x]
    mask = xx**2 + yy**2 <= radius**2
    canvas[mask] = color

def draw_point(canvas, pos, radius, color, outline_color=np.array([0, 0, 0]), outline_width=3):
    draw_disk(canvas, pos, radius + outline_width, outline_color)
    draw_disk(canvas, pos, radius, color)

def draw_line(canvas, pos1, pos2, color, width=2):
    x1, y1 = pos1
    x2, y2 = pos2
    length = np.linalg.norm((x1 - x2, y1 - y2))
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2

    yy, xx = np.indices(canvas.shape[:2])

    # Compute the distance from each point to the line
    # The distance from a point (x, y) to the line Ax + By + C = 0 is given by:
    # |Ax + By + C| / sqrt(A^2 + B^2)
    # For the line through (x1, y1) and (x2, y2), A = y2 - y1, B = x1 - x2, C = x2*y1 - x1*y2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    distance = np.abs(A * xx + B * yy + C) / np.sqrt(A**2 + B**2) # to the line
    disk = (xx - xc)**2 + (yy - yc)**2 <= (length / 2 + width)**2

    mask = (distance <= width / 2) & disk

    canvas[mask] = color

def draw_arrow_line(canvas, pos1, pos2, color, width=2, arrow_abs_length=50, arrow_rel_length=0.1, rel_to_abs=0.8, arrow_angle=25):
    x1, y1 = pos1
    x2, y2 = pos2
    direction = np.array([x2 - x1, y2 - y1])
    length = np.linalg.norm(direction)

    arrow_length = (length * arrow_rel_length) * rel_to_abs + arrow_abs_length * (1 - rel_to_abs)
    
    direction1 = rotate_vector_2d(direction / length, 180 - arrow_angle)
    direction2 = rotate_vector_2d(direction / length, 180 + arrow_angle)
    arrow_pos1 = pos2 + direction1 * arrow_length
    arrow_pos2 = pos2 + direction2 * arrow_length

    draw_line(canvas, pos1, pos2, color, width)
    draw_line(canvas, pos2, arrow_pos1, color, width)
    draw_line(canvas, pos2, arrow_pos2, color, width)

def combine_images_alpha_mixing(images):
    result = np.zeros_like(images[0])
    for img in images:
        alpha_mask = img[:, :, 3] != 0
        alpha_values = img[:, :, 3][alpha_mask].astype(float) / 255
        if np.sum(alpha_mask) == 0: continue
        result[alpha_mask] = (1 - alpha_values).reshape(-1, 1) * result[alpha_mask] + alpha_values.reshape(-1, 1) * img[alpha_mask]
    return result
