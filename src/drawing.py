from .util import progress_log
import numpy as np
from .vision import Bounds

def draw_border(pixels: np.ndarray, bounds: Bounds, color: np.ndarray, thickness=1):
    border = np.ones((*bounds.shape, pixels.shape[2]), dtype=pixels.dtype) * color
    h_thickness = min(thickness, (bounds.width - thickness) // 2)
    v_thickness = min(thickness, (bounds.height - thickness) // 2)
    border[v_thickness:-v_thickness, h_thickness:-h_thickness] = 0

    try:
        pixels[bounds.to_slice()] += border
    except:
        progress_log('draw border failed')

def draw_circle(canvas, pos, radius, color):
    ts = np.linspace(0, 2 * np.pi, int(1.1 * 2 * radius * np.pi))
    xs = np.round(np.cos(ts) * radius + pos[0]).astype(int)
    ys = np.round(np.sin(ts) * radius + pos[1]).astype(int)

    for x, y in zip(xs, ys):
        canvas[y, x] = color
