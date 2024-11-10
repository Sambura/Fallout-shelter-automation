from .debug import progress_log
from .vision import Bounds

import numpy as np

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

def combine_images_alpha_mixing(images):
    result = np.zeros_like(images[0])
    for img in images:
        alpha_mask = img[:, :, 3] != 0
        alpha_values = img[:, :, 3][alpha_mask].astype(float) / 255
        if np.sum(alpha_mask) == 0: continue
        result[alpha_mask] = (1 - alpha_values).reshape(-1, 1) * result[alpha_mask] + alpha_values.reshape(-1, 1) * img[alpha_mask]
    return result
