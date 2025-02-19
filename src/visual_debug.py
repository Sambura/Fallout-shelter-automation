from .drawing import combine_images_alpha_mixing

from PIL import ImageGrab
import numpy as np

debug_output = None
debug_frames = []
max_debug_frames = 3
screen_shape = np.array(ImageGrab.grab()).shape[:2] # somehow all ways of doing this in python are stupid

def clear_debug_canvas():
    global debug_output
    debug_output *= 0

def combined_debug_frames():
    return combine_images_alpha_mixing(debug_frames) if len(debug_frames) > 0 else 0

def override_debug_frame_shape(new_shape):
    global screen_shape, debug_frames, debug_output
    screen_shape = new_shape[:2]
    debug_frames = []
    debug_output = None

def create_debug_frame():
    global debug_output, debug_frames
    
    debug_output = np.zeros((*screen_shape, 4), dtype=int)
    debug_frames.append(debug_output)
    while len(debug_frames) > max_debug_frames:
        del debug_frames[0]

def set_max_debug_frames(max_frames):
    global max_debug_frames, debug_frames
    max_debug_frames = max_frames
    while len(debug_frames) > max_debug_frames: del debug_frames[0]

def delete_debug_frame():
    del debug_frames[-1]
    create_debug_frame()

def get_do(): return debug_output

def get_debug_frame(index):
    "0 for current frame, -1 for previous, -2, etc."
    if index > 0: raise Exception('Index should be non-positive')
    if -index > len(debug_frames) - 1: return None
    return debug_frames[index]
