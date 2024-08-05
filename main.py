from src.game_constants import *
from src.vision import get_debug_color, grab_screen, detect_fragment, fragment_mask_prepare, Bounds, Fragment
from src.util import *

import numpy as np
from PIL import Image
from funcy import print_durations
import tkinter as tk
import threading
import io
from pynput import keyboard, mouse
from time import sleep, perf_counter
import traceback
import os
import datetime

def get_room_type(room, loc):
    if loc != 'full': return 'unknown'
    return 'elevator' if room.width < room.height else 'room'

def get_deviation(img, color):
    return np.sum(np.abs(img - color), axis=2)

# Returns mask that matches all the pixels that match any color from primary to secondary (RGB blending)
# max_blending controls actual value of secondary_color: = primary_color * (1 - max_blending) + secondary_color * max_blending
# max_deviation lets match pixels that do not match blend ting exactly
def detect_blending_pixels(pixels, primary_color, secondary_color, max_blending=1, max_deviation=0):
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

# (room_type, location, bounds)
@print_durations()
def detect_rooms(pixels):
    rooms_detected = []
    mask = np.zeros(pixels.shape[:2], dtype=int)
    debug_output = np.zeros((*pixels.shape[:2], 4), dtype=int)

    fragments_mask = np.sum(np.abs(pixels - undiscovered_room_border_mean_color), axis=2) < undiscovered_room_border_color_max_deviation
    ys, xs = np.where(fragments_mask)
    fragments_mask = fragment_mask_prepare(fragments_mask)

    def analyze_room(fragment):
        fragment.compute(source_patch=True, patch_mask=True)
        bounds = fragment.bounds

        total_border = fragment.point_count
        total_room = np.sum(np.sum(np.abs(fragment.source_patch - undiscoveredRoomColor), axis=2) < undiscovered_room_color_max_deviation)
        match_fraction = (total_border + total_room) / bounds.area
        if match_fraction < min_room_match and total_room > 0:
            progress_log(f'Fragment {fragment.fragment_value} discarded: {("no room found" if total_room <= 0 else f"mismatch [{match_fraction * 100:0.2f}%]")}')
            return False, None

        sides = {
            'left': fragment.patch_mask[bounds.y - bounds.y_min, bounds.x_min - bounds.x_min],
            'top': fragment.patch_mask[bounds.y_max - bounds.y_min, bounds.x - bounds.x_min],
            'right': fragment.patch_mask[bounds.y - bounds.y_min, bounds.x_max - bounds.x_min],
            'bottom': fragment.patch_mask[bounds.y_min - bounds.y_min, bounds.x - bounds.x_min]
        }
        visible_sides = np.sum(np.array(list(sides.values())).astype(int))
        if visible_sides < 2: 
            progress_log(f'Fragment {fragment.fragment_value} discarded: less than 2 visible sides detected')
            return False, None
        location = 'full' if visible_sides == 4 else [k for k, v in sides.items() if not v][0]

        return True, location

    fragments = []
    fragment_index = 1
    for x, y in zip(xs, ys):
        if mask[y, x] > 0: continue
        fragment = detect_fragment(pixels, x, y, mask, fragment_index, fragments_mask)
        fragment_index += 1
        if fragment.bounds.width < min_room_size[0] or fragment.bounds.height < min_room_size[1]: continue

        fragments.append(fragment)

        # progress_log(f'New room fragment: {fragment.bounds}')
        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = 70

    for fragment in fragments:
        valid_room, location = analyze_room(fragment)
        if not valid_room: continue
        room_type = get_room_type(fragment.bounds, location)

        rooms_detected.append((room_type, location, fragment.bounds))

        # result_log(f'Room detected! {fragment.bounds} : {room_type}, location: {location}')
        if debug_show_result_visuals:
            debug_output[fragment.bounds.to_slice()] += np.array([255, 0, 255, 100])

    return rooms_detected, debug_output

# (med_name, bounds)
@print_durations()
def detect_med_buttons(pixels):
    debug_output = np.zeros((*pixels.shape[:2], 4), dtype=int)
    mask = np.zeros(pixels.shape[:2], dtype=int)

    fragments_mask = np.all(np.logical_and(pixels <= healing_color_high, pixels >= healing_color_low), axis=2)
    ys, xs = np.where(fragments_mask)
    fragments_mask = fragment_mask_prepare(fragments_mask)
    detected = []

    def get_icon_checker(name, target_color_ratio, color_ratio_tolerance, min_clean_colors):
        def _func(fragment, color_ratio, clean_color_fraction):
            if abs(color_ratio - target_color_ratio) <= color_ratio_tolerance and clean_color_fraction >= min_clean_colors:
                result_log(f'{name} detected at {fragment.bounds}')
                detected.append((name, fragment.bounds))
                return True
            return False

        return _func

    stimpak_check = get_icon_checker('Stimpak', stimpak_color_ratio, color_ratio_tolerance, stimpak_min_clean_colors)
    antirad_check = get_icon_checker('Antirad', antirad_color_ratio, color_ratio_tolerance, antirad_min_clean_colors)
    levelup_check = get_icon_checker('Level-up', levelup_color_ratio, 0, levelup_min_clean_colors)

    fragment_index = 1
    for x, y in zip(xs, ys):
        if mask[y, x] > 0: continue
        fragment = detect_fragment(pixels, x, y, mask, fragment_index, fragments_mask)
        fragment_index += 1

        fragment.compute(masked_patch=True)
        low_color_count = np.sum(np.all(fragment.masked_patch == healing_color_low, axis=2))
        high_color_count = np.sum(np.all(fragment.masked_patch == healing_color_high, axis=2))
        clean_color_fraction = (low_color_count + high_color_count) / fragment.point_count
        if high_color_count == 0:
            # progress_log(f'Skipped fragment due to 0 high color contents')
            continue
        color_ratio = low_color_count / high_color_count

        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = get_debug_color(fragment_index)
        # progress_log(f'New fragment: ratio: {color_ratio * 100:0.2f}%; {clean_color_fraction * 100:0.2f}% clean colors')

        if fragment.point_count < healing_button_min_pixels: continue

        checkers = [stimpak_check, levelup_check, antirad_check]
        if abs(color_ratio - stimpak_color_ratio) > abs(color_ratio - antirad_color_ratio):
            checkers = checkers[::-1]

        for checker in checkers: 
            if checker(fragment, color_ratio, clean_color_fraction): break
        else: continue

        if debug_show_result_visuals:
            debug_output[mask == fragment.fragment_value] = get_debug_color(fragment_index)
    
    return detected, debug_output

# bounds
@print_durations()
def detect_critical_button(pixels):
    debug_output = np.zeros((*pixels.shape[:2], 4), dtype=int)
    mask = np.zeros(pixels.shape[:2], dtype=int)

    fragments_mask = np.all(pixels == critical_cue_color, axis=2)
    ys, xs = np.where(fragments_mask)
    if len(xs) < critical_cue_fragment_min_pixels * 9: return [], debug_output

    fragments_mask = fragment_mask_prepare(fragments_mask)
    detected = []

    fragment_index = 1
    fragments = []
    for x, y in zip(xs, ys):
        if mask[y, x] > 0: continue
        fragment = detect_fragment(pixels, x, y, mask, fragment_index, fragments_mask)
        fragment_index += 1

        if fragment.point_count < critical_cue_fragment_min_pixels: continue
        fragments.append(fragment)
        # progress_log(f'Detected critical cue fragment: {fragment.point_count} pixel count')
        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = get_debug_color(fragment_index)

    assigned_fragments = [False] * len(fragments)
    index = 0
    while index < len(fragments):
        start = fragments[index] # assume this is the top piece
        index += 1
        if assigned_fragments[index - 1]: continue
        # search for matching central piece
        for i, frag in enumerate(fragments[index:]):
            if frag.bounds.y_max + frag.bounds.height * 2 < start.bounds.y_min: continue
            if start.bounds.x_min >= frag.bounds.x or frag.bounds.x >= start.bounds.x_max: continue
            center, center_index = frag, i
            break
        else: continue
        
        button_bounds = center.bounds.get_scaled_from_center(3)
        fragments_contained = [i for i, frag in enumerate(fragments) if not assigned_fragments[i] and button_bounds.contains_bounds(frag.bounds)]
        if len(fragments_contained) < 9: continue

        result_log(f'Detected critical button: {button_bounds}')
        for i in fragments_contained: assigned_fragments[i] = True
        detected.append(button_bounds)
        
        if debug_show_result_visuals:
            debug_output[button_bounds.to_slice()] = np.array([255, 255, 255, 80])
            debug_output[button_bounds.get_scaled_from_center(0.9).to_slice()] = 0

    return detected, debug_output

@print_durations()
def detect_structural(pixels):
    debug_output = np.zeros((*pixels.shape[:2], 4), dtype=int)
    mask = np.zeros(pixels.shape[:2], dtype=int)

    fragments_mask = np.all(pixels == structural_color, axis=2)
    ys, xs = np.where(fragments_mask)

    fragments_mask = fragment_mask_prepare(fragments_mask)

    fragment_index = 1
    fragments = []
    for x, y in zip(xs, ys):
        if mask[y, x] > 0: continue
        fragment = detect_fragment(pixels, x, y, mask, fragment_index, fragments_mask)
        fragment_index += 1

        if fragment.point_count < critical_cue_fragment_min_pixels: continue
        fragments.append(fragment)
        
        progress_log(f'Detected structural fragment: {fragment.point_count} pixel count')
        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = get_debug_color(fragment_index)

    best_candidate = max(fragments, key=lambda x: x.point_count)

    return best_candidate, debug_output

def is_rectangular(fragment: Fragment, report_fraction=False):
    if not hasattr(fragment, 'patch_mask'): fragment.compute(patch_mask=True)

    rect_mask = np.ones(fragment.bounds.shape, dtype=bool)
    rect_mask[1:-1, 1:-1] = False

    # same as np.sum(rect_mask)
    mask_ones_count = 2 * (rect_mask.shape[0] + rect_mask.shape[1] - 2)
    mask_ones_count = np.sum(rect_mask)
    matching_pixels = np.sum(np.logical_and(fragment.patch_mask, rect_mask))
    matching_fraction = matching_pixels / mask_ones_count
 
    if report_fraction:
        return matching_fraction == 1, matching_fraction

    return matching_fraction == 1

@print_durations()
def detect_enemies(pixels):
    debug_output = np.zeros((*pixels.shape[:2], 4), dtype=int)
    mask = np.zeros(pixels.shape[:2], dtype=int)

    fragments_mask = detect_blending_pixels(pixels, enemy_healthbar_border_color, enemy_healthbar_outline_color, 
        enemy_healthbar_detect_blending, enemy_healthbar_color_max_deviation)
    ys, xs = np.where(fragments_mask)

    fragments_mask = fragment_mask_prepare(fragments_mask)

    fragment_index = 1
    detected = []
    for x, y in zip(xs, ys):
        if mask[y, x] > 0: continue
        fragment = detect_fragment(pixels, x, y, mask, fragment_index, fragments_mask)
        fragment_index += 1

        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = get_debug_color(fragment_index)

        if fragment.point_count < enemy_healthbar_min_border_pixel_count: continue

        if is_rectangular(fragment):
            detected.append(fragment)
            if debug_show_progress_visuals:
                debug_output[mask == fragment.fragment_value] = np.array([50, 255, 20, 255])

    return len(detected) > 0, detected, debug_output

###
### ======================== MAIN ========================
### 

print('Welcome to FSA! Initializing...')

update_interval = 20
camera_pan_deadzone = 0.1
camera_pan_initial_duration = 0.1
crit_wait_count = 4

task_in_progress = False
shortcut_chord_pending = False
terminate_pending = False
script_running = False
current_execution_target = None
mission_ctl = False
show_log = True

def update_overlay(param): pass

def make_async_task(f):
    def _func():
        global task_in_progress
        try:
            f()
        except:
            print('Async task exception:')
            traceback.print_exc()
            update_overlay(FSA_overlay_loading)

        task_in_progress = False

    return _func

screen_img = grab_screen(None)
screen_shape = np.array(screen_img.shape[:2])
screen_shape_xy = screen_shape[::-1]
screen_center_xy = screen_shape_xy // 2
FSA_overlay = np.ones((*screen_shape, 4), dtype=np.uint8) * np.array([255, 0, 0, 255])
FSA_overlay_loading = np.ones((*screen_shape, 4), dtype=np.uint8) * np.array([180, 255, 0, 255])
FSA_overlay[1:-1,1:-1] = 0
FSA_overlay_loading[2:-2,2:-2] = 0

sx, sy = screen_center_xy
sw, sh = (screen_shape_xy * camera_pan_deadzone / 2).astype(int)
camera_deadzone = Bounds(sx - sw, sy - sh, sx + sw, sy + sh)

def terminate():
    global terminate_pending
    terminate_pending = True
    return False

def start_chord():
    global shortcut_chord_pending
    shortcut_chord_pending = True
    progress_log('Waiting for chord completion...')

def chord_handler(key):
    global shortcut_chord_pending, current_execution_target, mission_ctl, show_log
    if script_running and key == keyboard.Key.esc: 
        terminate()

    if shortcut_chord_pending:
        shortcut_chord_pending = False

        if key == keyboard.Key.esc: 
            terminate()
        elif key == keyboard.Key.backspace:
            update_overlay()
            current_execution_target = None
        elif key == keyboard.Key.enter:
            current_execution_target = mission_script

        if not hasattr(key, 'char'): 
            progress_log('Unknown chord...')
            return
        if key.char == '\r': # ??????
            shortcut_chord_pending = True
            return

        if key.char == 'm':
            current_execution_target = start_detect_meds
        elif key.char == 'c':
            current_execution_target = start_detect_crit
        elif key.char == 'r':
            current_execution_target = start_detect_rooms
        elif key.char == 'b':
            current_execution_target = start_battle_detect
        elif key.char == 'g':
            mission_ctl = True
        elif key.char == 'p':
            current_execution_target = start_diff_visualize
        elif key.char == 'l':
            show_log = not show_log
        elif key.char == 'f':
            current_execution_target = start_detect_structural
        elif key.char == 'e':
            current_execution_target = start_detect_enemies
        else: progress_log('Unknown chord...')

root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-transparentcolor','#f0f0f0')
root.attributes("-topmost", True)   

log(
'''Start mission script: Ctrl + F; Enter (Esc to terminate)
Toggle log display: Ctrl + F; L
Shutdown: Ctrl + F; Esc
''')

panel = tk.Label(root)
log_label = tk.Label(root, text='Error text...', background='#000000', foreground='#44dd11', justify='left')
log_label.pack(side='left')

keyboard_input = keyboard.Controller()
mouse_input = mouse.Controller()

launch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f'output/{launch_time}/'
os.makedirs(output_folder)
result_log(f"Output folder created: {output_folder}")
output_index = 0

def update():
    global current_execution_target, task_in_progress

    if terminate_pending: 
        print('Terminating...')
        quit()
    root.after(update_interval, update)
    log_label.config(text=get_current_log())
    if current_execution_target is not None and not task_in_progress:
        task_in_progress = True
        task = threading.Thread(target=make_async_task(current_execution_target), daemon=True)
        task.start()

def show_overlay(): 
    panel.place(relheight=1, relwidth=1); 
    if show_log: log_label.pack(side='left')
def hide_overlay(): panel.place_forget(); log_label.pack_forget()

def update_overlay(image=0):
    image_data = np.clip(image + FSA_overlay, 0, 255).astype(np.uint8)

    with io.BytesIO() as output:
        Image.fromarray(image_data).save(output, format='PNG')
        new_overlay = tk.PhotoImage(data=output.getvalue(), format='png')
        panel.configure(image=new_overlay)
        panel.image = new_overlay
    
    show_overlay()
    
def no_overlay_grab_screen(rect=None):
    global screen_img
    hide_overlay()
    sleep(0.01)
    screen = grab_screen(rect)
    if rect is None: screen_img = screen
    show_overlay()
    return screen

def start_detect_meds():
    result_log('Starting meds detection...')
    meds, do = detect_med_buttons(no_overlay_grab_screen())
    update_overlay(do)

def start_detect_crit():
    result_log('Starting critical cue detection...')
    crits, do = detect_critical_button(no_overlay_grab_screen())
    update_overlay(do)

def start_battle_detect():
    result_log('Starting battle cues detection...')
    crits, do1 = detect_critical_button(no_overlay_grab_screen())
    meds, do2 = detect_med_buttons(no_overlay_grab_screen())
    update_overlay(do1 + do2)

def start_detect_rooms():
    result_log('Starting rooms detection...')
    rooms, do = detect_rooms(no_overlay_grab_screen())
    update_overlay(do)

def start_detect_structural():
    result_log('Starting structural detection...')
    fragments, do = detect_structural(no_overlay_grab_screen())
    update_overlay(do)

def start_detect_enemies():
    result_log('Starting enemies detection...')
    _, fragments, do = detect_enemies(no_overlay_grab_screen())
    update_overlay(do)

diff_image = no_overlay_grab_screen(None)
def start_diff_visualize():
    global diff_image
    result_log('Starting diff detection...')
    image2 = no_overlay_grab_screen(None)

    do = np.zeros_like(FSA_overlay)
    diff = np.any(diff_image != image2, axis=2)
    do[:, :, 3] = do[:, :, 0] = diff * 255
    diff_image = image2

    update_overlay(do)
    sleep(0.2)

def direction_to_screen_center(loc: Bounds, blocked: str):
    if not is_horizontal(blocked):
        if loc.x < camera_deadzone.x_min: return 'left'
        if loc.x > camera_deadzone.x_max: return 'right'

    if not is_vertical(blocked):
        if loc.y < camera_deadzone.y_min: return 'down'
        if loc.y > camera_deadzone.y_max: return 'up'

    return None

CAMERA_PAN_KEYS = { 'left': 'a', 'right': 'd', 'down': 'w', 'up': 's' }

def pan_camera(direction, duration=0.05):
    keyboard_input.press(CAMERA_PAN_KEYS[direction])
    sleep(duration)
    keyboard_input.release(CAMERA_PAN_KEYS[direction])

def get_panning_bbox(bounds, direction):
    if direction == 'left':
        return Bounds(bounds.x_min, bounds.y_min, screen_shape_xy[0] - 1, bounds.y_max).get_bbox()
    elif direction == 'right':
        return Bounds(0, bounds.y_min, bounds.x_max, bounds.y_max).get_bbox()
    elif direction == 'up':
        return Bounds(bounds.x_min, 0, bounds.x_max, bounds.y_max).get_bbox()
    elif direction == 'down':
        return Bounds(bounds.x_min, bounds.y_min, bounds.x_max, screen_shape_xy[1] - 1).get_bbox()

def restore_offset(bounds, clip_bounds):
    x, y, _, _ = clip_bounds
    return Bounds(bounds.x_min + x, bounds.y_min + y, bounds.x_max + x, bounds.y_max + y)

def mouse_click(x, y):
    mouse_input.position = (x, y)
    sleep(0.07)
    mouse_input.press(mouse.Button.left)
    sleep(0.12)
    mouse_input.release(mouse.Button.left)

def log_image(img, postfix, increment_counter=True):
    global output_index
    Image.fromarray(img).save(f'{output_folder}/{output_index}-{postfix}.png')
    if increment_counter: output_index += 1

def debug_log_image(img, postfix, increment_counter=True): log_image(img, postfix, increment_counter)

def compute_img_diff_ratio(img1, img2):
    return np.sum(np.any(img1 != img2, axis=2)) / np.prod(img1.shape[:2])

def navigate_to_room(room_bounds, click=True):
    progress_log('Navigating to the room...')

    pan_duration = camera_pan_initial_duration
    last_direction = None
    last_bounds = Bounds(*(np.ones(4)*(-10000000000))) # idk lol
    blocked_direction = None

    direction = direction_to_screen_center(room_bounds, blocked_direction)
    while direction is not None:
        if are_opposite_directions(last_direction, direction): pan_duration /= 2

        progress_log(f'Panning now: {direction}')    
        pan_camera(direction, pan_duration)

        last_bounds = room_bounds
    
        bbox = get_panning_bbox(room_bounds, direction)
        start_time = perf_counter()
        progress_log(f'Finding room again...')
        while True: # put iteration limit?
            screen_crop = no_overlay_grab_screen(bbox) # how can this be not assigned + the outer loop is terminated??
            debug_log_image(screen_crop, 'navigation-rescan')
            rooms, do = detect_rooms(screen_crop)
            if perf_counter() - start_time > 2:
                progress_log('Panning failed: timeout')
                return False, None, None # timeout
            if len(rooms) > 0: break

        room_bounds = restore_offset(rooms[0][2], bbox)

        # dof = np.zeros_like(FSA_overlay)
        # dof[Bounds(*bbox).to_slice()] = do
        # update_overlay(dof)

        # progress_log(f'New bounds: {room_bounds}')
        if last_bounds.x == room_bounds.x and last_bounds.y == room_bounds.y:
            progress_log(f'Panning blocked... `{direction}`')
            if blocked_direction is not None and blocked_direction != direction: break
            blocked_direction = direction
        
        last_direction = direction
        direction = direction_to_screen_center(room_bounds, blocked_direction)

    progress_log('Panning complete')
    if click:
        for _ in range(5):
            pre_click_room = no_overlay_grab_screen(room_bounds.get_bbox())
            progress_log(f'Clicking: {room_bounds.x, room_bounds.y}')
            mouse_click(room_bounds.x, room_bounds.y)
            sleep(0.5) # click diff wait
            post_click_room = no_overlay_grab_screen(room_bounds.get_bbox())

            click_diff = compute_img_diff_ratio(post_click_room, pre_click_room)

            progress_log(f'Post-click pixel diff: {click_diff*100:0.2f}%')

            log_image(pre_click_room, 'preclick', increment_counter=False)
            log_image(post_click_room, 'postclick', increment_counter=False)
            log_image(np.any(post_click_room != pre_click_room, axis=2), 'clickdiff')

            # room click diff threshold
            if click_diff >= 0.01: return True, room_bounds, pre_click_room
            progress_log(f'Click failed, repeat:')
        
        progress_log('Gave up on clicking...')
        return False, None

    return True, room_bounds, pre_click_room

def make_critical_strike(crit_bounds):
    grab_size = min(crit_bounds.width, crit_bounds.height) // 3
    grab_bounds = Bounds.from_center(crit_bounds.x, crit_bounds.y, grab_size, grab_size)
    grab_bbox = grab_bounds.get_bbox()
    hit_times = []
    debug_grabs = []

    def get_crit_stats():
        ts = perf_counter()
        img = grab_screen(grab_bbox)
        # debug_grabs.append(img)
        yp = np.sum(np.all(img == critical_progress_color, axis=2))
        gp = np.sum(np.sum(np.abs(img - critical_hit_color), axis=2) < critical_hit_color_deviation)
        return ts, yp, gp
    
    progress_log(f'Starting crit scoring: bbox {grab_bbox}')

    # analysis

    min_crit_pixels = 10
    _, last_cue_pixels, last_crit_pixels = get_crit_stats()
    crit_over = last_crit_pixels < min_crit_pixels
    while last_cue_pixels > 10:
        timestamp, cue_pixels, crit_pixels = get_crit_stats()

        if not crit_over:
            if crit_pixels < min_crit_pixels:
                crit_over = True
            else:
                hit_times[-1].append(timestamp)
                continue

        if cue_pixels < last_cue_pixels and crit_pixels > last_crit_pixels:
            hit_times.append([timestamp])
            crit_over = False

        if len(hit_times) >= crit_wait_count: break
        last_cue_pixels, last_crit_pixels = cue_pixels, crit_pixels

    if len(hit_times) < 2:
        progress_log('Crit failed...')
        return

    mean_hit_times = [np.mean(x) for x in hit_times]
    avg_diff = np.mean(np.diff(mean_hit_times))
    next_crit = mean_hit_times[-1]

    for _ in range(10): # limit iteration count to 10
        if next_crit > perf_counter(): break
        next_crit += avg_diff

    while perf_counter() < next_crit: sleep(0.0001)

    mouse_input.press(mouse.Button.left)
    sleep(0.1)
    mouse_input.release(mouse.Button.left)

    progress_log(f'Crit diff info gathered: {[f"{x:0.2f}" for x in np.diff(mean_hit_times)]}, avg: {avg_diff:0.2f}s')

    for img in debug_grabs: debug_log_image(img, 'crit-scan')

def battle_iteration():
    meds, do = detect_med_buttons(screen_img)
    for name, bounds in meds:
        progress_log('Clicking meds...')
        mouse_click(bounds.x, bounds.y)
        sleep(0.05)

    crits, do = detect_critical_button(screen_img)
    for bounds in crits:
        progress_log('Clicking crits...')
        mouse_click(bounds.x, bounds.y)
        sleep(0.3)
        make_critical_strike(bounds)
        sleep(0.8)

def same_rooms(rooms1, rooms2):
    if rooms1 is None and rooms2 is None: return True
    if rooms1 is None: return len(rooms2) == 0
    if rooms2 is None: return len(rooms1) == 0
    if len(rooms1) != len(rooms2): return False
    for room1, room2 in zip(rooms1, rooms2):
        if room1[2].x != room2[2].x: return False
        if room1[2].y != room2[2].y: return False
        if room1[2].width != room2[2].width: return False
        if room1[2].height != room2[2].height: return False
    
    return True

def zoom_out():
    mouse_input.position = screen_center_xy
    init_screen = no_overlay_grab_screen()
    for _ in range(20):
        mouse_input.scroll(0, -1)
    sleep(0.05) # zoom initial delay
    new_screen = no_overlay_grab_screen()
    if np.sum(new_screen != init_screen) / np.prod(new_screen.shape) > 0.5:
        sleep(0.5) # zoom delay

def has_new_room(rooms, last_rooms):
    if len(rooms) == 0: return False
    if len(last_rooms) == 0: return True

    room_height = rooms[0][2].height
    # we have a threshold instead of strict match due to things like screen-shakes
    location_threshold = 0.6 # for each axis separately

    matches = [False] * len(rooms)
    for i, (room_type, location, bounds) in enumerate(rooms):
        # look for the same (similar?) room in last_rooms:
        for _, _, last_bounds in last_rooms:
            if bounds.shape != last_bounds.shape: continue
            if abs(bounds.x - last_bounds.x) > location_threshold * room_height: continue
            if abs(bounds.y - last_bounds.y) > location_threshold * room_height: continue
            matches[i] = True # found match

    return not np.all(matches)

def mission_script():
    global script_running, current_execution_target, mission_ctl

    update_overlay()
    progress_log('>>> Starting mission script')
    progress_log('Looking for starting room...')
    script_running = True
    current_execution_target = None

    last_rooms = None
    while True:
        progress_log('General mission iteration...')

        # new 
        in_battle = False
        new_room_discovered = False
        
        zoom_out() # general iteration on max zoom out for simplicity
        no_overlay_grab_screen() # grab screen for this iteration
        debug_log_image(screen_img, 'iteration-capture')
        battle_iteration() # take care of meds / crits

        rooms, do = detect_rooms(screen_img)

        if len(rooms) == 0: 
            progress_log('No rooms detecting, skipping iteration...')
            continue

        # if same_rooms(last_rooms, rooms) or len(rooms) == 0: continue

        progress_log(f'Found room: {rooms[0][2]}, type: {rooms[0][0]}')
        navigate_succesfull, room_bounds, pre_click_img = navigate_to_room(rooms[0][2])
        if not navigate_succesfull: continue
        progress_log(f'Room navigation successful, waiting for walk complete...')
        last_rooms, _ = detect_rooms(no_overlay_grab_screen())

        # wait until the room is reached
        wait_interval = 0.5 # otherwise it's too fast ?
        while True:
            start_time = perf_counter()
            current_room = no_overlay_grab_screen(room_bounds.get_bbox())
            diff = compute_img_diff_ratio(pre_click_img, current_room)
            debug_log_image(current_room, f'walk-wait-{diff*100:0.2f}-diff')
            progress_log(f'Waiting for walk completion: current diff {100*diff:0.2f}%')

            # min room entered diff threshold
            if diff >= 0.8: break # room reached!
            elapsed = perf_counter() - start_time
            if elapsed < wait_interval: sleep(wait_interval - elapsed)

        # we reached room - wait until enemies are detected OR a new room is detected OR 4 seconds pass
        progress_log(f'Walk complete! Analyzing situation...')
        wait_timeout = 5
        start_time = perf_counter()
        while perf_counter() - start_time <= wait_timeout:
            no_overlay_grab_screen()
            debug_log_image(screen_img, f'wait-iteration')
            have_enemies, _, _ = detect_enemies(screen_img)
            if have_enemies:
                in_battle = True
                progress_log('Enemies detected!')
                break
            
            rooms, _ = detect_rooms(screen_img)
            if has_new_room(rooms, last_rooms):
                new_room_discovered = True
                progress_log('New room detected, proceeding...')
                break

        if not new_room_discovered and not have_enemies:
            progress_log('Wait timeout, aborting iteration')
            continue

        while have_enemies:
            no_overlay_grab_screen() # grab screen for this iteration
            debug_log_image(screen_img, 'battle-iteration-capture')
            battle_iteration() # take care of meds / crits
            have_enemies, enemies, _ = detect_enemies(screen_img)
            progress_log(f'Battle iteration: {len(enemies)} enemies detected')
        
        progress_log('Noting to do, proceeding to next iteration')
        continue

    progress_log('>>> Mission script complete')
    script_running = False

def run_keyboard_listener():
    def for_canonical(f):
        def _func(k):
            f(l.canonical(k))
            if terminate_pending: return False

        return _func
    
    def keyboard_on_press(key):
        try:
            chord_handler(key) 
        except:
            progress_log('Listener thread encountered exception:')
            traceback.print_exc()
        
        for_canonical(fsa_hotkey.press)(key)

    fsa_hotkey = keyboard.HotKey(keyboard.HotKey.parse('<ctrl>+f'), start_chord)
    with keyboard.Listener(
            on_press=keyboard_on_press,
            on_release=for_canonical(fsa_hotkey.release)) as l:
        l.join()

with Image.open("./resources/test_overlay.png") as img:
    update_overlay(np.array(img))
thread = threading.Thread(target=run_keyboard_listener)
thread.start()
root.after(update_interval, update)
root.mainloop()
