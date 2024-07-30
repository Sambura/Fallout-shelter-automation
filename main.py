from src.game_constants import *
from src.vision import get_debug_color, grab_screen, Bounds
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

def get_room_type(room, loc):
    if loc != 'full': return 'unknown'
    return 'elevator' if room.width < room.height else 'room'

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

        progress_log(f'New room fragment: {fragment.bounds}')
        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = 70

    for fragment in fragments:
        valid_room, location = analyze_room(fragment)
        if not valid_room: continue
        room_type = get_room_type(fragment.bounds, location)

        rooms_detected.append((room_type, location, fragment.bounds))

        result_log(f'Room detected! {fragment.bounds} : {room_type}, location: {location}')
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
            progress_log(f'Skipped fragment due to 0 high color contents')
            continue
        color_ratio = low_color_count / high_color_count

        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = get_debug_color(fragment_index)
        progress_log(f'New fragment: ratio: {color_ratio * 100:0.2f}%; {clean_color_fraction * 100:0.2f}% clean colors')

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
        progress_log(f'Detected critical cue fragment: {fragment.point_count} pixel count')
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

def mission_script(): pass

###
### ======================== MAIN ========================
### 

print('Welcome to FSA! Initializing...')

update_interval = 20
camera_pan_deadzone = 0.1
camera_pan_initial_duration = 0.1

task_in_progress = False
shortcut_chord_pending = False
terminate_pending = False
script_running = False
current_execution_target = None
mission_ctl = False

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

screen = grab_screen(None)
screen_shape = np.array(screen.shape[:2])
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

def chord_handler(key):
    global shortcut_chord_pending, current_execution_target, mission_ctl
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

        if not hasattr(key, 'char'): return
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

root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-transparentcolor','#f0f0f0')
root.attributes("-topmost", True)

panel = tk.Label(root)
keyboard_input = keyboard.Controller()
mouse_input = mouse.Controller()

def update():
    global current_execution_target, task_in_progress

    if terminate_pending: 
        print('Terminating...')
        quit()
    root.after(update_interval, update)
    if current_execution_target is not None and not task_in_progress:
        task_in_progress = True
        task = threading.Thread(target=make_async_task(current_execution_target), daemon=True)
        task.start()

def show_overlay(): panel.pack(side="bottom", fill="both", expand="yes")
def hide_overlay(): panel.pack_forget()

def update_overlay(image=0):
    image_data = np.clip(image + FSA_overlay, 0, 255).astype(np.uint8)

    with io.BytesIO() as output:
        Image.fromarray(image_data).save(output, format='PNG')
        new_overlay = tk.PhotoImage(data=output.getvalue(), format='png')
        panel.configure(image=new_overlay)
        panel.image = new_overlay
    
    show_overlay()
    
def no_overlay_grab_screen(rect=None):
    hide_overlay()
    sleep(0.01)
    screen = grab_screen(rect)
    show_overlay()
    return screen

def start_detect_meds():
    print('Starting meds detection...')
    meds, do = detect_med_buttons(no_overlay_grab_screen())
    update_overlay(do)

def start_detect_crit():
    print('Starting critical cue detection...')
    crits, do = detect_critical_button(no_overlay_grab_screen())
    update_overlay(do)

def start_battle_detect():
    print('Starting battle cues detection...')
    crits, do1 = detect_critical_button(no_overlay_grab_screen())
    meds, do2 = detect_med_buttons(no_overlay_grab_screen())
    update_overlay(do1 + do2)

def start_detect_rooms():
    print('Starting rooms detection...')
    rooms, do = detect_rooms(no_overlay_grab_screen())
    update_overlay(do)

diff_image = no_overlay_grab_screen(None)
def start_diff_visualize():
    global diff_image
    print('Starting diff detection...')
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

def navigate_to_room(room_bounds, click=True):
    print('Navigating to the room...')

    pan_duration = camera_pan_initial_duration
    last_direction = None
    last_bounds = Bounds(*(np.ones(4)*(-10000000000))) # idk lol
    blocked_direction = None

    direction = direction_to_screen_center(room_bounds, blocked_direction)
    while direction is not None:
        if are_opposite_directions(last_direction, direction): pan_duration /= 2

        print(f'Panning now: {direction}')    
        pan_camera(direction, pan_duration)

        last_bounds = room_bounds
    
        bbox = get_panning_bbox(room_bounds, direction)
        start_time = perf_counter()
        while True: # put iteration limit?
            rooms, do = detect_rooms(no_overlay_grab_screen(bbox))
            if perf_counter() - start_time > 2:
                print('Panning failed: timeout')
                return False # timeout
            if len(rooms) > 0: break

        room_bounds = restore_offset(rooms[0][2], bbox)

        # dof = np.zeros_like(FSA_overlay)
        # dof[Bounds(*bbox).to_slice()] = do
        # update_overlay(dof)

        print(f'New bounds: {room_bounds}')
        if last_bounds.x == room_bounds.x and last_bounds.y == room_bounds.y:
            print(f'Panning blocked... `{direction}`')
            if blocked_direction is not None and blocked_direction != direction: break
            blocked_direction = direction
        
        last_direction = direction
        direction = direction_to_screen_center(room_bounds, blocked_direction)

    print('Panning complete')
    if click:
        print(f'Clicking: {room_bounds.x, room_bounds.y}')
        mouse_click(room_bounds.x, room_bounds.y)

    return True

def make_critical_strike(crit_bounds):
    grab_size = min(crit_bounds.width, crit_bounds.height) // 3
    grab_bounds = Bounds.from_center(crit_bounds.x, crit_bounds.y, grab_size, grab_size)
    grab_bbox = grab_bounds.get_bbox()

    def get_crit_stats():
        img = grab_screen(grab_bbox)
        yp = np.sum(img == critical_cue_color)
        gp = np.sum(np.abs(img - critical_hit_color) < critical_hit_color_deviation)
        return yp, gp
    
    print('Starting crit scoring...')
    last_cue_pixels, last_crit_pixels = get_crit_stats()
    while last_cue_pixels > 10:
        cue_pixels, crit_pixels = get_crit_stats()
        if cue_pixels < last_cue_pixels and crit_pixels > last_crit_pixels:
            mouse_input.press(mouse.Button.left)
            sleep(0.1)
            mouse_input.release(mouse.Button.left)
            break

        last_cue_pixels, last_crit_pixels = cue_pixels, crit_pixels

def battle_iteration():
    screen = no_overlay_grab_screen()
    meds, do = detect_med_buttons(screen)
    crits, do = detect_critical_button(screen)

    for name, bounds in meds:
        print('Clicking meds...')
        mouse_click(bounds.x, bounds.y)
        sleep(0.05)

    for bounds in crits:
        print('Clicking crits...')
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
    for _ in range(20):
        mouse_input.scroll(0, -1)

def mission_script():
    global script_running, current_execution_target, mission_ctl

    update_overlay()
    print('>>> Starting mission script')
    print('Looking for starting room...')
    script_running = True
    current_execution_target = None

    last_rooms = None
    while True:
        battle_iteration()
        
        zoom_out()
        rooms, do = detect_rooms(no_overlay_grab_screen())

        if same_rooms(last_rooms, rooms) or len(rooms) == 0: continue

        while True:
            print(f'Found room: {rooms[0][2]}, type: {rooms[0][0]}')
            navigate_succesfull = navigate_to_room(rooms[0][2])
            sleep(0.4)
            last_rooms = detect_rooms(no_overlay_grab_screen())[0]
            if navigate_succesfull: break

            while True:
                zoom_out()
                rooms, do = detect_rooms(no_overlay_grab_screen())
                if len(rooms) > 0: break

    print('>>> Mission script complete')
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
            print('Listener thread encountered exception:')
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
