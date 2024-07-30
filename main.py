import numpy as np
import cv2
from PIL import ImageGrab, Image
import matplotlib.pyplot as plt
from funcy import print_durations
import tkinter as tk
import threading
import io
from pynput import keyboard, mouse
from time import sleep, perf_counter
import traceback

# constants
undiscoveredRoomColor = np.array([0, 0, 0])
undiscovered_room_color_max_deviation = 15
undiscovered_room_border_mean_color = np.array([0, 220, 0])
undiscovered_room_border_color_max_deviation = 40
min_room_match = 0.7
min_room_size = (50, 50)
healing_color_low = np.array([0, 90, 1])
healing_color_high = np.array([1, 255, 2])
healing_button_min_pixels = 200
stimpak_color_ratio = 0.9
stimpak_min_clean_colors = 0.75
antirad_color_ratio = 0.95
antirad_min_clean_colors = 0.8
levelup_color_ratio = 0
levelup_min_clean_colors = 0.8
color_ratio_tolerance = 0.15
critical_cue_color = np.array([253, 222, 0])
critical_cue_fragment_min_pixels = 25
critical_hit_color = np.array([17, 243, 20])
critical_hit_color_deviation = 10

# configuration
debug_show_progress_visuals = True
debug_show_progress_log = True
debug_show_result_visuals = True
debug_show_result_log = True

def progress_log(str):
    if debug_show_progress_log: print(f'[PDEBUG] {str}')

def result_log(str):
    if debug_show_result_log: print(f'[RDEBUG] {str}')

@print_durations
def grabRealScreen(rect):
    if rect is not None:
        rect = rect[:]
        rect[2] += 1; rect[3] += 1
    return np.array(ImageGrab.grab(rect))

grabScreen = grabRealScreen

def ccw90(v):
    return np.array([v[1], -v[0]], dtype=v.dtype)

def cw90(v): # lmao
    return ccw90(ccw90(ccw90(v)))

def get_room_type(room, loc):
    if loc != 'full': return 'unknown'
    return 'elevator' if room.width < room.height else 'room'

DIRECTIONS = {
    'left': (0, -1),
    'right': (0, 1),
    'up': (-1, 0),
    'down': (1, 0)
}

COLORS = (np.array([plt.get_cmap('rainbow', 10)(i) for i in np.linspace(0, 1, 10)]) * 255).astype(int)

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

    def from_rect(x, y, width, height):
        return Bounds(x, y, x + width - 1, y + height - 1)

    def from_center(x, y, width, height):
        x_min = x - width // 2
        y_min = y - height // 2
        return Bounds.from_rect(x_min, y_min, width, height)

    def get_scaled_from_center(self, scale):
        half_width = int(scale * self.width / 2)
        half_height = int(scale * self.height / 2)
        return Bounds(self.x - half_width, self.y - half_height, self.x + half_width, self.y + half_height)
    
    def contains_bounds(self, bounds):
        if self.x_max < bounds.x_max or bounds.x_min < self.x_min: return False
        if self.y_max < bounds.y_max or bounds.y_min < self.y_min: return False
        return True

    def to_slice(self):
        return (slice(self.y_min, self.y_max + 1), slice(self.x_min, self.x_max + 1))

    def get_bbox(self): return [self.x_min, self.y_min, self.x_max, self.y_max]

    def __str__(self):
        return f'(Bounds) x: {self.x} y: {self.y}; {self.width}x{self.height}'

class Fragment:
    def __init__(self, source_pixels, mask, value, rect):
        self.points = np.array(np.where(mask == value)).T
        self.fragment_value = value
        self.point_count = len(self.points)
        self.bounds = Bounds.from_rect(*rect)
        self.source_patch = source_pixels[self.bounds.to_slice()]
        self.patch_mask = np.zeros(self.source_patch.shape[:2], dtype=bool)
        self.patch_mask[self.points[:,0] - self.bounds.y_min, self.points[:,1] - self.bounds.x_min] = True
        self.masked_patch = self.source_patch * self.patch_mask[:,:,np.newaxis]

def count_matching_pixels(bitmap):
    return np.sum(bitmap)

def tolerant_compare(observed, target, tolerance):
    return observed >= target - tolerance and observed <= target + tolerance

def detect_fragment(pixels, x, y, mask, value, matching_mask):
    _, _, _, rect = cv2.floodFill(mask, matching_mask, (x, y), value, 255, 255, cv2.FLOODFILL_FIXED_RANGE)
    return Fragment(pixels, mask, value, rect)

def fragment_mask_prepare(mask):
    return np.pad(np.logical_not(mask), 1, 'constant', constant_values=True).astype(np.uint8)

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
            if tolerant_compare(color_ratio, target_color_ratio, color_ratio_tolerance) and clean_color_fraction >= min_clean_colors:
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

        low_color_count = count_matching_pixels(np.all(fragment.masked_patch == healing_color_low, axis=2))
        high_color_count = count_matching_pixels(np.all(fragment.masked_patch == healing_color_high, axis=2))
        clean_color_fraction = (low_color_count + high_color_count) / fragment.point_count
        if high_color_count == 0:
            progress_log(f'Skipped fragment due to 0 high color contents')
            continue
        color_ratio = low_color_count / high_color_count

        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = COLORS[fragment_index % len(COLORS)]
        progress_log(f'New fragment: ratio: {color_ratio * 100:0.2f}%; {clean_color_fraction * 100:0.2f}% clean colors')

        if fragment.point_count < healing_button_min_pixels: continue

        checkers = [stimpak_check, levelup_check, antirad_check]
        if abs(color_ratio - stimpak_color_ratio) > abs(color_ratio - antirad_color_ratio):
            checkers = checkers[::-1]

        for checker in checkers: 
            if checker(fragment, color_ratio, clean_color_fraction): break
        else: continue

        if debug_show_result_visuals:
            debug_output[mask == fragment.fragment_value] = COLORS[fragment_index % len(COLORS)]
    
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
            debug_output[mask == fragment.fragment_value] = COLORS[fragment_index % len(COLORS)]

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

screen = grabScreen(None)
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
    screen = grabScreen(rect)
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

def start_diff_visualize():
    print('Starting diff detection...')
    
    image = no_overlay_grab_screen(None)
    sleep(0.05)
    image2 = no_overlay_grab_screen(None)

    do = np.zeros_like(FSA_overlay)
    diff = np.any(image != image2, axis=2)
    do[:, :, 3] = do[:, :, 0] = diff * 255

    update_overlay(do)
    sleep(0.2)

def are_opposite(direction, other_direction):
    if set([direction, other_direction]) == set(['left', 'right']): return True
    if set([direction, other_direction]) == set(['up', 'down']): return True
    return False

def is_vertical(direction: str) -> bool: return direction == 'up' or direction == 'down'
def is_horizontal(direction: str) -> bool: return direction == 'left' or direction == 'right'

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
        if are_opposite(last_direction, direction): pan_duration /= 2

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
        img = grabScreen(grab_bbox)
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
