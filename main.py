from src.game_constants import *
from src.vision import *
from src.util import *
from src.drawing import *

import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from funcy import print_durations
import tkinter as tk
import threading
import io
from pynput import keyboard, mouse
from time import sleep, perf_counter
import traceback
from random import randrange

def get_room_type(room, loc):
    if loc != 'full': return 'unknown'
    return 'elevator' if room.width < room.height else 'room'

# (room_type, location, bounds)
@print_durations()
def detect_rooms(pixels, return_fragments=False):
    create_debug_frame()

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

    fragments_mask = match_color_fuzzy(pixels, undiscovered_room_border_mean_color, undiscovered_room_border_color_max_deviation)
    fragments, _, _, mask = detect_fragments(pixels, fragments_mask)

    rooms_detected = []
    detected_fragments = []
    for fragment in fragments:
        if fragment.bounds.are_smaller_than(*min_room_size): continue
        
        # progress_log(f'New room fragment: {fragment.bounds}')
        if debug_show_progress_visuals and np.all(pixels.shape[:2] == screen_shape):
            draw_border(debug_output, fragment.bounds, np.array([255, 0, 0, 255]), thickness=1)

        valid_room, location = analyze_room(fragment)
        if not valid_room: continue
        room_type = get_room_type(fragment.bounds, location)

        rooms_detected.append((room_type, location, fragment.bounds))
        detected_fragments.append(fragment)

        result_log(f'Room detected! {fragment.bounds} : {room_type}, location: {location}')
        if debug_show_result_visuals and np.all(pixels.shape[:2] == screen_shape):
            draw_border(debug_output, fragment.bounds, np.array([0, 0, 180, 255]), thickness=3)

    if return_fragments: return detected_fragments
    return rooms_detected

# (med_name, bounds)
@print_durations()
def detect_med_buttons(pixels):
    create_debug_frame()
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

    fragments_mask = np.all(np.logical_and(pixels <= healing_color_high, pixels >= healing_color_low), axis=2)
    fragments, _, _, mask = detect_fragments(pixels, fragments_mask, masked_patch=True)

    for fragment in fragments:
        low_color_count = np.sum(np.all(fragment.masked_patch == healing_color_low, axis=2))
        high_color_count = np.sum(np.all(fragment.masked_patch == healing_color_high, axis=2))
        clean_color_fraction = (low_color_count + high_color_count) / fragment.point_count
        if high_color_count == 0:
            # progress_log(f'Skipped fragment due to 0 high color contents')
            continue
        color_ratio = low_color_count / high_color_count

        # if debug_show_progress_visuals:
        #     debug_output[mask == fragment.fragment_value] = get_debug_color(fragment.fragment_value)
        # progress_log(f'New fragment: ratio: {color_ratio * 100:0.2f}%; {clean_color_fraction * 100:0.2f}% clean colors')

        if fragment.point_count < healing_button_min_pixels: continue

        checkers = [stimpak_check, levelup_check, antirad_check]
        if abs(color_ratio - stimpak_color_ratio) > abs(color_ratio - antirad_color_ratio):
            checkers = checkers[::-1]

        for checker in checkers: 
            if checker(fragment, color_ratio, clean_color_fraction): break
        else: continue

        if debug_show_result_visuals:
            debug_output[mask == fragment.fragment_value] += get_debug_color(fragment.fragment_value)
    
    return detected

# bounds
@print_durations()
def detect_critical_button(pixels):
    create_debug_frame()
    fragments_mask = match_color_exact(pixels, critical_cue_color)
    if np.sum(fragments_mask) < critical_cue_fragment_min_pixels * 9: return [] # early stop

    fragments, _, _, mask = detect_fragments(pixels, fragments_mask)

    for fragment in fragments[:]: # copy list
        if fragment.point_count < critical_cue_fragment_min_pixels:
            fragments.remove(fragment); continue
        
        # progress_log(f'Detected critical cue fragment: {fragment.point_count} pixel count')
        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] += get_debug_color(fragment.fragment_value)

    detected = []
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
            debug_output[button_bounds.to_slice()] += np.array([255, 255, 255, 80])
            debug_output[button_bounds.get_scaled_from_center(0.9).to_slice()] = 0

    return detected

# [?] Detects the largest fragment that has structural color (black)
# structural fragment, obscured_directions, debug_output
@print_durations()
def detect_structural(pixels):
    create_debug_frame()
    fragments_mask = match_color_exact(pixels, structural_color)
    fragments, _, _, mask = detect_fragments(pixels, fragments_mask)

    for fragment in fragments[:]: # copy list
        if fragment.point_count < critical_cue_fragment_min_pixels: 
            fragments.remove(fragment); continue
        
        # progress_log(f'Detected structural fragment: {fragment.point_count} pixel count')
        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] = get_debug_color(fragment.fragment_value)

    if len(fragments) == 0: return None, None

    best_candidate = max(fragments, key=lambda x: x.point_count)
    obscured_directions = []
    best_candidate.compute(patch_mask=True)
    if best_candidate.bounds.y_min == 0: obscured_directions.append('down')
    if best_candidate.bounds.y_max == pixels.shape[0] - 1: obscured_directions.append('up')
    if best_candidate.bounds.x_min == 0: obscured_directions.append('left')
    if best_candidate.bounds.x_max == pixels.shape[1] - 1: obscured_directions.append('right')

    return best_candidate, obscured_directions

@print_durations()
def detect_enemies(pixels):
    create_debug_frame()
    fragments_mask = detect_blending_pixels(pixels, enemy_healthbar_border_color, enemy_healthbar_outline_color, 
        enemy_healthbar_detect_blending, enemy_healthbar_color_max_deviation)

    fragments, _, _, mask = detect_fragments(pixels, fragments_mask)
    detected = []

    for fragment in fragments:
        if debug_show_progress_visuals:
            debug_output[mask == fragment.fragment_value] += get_debug_color(fragment.fragment_value)

        if fragment.point_count < enemy_healthbar_min_border_pixel_count: continue

        if is_fragment_rectangular(fragment):
            detected.append(fragment)
            if debug_show_progress_visuals:
                debug_output[mask == fragment.fragment_value] += np.array([50, 255, 20, 255])

    return len(detected) > 0, detected

# [?] Accepts a structural fragment (from detect_structural) and returns the list of rooms
# in this structural (as an array of Bounds)
# ended up not using this :( 
# primary problem is that sometimes some text appears above characters
# and messes up with structural detection
@print_durations()
def detect_structural_rooms(structural: Fragment):
    if structural is None: return []
    create_debug_frame()
    structural.compute(patch_mask=True)
    structure = structural.patch_mask # patch_mask is true everywhere where there is structural 
    hole_pixels = np.logical_not(structure)

    fragments, _, _, _ = detect_fragments(structure, hole_pixels)
    rooms = []

    for fragment in fragments:
        if fragment.bounds.area < struct_min_room_pixels: continue
        # these just remove some of the edge cases to reduce number of false positive room detections
        if fragment.bounds.x_min == 0 and structural.bounds.x_min != 0: continue
        if fragment.bounds.y_min == 0 and structural.bounds.y_min != 0: continue
        if fragment.bounds.x_max == structural.bounds.width - 1 and structural.bounds.x_max != screen_shape[1] - 1: continue
        if fragment.bounds.y_max == structural.bounds.height - 1 and structural.bounds.y_max != screen_shape[0] - 1: continue

        clearance_fraction = np.sum(hole_pixels[fragment.bounds.to_slice()]) / fragment.bounds.area
        if clearance_fraction < struct_min_room_clearance_fraction: continue
        _, rect_fraction = is_fragment_rectangular(fragment, report_fraction=True)
        if rect_fraction < struct_min_room_border_fraction: continue

        rooms.append(fragment.bounds.offset(structural.bounds.low_pos))
        
        # progress_log(f'Detected structural room ({fragment.bounds}) : border fraction {rect_fraction*100:0.2f}%, clearance: {clearance_fraction*100:0.2f}%')
        if debug_show_progress_visuals:
            draw_border(debug_output, rooms[-1], get_debug_color(fragment.fragment_value), thickness=2)

    return rooms

@print_durations()
def detect_dialogue_buttons(pixels):
    create_debug_frame()
    fragments_mask = match_color_fuzzy(pixels, primary_dialogue_button_color, dialogue_button_color_max_deviation)
    fragments, _, _, _ = detect_fragments(pixels, fragments_mask)
    detected = []

    for fragment in fragments:
        rel_width, rel_height = fragment.bounds.width / screen_shape_xy[0], fragment.bounds.height / screen_shape_xy[1]
        if rel_width < min_dialogue_button_rel_size[0] or rel_height < min_dialogue_button_rel_size[1]: continue
        rel_x = fragment.bounds.x / screen_shape_xy[0]
        if abs(rel_x - 0.5) > dialogue_button_max_center_deviation_rel: continue
        _, rect_fraction = is_fragment_rectangular(fragment, report_fraction=True)
        if rect_fraction < min_dialogue_button_border_fraction: continue

        # progress_log(f'Detected dialogue button: {fragment.bounds}')
        if debug_show_progress_visuals:
            draw_border(debug_output, fragment.bounds, np.array([200, 25, 10, 255], dtype=np.uint8), 4)

        detected.append(fragment)

    # just in case
    detected = sorted(detected, key=lambda x: x.bounds.y)

    return len(detected) > 0, detected

@print_durations()
def detect_loot(pixels):
    create_debug_frame()
    structural, _ = detect_structural(pixels)
    delete_debug_frame()
    rooms = detect_structural_rooms(structural)
    central_room = next((x for x in rooms if x.contains_point(*screen_center_xy)), None)
    if central_room is None:
        progress_log(f'!Critical: failed to detect central room! Fallback to whole screen')
        central_room = Bounds(0, 0, screen_shape_xy[0] - 1, screen_shape_xy[1] - 1)
    else:
        progress_log(f'Detected main room for loot collection: {central_room}')
        central_room = Bounds(0, 0, screen_shape_xy[0] - 1, screen_shape_xy[1] - 1)
        
    search_iterations = 24
    fps = 8
    progress_log(f'Loot collection start: {search_iterations} frames at {fps} FPS')
    
    # collect frame data
    frames = []
    for x in range(search_iterations):
        start_time = perf_counter()
        frames.append(no_overlay_grab_screen(central_room.get_bbox()))
        sleep(max(0, 1 / fps - (perf_counter() - start_time)))

    for i in range(search_iterations):
        debug_log_image(frames[i], f'loot-frame-{i}')

    delete_debug_frame()
    progress_log(f'Frames collected, computing...')

    color_diffs = [np.abs(x.astype(int) - y).astype(np.uint8) for x, y in zip(frames[:-1], frames[1:])] # colorful diffs
    any_changes = np.sum([np.any(x > 0, axis=2) for x in color_diffs], axis=0) > 0 # mask where at least one pixel changed
    # not `scores` anymore but ok
    scores = [match_color_tone(x, loot_particles_base_color, min_pixel_value=100, tolerance=10) for x in color_diffs] # scores for loot detection

    # debug snippet
    # debug_output[:, :, 3] = 255
    # for i in range(len(color_diffs)):
    #     progress_log(f'showing diff #{i + 1}')
    #     debug_output[:, :, :3] = color_diffs[i]
    #     sleep(2.5)
    #     progress_log(f'showing scores')
    #     debug_output[:, :, 0] = scores[i] * 255
    #     debug_output[:, :, 1] = scores[i] * 255
    #     debug_output[:, :, 2] = 0
    #     sleep(2.5)
    # debug snippet

    matches = [match_cont_color(x, corpse_particles_base_color, corpse_color_detection_threshold) for x in color_diffs] # matches for corpse detection
    # pos_matches = np.sum(scores, axis=0) * any_changes > 0 # pixels that match loot
    pos_matches = np.sum(scores, axis=0) * any_changes > 0 # pixels that match loot
    neg_matches = any_changes ^ pos_matches # pixels that don't match loot
    s_scores = pos_matches.astype(int) - neg_matches.astype(int) # combination of last 2
    for x, y in zip(matches, color_diffs): x[np.all(y == 0, axis=2)] = 0 # filter matches (?)
    cum_match = np.sum(matches, axis=0) > 0 # cumulative matches
    # debug_output[:,:,0] = 255 * neg_matches
    # debug_output[:,:,2] = 255 * pos_matches
    # debug_output[:,:,3] = 255 * any_changes

    progress_log(f'Convolving...')
    kernel_size = 11
    threshold = 0.75
    smoothed = convolve2d(cum_match.astype(int), np.ones((kernel_size, kernel_size)), mode='same')
    filtered_mask = smoothed > kernel_size * kernel_size * threshold

    progress_log(f'Detecting fragments...')

    loot_frags, _, _, loot_frag_mask = detect_fragments(s_scores.reshape(*s_scores.shape, 1), s_scores != 0, masked_patch=True)
    corpse_frags, _, _, corpse_frag_mask = detect_fragments(filtered_mask, filtered_mask)

    for i in range(len(corpse_frags))[::-1]:
        if corpse_frags[i].point_count < corpse_min_fragment_pixels:
            progress_log(f'Discarded corpse candidate: not enouch pixels: {corpse_frags[i].bounds}')
            del corpse_frags[i]

    for fragment in loot_frags[:]: # copy list
        fragment_val = np.mean(fragment.masked_patch)
        if fragment_val < 0: 
            loot_frags.remove(fragment)
            continue
        # progress_log(f'Fragment: {fragment.bounds}, value: {fragment_val}')

        draw_border(debug_output, fragment.bounds, get_debug_color(randrange(10)), 1)

    groups = group_fragments(loot_frags, general_loot_particle_search_radius)
    group_bounds = [Bounds.from_points(*[y for x in group for y in x.bounds.get_corners()]) for group in groups]

    progress_log(f'Grouped fragments: {len(groups)} groups')

    filtered_groups = []
    for i, group in enumerate(groups):
        if group_bounds[i].are_smaller_than(loot_min_size):
            progress_log(f'Discarding too small group: {group_bounds[i]}')
            continue
        else:
            filtered_groups.append(group_bounds[i].offset(central_room.low_pos))
        
        # for fragment in group:
        #     debug_output[fragment.bounds.to_slice(offset=central_room.low_pos)] = get_debug_color(i)

        draw_border(debug_output, group_bounds[i].offset(central_room.low_pos), np.array([28, 125, 199, 255]), 3)

    for fragment in corpse_frags:
        if fragment.bounds.are_smaller_than(corpse_min_fragment_size):
            progress_log(f'Discarding corpse candidate: {fragment.bounds}')
            continue
        else:
            filtered_groups.append(fragment.bounds.offset(central_room.low_pos))

        draw_border(debug_output, fragment.bounds.offset(central_room.low_pos), np.array([255, 0, 0, 255]), 3)

    return filtered_groups

def radius_probe_debug(fragments_mask, radius=40):
    create_debug_frame()
    fragments, _, _ = detect_fragments(fragments_mask, fragments_mask, patch_mask=True)

    # for fragment in fragments:
    #     draw_circle(debug_output, fragment.bounds.pos, radius, np.array([255,255,255,255]))
    
    if len(fragments) == 0: return
    start_fragment = fragments[randrange(len(fragments))]
    # debug_output[start_fragment.bounds.to_slice()][start_fragment.patch_mask] = np.array([255, 0, 0, 255])
    draw_circle(debug_output, start_fragment.bounds.pos, radius, np.array([255,0,0,255]))
    connected_fragments = [start_fragment]
    connect = True

    while connect:
        connect = False
        for fragment in fragments:
            if fragment in connected_fragments: continue

            break_flag = False
            for base_frag in connected_fragments:
                for corner in fragment.bounds.get_corners():
                    if np.linalg.norm(np.array(corner) - base_frag.bounds.pos) <= radius:
                        connect = True
                        connected_fragments.append(fragment)
                        draw_circle(debug_output, fragment.bounds.pos, radius, np.array([0,255,0,255]))
                        break_flag = True
                        break
                if break_flag: break

###
### ======================== MAIN ========================
### 

print('Welcome to FSA! Initializing...')

update_interval = 20
camera_pan_deadzone = 0.1
camera_pan_initial_duration = 0.1
crit_wait_count = 4
dialogue_mode = 'manual' # random / manual

task_in_progress = False
shortcut_chord_pending = False
terminate_pending = False
script_running = False
current_execution_target = None
mission_ctl = False
show_log = True
last_key_pressed = None
grab_screen, native_grab_screen = init_screen_capture(mode='real', window_title='Fallout Shelter')

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
overlay_on = True

debug_output = None
debug_frames = []
max_debug_frames = 3

def clear_debug_canvas():
    # remove function from places?
    global debug_output
    debug_output *= 0

def combine_images_alpha_mixing(images):
    result = np.zeros_like(images[0])
    for img in images:
        alpha_mask = img[:, :, 3] != 0
        alpha_values = img[:, :, 3][alpha_mask].astype(float) / 255
        if np.sum(alpha_mask) == 0: continue
        result[alpha_mask] = (1 - alpha_values).reshape(-1, 1) * result[alpha_mask] + alpha_values.reshape(-1, 1) * img[alpha_mask]
    return result

def display_debug():
    update_overlay(combine_images_alpha_mixing(debug_frames), autoshow=False)

def create_debug_frame():
    global debug_output, debug_frames
    
    debug_output = np.zeros((*screen_shape[:2], 4), dtype=int)
    debug_frames.append(debug_output)
    while len(debug_frames) > max_debug_frames:
        del debug_frames[0]

def delete_debug_frame():
    del debug_frames[-1]
    create_debug_frame()

create_debug_frame()

sx, sy = screen_center_xy
sw, sh = (screen_shape_xy * camera_pan_deadzone / 2).astype(int)
camera_deadzone = Bounds(sx - sw, sy - sh, sx + sw, sy + sh)

def terminate():
    global terminate_pending
    terminate_pending = True
    return False

def hide_overlay(): global overlay_on; panel.place_forget(); log_label.pack_forget(); overlay_on = False

def no_overlay_grab_screen(bbox=None):
    global screen_img
    hide_overlay()
    sleep(0.06) # last tried: 0.05, sometimes overlay still gets captured
    screen = grab_screen(bbox)
    if bbox is None: screen_img = screen
    show_overlay()
    return screen

def no_overlay_grab_screen_native(bbox=None):
    global screen_img
    screen = grab_screen(bbox)
    if bbox is None: screen_img = screen
    return screen

if native_grab_screen:
    no_overlay_grab_screen = no_overlay_grab_screen_native
    result_log('Native screen capture succesfully initialized')

# ???? do something with this
if False: # mock grab screen
    load_mock_frames('_refs/general_loot_frames2/')
    set_grab_screen(mode='mock')
    max_debug_frames = 100
    create_debug_frame()
    debug_output[:, :, :3] = get_mock_frames()[0]
    debug_output[:, :, 3] = 255
    create_debug_frame()

def start_chord():
    global shortcut_chord_pending
    shortcut_chord_pending = True
    progress_log('Waiting for chord completion...')

def start_detect_meds():
    clear_debug_canvas()
    meds = detect_med_buttons(no_overlay_grab_screen())
    display_debug()

def start_detect_crit():
    clear_debug_canvas()
    crits = detect_critical_button(no_overlay_grab_screen())
    display_debug()

def start_battle_detect():
    clear_debug_canvas()
    detect_critical_button(no_overlay_grab_screen())
    detect_med_buttons(no_overlay_grab_screen())
    update_overlay(debug_output)

def start_detect_rooms():
    clear_debug_canvas()
    rooms = detect_rooms(no_overlay_grab_screen())
    display_debug()

def start_detect_structural():
    clear_debug_canvas()
    fragments, _ = detect_structural(no_overlay_grab_screen())
    display_debug()

def start_detect_structural_rooms():
    structural, _ = detect_structural(no_overlay_grab_screen())
    clear_debug_canvas()
    detect_structural_rooms(structural)
    display_debug()

def start_detect_enemies():
    clear_debug_canvas()
    _, fragments = detect_enemies(no_overlay_grab_screen())
    display_debug()

def start_detect_dialogue_buttons():
    clear_debug_canvas()
    detect_dialogue_buttons(no_overlay_grab_screen())
    display_debug()

def start_detect_loot():
    clear_debug_canvas()
    global current_execution_target
    current_execution_target = None
    detect_loot(no_overlay_grab_screen())
    display_debug()

def start_radius_debug():
    global current_execution_target
    current_execution_target = None
    radius_probe_debug()
    update_overlay(debug_output)

def start_diff_visualize():
    global diff_image
    image2 = no_overlay_grab_screen(None)

    output = np.zeros_like(FSA_overlay)
    diff = np.any(diff_image != image2, axis=2)
    output[:, :, 3] = output[:, :, 0] = diff * 255
    diff_image = image2

    update_overlay(output)
    sleep(0.2)

"characters on keyboard to functions to start and messages to print"
execution_target_map = {
    'm': (start_detect_meds, 'Starting meds detection...'),
    'c': (start_detect_crit, 'Starting critical cue detection...'),
    'r': (start_detect_rooms, 'Starting rooms detection...'),
    'b': (start_battle_detect, 'Starting battle cues detection...'),
    'p': (start_diff_visualize, 'Starting diff detection...'),
    'f': (start_detect_structural, 'Starting structural detection...'),
    'e': (start_detect_enemies, 'Starting enemies detection...'),
    'h': (start_detect_structural_rooms, 'Starting structural room detection...'),
    'n': (start_detect_dialogue_buttons, 'Starting dialogue button detection...'),
    '/': (start_detect_loot, 'Starting loot detection...'),
    'd': (start_radius_debug, 'Radius visualization..')
}

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
        
        if key.char in execution_target_map:
            current_execution_target, message = execution_target_map[key.char]
            if message is not None: result_log(message)
        elif key.char == 'g':
            mission_ctl = True
        elif key.char == 'l':
            show_log = not show_log
            if overlay_on:
                if show_log:
                    log_label.pack(side='left')
                else:
                    log_label.pack_forget()
        else: progress_log('Unknown chord...')

init_output_directory()
root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-transparentcolor','#f0f0f0')
root.attributes("-topmost", True)   

log(
'''Fallout Shelter Automation: v0.4.2
Start mission script: Ctrl + F; Enter (Esc to terminate)
Toggle log display: Ctrl + F; L
Shutdown: Ctrl + F; Esc
''')

panel = tk.Label(root)
log_label = tk.Label(root, text='Error text...', background='#000000', foreground='#44dd11', justify='left')
log_label.pack(side='left')

keyboard_input = keyboard.Controller()
mouse_input = mouse.Controller()
CAMERA_PAN_KEYS = { 'left': 'a', 'right': 'd', 'down': 'w', 'up': 's' }

fcounter = 0

def update():
    global current_execution_target, task_in_progress, fcounter

    if terminate_pending: 
        print('Terminating...')
        finish_screen_capture()
        quit()
    
    root.after(update_interval, update)
    fcounter += 1
    if fcounter % 25 == 0: display_debug()
    
    log_label.config(text=get_current_log())
    if current_execution_target is not None and not task_in_progress:
        task_in_progress = True
        task = threading.Thread(target=make_async_task(current_execution_target), daemon=True)
        task.start()

def show_overlay():
    global overlay_on
    panel.place(relheight=1, relwidth=1)
    if show_log: log_label.pack(side='left')
    overlay_on = True

def update_overlay(image=0, autoshow=True):
    image_data = np.clip(image + FSA_overlay, 0, 255).astype(np.uint8)

    with io.BytesIO() as output:
        Image.fromarray(image_data).save(output, format='PNG')
        new_overlay = tk.PhotoImage(data=output.getvalue(), format='png')
        panel.configure(image=new_overlay)
        panel.image = new_overlay
    
    if autoshow: show_overlay()

diff_image = no_overlay_grab_screen(None)

def direction_to_screen_center(loc: Bounds, blocked: str):
    if not is_horizontal(blocked):
        if loc.x < camera_deadzone.x_min: return 'left'
        if loc.x > camera_deadzone.x_max: return 'right'

    if not is_vertical(blocked):
        if loc.y < camera_deadzone.y_min: return 'down'
        if loc.y > camera_deadzone.y_max: return 'up'

    return None

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
    restore_overlay = overlay_on
    hide_overlay()
    mouse_input.position = (x, y)
    sleep(0.07)
    mouse_input.press(mouse.Button.left)
    sleep(0.12)
    mouse_input.release(mouse.Button.left)
    if restore_overlay: show_overlay()

def dialogue_random_handler(buttons):
    button_index = randrange(len(buttons))

    mouse_click(*buttons[button_index].bounds.pos)
    progress_log(f'Clicking button #{button_index + 1}...')

def dialogue_manual_handler(buttons):
    global last_key_pressed
    progress_log(f'Waiting input: 1-{len(buttons)}...')

    # technically it is *kindof* a race condition, but we don't care right?
    last_key_pressed = None
    while last_key_pressed is None or not hasattr(last_key_pressed, 'char') or not ('1' <= last_key_pressed.char <= '9'):
        sleep(0.1)

    button_index = int(last_key_pressed.char) - 1

    progress_log(f'Clicking button #{button_index + 1}...')
    mouse_click(*buttons[button_index].bounds.pos)
    
dialogue_handlers = {
    'random': dialogue_random_handler,
    'manual': dialogue_manual_handler
}

def is_room_click(diff: np.ndarray):
    border_width = int(room_click_diff_border_width * diff.shape[0])
    mask = np.ones_like(diff, dtype=bool)
    mask[border_width:-border_width, border_width:-border_width] = False
    hh, hw = mask.shape[0] // 2, mask.shape[1] // 2
    border = np.logical_and(diff, mask)

    # check there are pixels in border area in each of the 4 quadrants
    return np.any(border[:hh, :hw]) and np.any(border[:hh, hw:]) and np.any(border[hh:, :hw]) and np.any(border[hh:, hw:])

def navigate_to_room(room_bounds: Bounds, click=True):
    progress_log(f'Navigating to the room... : {room_bounds}')

    pan_duration = camera_pan_initial_duration
    last_direction = None
    last_bounds = Bounds(*(np.ones(4)*(-10000000000))) # idk lol
    blocked_direction = None

    room_anchors = room_bounds.get_corners(get_8=True)
    furthest_anchor_i = np.argmax([np.linalg.norm(x) for x in np.array(room_anchors) - screen_center_xy])
    # cursor at the edge of the screen will pan camera, make sure to clip position
    zoom_point = np.clip(room_anchors[furthest_anchor_i], [40, 40], screen_shape_xy - 41)
    zoom_in(*zoom_point)
    progress_log('zoomed in, rediscovering room...')
    sleep(1.5)
    no_overlay_grab_screen() # will need to find room again after zoom-in
    rooms = detect_rooms(screen_img)
    if len(rooms) == 0:
        progress_log('Navigation: target lost')
        return False, None, None
    
    # find room closest to where we had cursor for zooming in
    room_bounds = rooms[np.argmin([np.linalg.norm(zoom_point - x.pos) for z, y, x in rooms])][2]

    direction = direction_to_screen_center(room_bounds, blocked_direction)
    while direction is not None:
        if are_opposite_directions(last_direction, direction): pan_duration /= 2

        progress_log(f'Panning now: {direction}')    
        pan_camera(direction, pan_duration)

        last_bounds = room_bounds
    
        bbox = get_panning_bbox(room_bounds, direction)
        start_time = perf_counter()
        progress_log(f'Finding room again...')
        sleep(0.1) # post-pan delay
        while True: # put iteration limit?
            screen_crop = no_overlay_grab_screen(bbox) # how can this be not assigned + the outer loop is terminated??
            debug_log_image(screen_crop, 'navigation-rescan')
            rooms = detect_rooms(screen_crop)
            if perf_counter() - start_time > 2:
                progress_log('Panning failed: timeout')
                return False, None, None # timeout
            if len(rooms) > 0: break

        room_bounds = restore_offset(rooms[0][2], bbox)

        # progress_log(f'New bounds: {room_bounds}')
        if last_bounds.x == room_bounds.x and last_bounds.y == room_bounds.y:
            progress_log(f'Panning blocked... `{direction}`')
            if blocked_direction is not None and blocked_direction != direction: break
            blocked_direction = direction
        
        last_direction = direction
        direction = direction_to_screen_center(room_bounds, blocked_direction)

    progress_log('Panning complete')
    if click:
        sleep(0.1) # post panning delay (camera seems to be panning a bit after a key is released)
        no_overlay_grab_screen()
        debug_log_image(screen_img, 'navigation-end-grab')
        rooms = detect_rooms(screen_img)
        filtered_rooms = [x for _, _, x in rooms if x.shape == room_bounds.shape]
        if len(filtered_rooms) == 0:
            progress_log('Navigation: target lost')
            return False, None, None

        room_bounds = min(filtered_rooms, key=lambda x: np.linalg.norm(x.pos - room_bounds.pos))

        for _ in range(5):
            pre_click_room = no_overlay_grab_screen(room_bounds.get_bbox())

            progress_log(f'Clicking: {room_bounds.x, room_bounds.y}')
            mouse_click(room_bounds.x, room_bounds.y)
            sleep(0.5) # click diff wait
            post_click_room = no_overlay_grab_screen(room_bounds.get_bbox())

            click_diff = compute_img_diff_ratio(post_click_room, pre_click_room)

            progress_log(f'Post-click pixel diff: {click_diff*100:0.2f}%')

            diff_mask = np.any(post_click_room != pre_click_room, axis=2)
            log_image(pre_click_room, 'preclick', increment_counter=False)
            log_image(post_click_room, 'postclick', increment_counter=False)
            log_image(diff_mask, 'clickdiff')

            if True:
                temp_test_rooms = detect_rooms(pre_click_room, return_fragments=True)
                if len(temp_test_rooms) != 1 or not is_fragment_rectangular(temp_test_rooms[0]):
                    progress_log('ALERT: room_focus assertion failed')
                    log_image(pre_click_room, 'assertion-failed', increment_counter=False)

            # room click diff threshold
            if click_diff >= 0.01: 
                if is_room_click(diff_mask):
                    return True, room_bounds, pre_click_room
                progress_log(':: Click diff detected, flagged as false positive')
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
                if len(hit_times) > 0: hit_times[-1].append(timestamp)
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
    diffs = np.diff(mean_hit_times)
    avg_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    next_crit = mean_hit_times[-1]
    crit_offset = avg_diff

    # if std_diff > 0.04:
    #     progress_log('Warning: diffs deviation is too high, stripping data')
    #     crit_offset = np.min(diffs)

    crit_offset = np.min(diffs) # experimental approach

    for _ in range(10): # limit iteration count to 10
        if next_crit > perf_counter(): break
        next_crit += avg_diff

    while perf_counter() < next_crit: sleep(0.0001)

    mouse_input.press(mouse.Button.left)
    sleep(0.1)
    mouse_input.release(mouse.Button.left)

    progress_log(f'Crit diff info gathered: {[f"{x:0.2f}" for x in diffs]}, avg: {avg_diff:0.2f}s, std: {std_diff:0.4f}')

    for img in debug_grabs: debug_log_image(img, 'crit-scan')
    sleep(1.2) # wait for crit message to disappear

def battle_iteration():
    meds = detect_med_buttons(screen_img)
    for name, bounds in meds:
        progress_log('Clicking meds...')
        mouse_click(bounds.x, bounds.y)
        sleep(0.05)

    crits = detect_critical_button(screen_img)
    for bounds in crits:
        progress_log('Clicking crits...')
        mouse_click(bounds.x, bounds.y)
        sleep(0.3)
        make_critical_strike(bounds)
        sleep(0.8)

    return len(meds) > 0, len(crits) > 0

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
        progress_log('Zoomout: full delay')

def zoom_in(x, y):
    mouse_input.position = (x, y)
    for _ in range(5): mouse_input.scroll(0, 1)
    sleep(1)

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

def look_for_room_using_structural():
    zoom_out()
    reached_left, reached_right = False, False
    direction = 'down' # camera actually goes up

    def scan_primary(directions):
        nonlocal reached_left, reached_right
        if direction in directions: 
            pan_camera(direction, camera_pan_initial_duration)
            reached_left = False
            reached_right = False
            return True
        
        return False

    def scan_secondary():
        if not reached_left:
            pan_camera('left', camera_pan_initial_duration)
            return True
        if not reached_right:
            pan_camera('right', camera_pan_initial_duration)
            return True
        
        return False

    while True:
        progress_log('Structural room scanning iteration...')
        no_overlay_grab_screen()
        rooms = detect_rooms(screen_img)
        if len(rooms) > 0: 
            progress_log(f'Structural scan found room: {rooms[0][2]}')
            return True
        
        structural, directions = detect_structural(screen_img)
        progress_log(f'Obscured directions: {directions}')
        reached_left = reached_left or 'left' not in directions
        reached_right = reached_right or 'right' not in directions

        if direction == 'down': # we want reach `top` as soon as possible
            if scan_primary(directions) or scan_secondary(): continue
        else: # on the way to the `bottom` we want scan everything
            if scan_secondary() or scan_primary(directions): continue
        
        # we reached all: direction, left and right!
        if direction == 'down' and 'up' in directions:
            progress_log('Top-way scan finished, starting reverse scan...')
            direction = 'up'
            continue
        else:
            progress_log('Scan finished, noting found :(')
            return False

def match_room_with_structural(structural_rooms, room_bounds):
    for room in structural_rooms:
        if abs(room.x - room_bounds.x) > struct_room_match_tolerance * room_bounds.width: continue
        if abs(room.y - room_bounds.y) > struct_room_match_tolerance * room_bounds.height: continue
        if abs(room.width - room_bounds.width) > struct_room_match_tolerance * room_bounds.width: continue
        if abs(room.height - room_bounds.height) > struct_room_match_tolerance * room_bounds.height: continue

        progress_log(f'Matched structural room: {room_bounds}')
        progress_log(f'Matching room: {room}')
        return True

    return False
        
def mission_script():
    global script_running, current_execution_target, mission_ctl

    update_overlay()
    progress_log('>>> Starting mission script')
    progress_log('Looking for starting room...')
    script_running = True
    current_execution_target = None

    last_rooms = None
    last_time_room_detected = perf_counter()
    last_iteration_room_detected = True
    while True:
        progress_log('General mission iteration...')
        
        if perf_counter() - last_time_room_detected >= structural_scan_begin_timeout and not last_iteration_room_detected:
            progress_log('General iteration timeout, engaging structure-based scanning...')
            if not look_for_room_using_structural():
                progress_log('Failed to detect new rooms, aborting execution')
                break
        
        last_iteration_room_detected = False

        # new 
        in_battle = False
        new_room_discovered = False
        in_dialogue = False
        
        zoom_out() # general iteration on max zoom out for simplicity
        no_overlay_grab_screen() # grab screen for this iteration
        debug_log_image(screen_img, 'iteration-capture')
        clear_debug_canvas()
        battle_iteration() # take care of meds / crits (if any)

        # our next action - go to the new room, detect some..
        rooms = detect_rooms(screen_img)

        if len(rooms) == 0:
            progress_log('No rooms detecting, skipping iteration...')
            continue

        last_time_room_detected = perf_counter()
        last_iteration_room_detected = True

        progress_log(f'Found room: {rooms[0][2]}, type: {rooms[0][0]}')
        last_room_type = rooms[0][0]
        navigate_successful, room_bounds, pre_click_img = navigate_to_room(rooms[0][2], click=True)
        if not navigate_successful: continue
        progress_log(f'Room navigation successful, waiting for walk complete...')
        last_rooms = detect_rooms(no_overlay_grab_screen())

        # wait until the room is reached
        while True:
            start_time = perf_counter()
            current_room = no_overlay_grab_screen(room_bounds.get_bbox())
            diff = compute_img_diff_ratio(pre_click_img, current_room)
            debug_log_image(current_room, f'walk-wait-{diff*100:0.2f}-diff')
            progress_log(f'Waiting for walk completion: current diff {100*diff:0.2f}%')

            # min room entered diff threshold
            if diff >= room_reached_min_diff: break # room reached!
            elapsed = perf_counter() - start_time
            if elapsed < walk_wait_min_interval: sleep(walk_wait_min_interval - elapsed)

        # we reached room - wait until enemies are detected OR a new room is detected OR 4 seconds pass
        progress_log(f'Walk complete! Analyzing situation...')
        post_walk_screen_img = no_overlay_grab_screen()
        debug_log_image(post_walk_screen_img, f'post-walk-structural')

        proceed = False
        while not proceed:
            start_time = perf_counter()
            while perf_counter() - start_time <= room_analysis_timeout:
                no_overlay_grab_screen()
                debug_log_image(screen_img, f'wait-iteration')
                have_enemies, _ = detect_enemies(screen_img)
                have_crits = detect_critical_button(screen_img)
                if have_enemies or have_crits:
                    in_battle = True
                    have_enemies = True
                    progress_log('Enemies detected!')
                    proceed = True # enemies, no need to stay
                    break
                
                rooms = detect_rooms(screen_img)
                if has_new_room(rooms, last_rooms):
                    new_room_discovered = True
                    progress_log('New room detected (?), proceeding...')
                    break
                
            new_room_confirmed = new_room_discovered and room_bounds.height == rooms[0][2].height
            if new_room_confirmed:
                progress_log('New room confirmed, proceeding...')
                proceed = True # new room, no need to stay
            elif not have_enemies:
                progress_log('Matching structural...')
                no_overlay_grab_screen()
                debug_log_image(screen_img, f'structural-room-match')
                structural, _ = detect_structural(post_walk_screen_img)
                structural_match = np.sum(np.all(screen_img[structural.points[:,0], structural.points[:,1]] == structural_color, axis=1))
                structural_match /= structural.point_count
                # structural_rooms = detect_structural_rooms(structural)
                # in_dialogue = not match_room_with_structural(structural_rooms, room_bounds)
                in_dialogue = structural_match < 0.6
                progress_log(f'Structural match: {100*structural_match:0.2f}%')
                # central_room = next((x for x in structural_rooms if x.contains_point(*screen_center_xy)), None)
                if in_dialogue: 
                    progress_log(f'Dialogue detected!')
                elif not new_room_discovered:
                    progress_log('Structural matched, nothing to do - aborting iteration')
                    proceed = True # nothing to do - proceeding
                    continue
                else:
                    proceed = True # unconfirmed, but new room - no need to stay

            while have_enemies:
                no_overlay_grab_screen() # grab screen for this iteration
                debug_log_image(screen_img, 'battle-iteration-capture')
                had_meds, had_crits = battle_iteration() # take care of meds / crits
                if had_crits: continue # critical hit marker can obscure enemies and they go undetected

                have_enemies, enemies = detect_enemies(screen_img)
                progress_log(f'Battle iteration: {len(enemies)} enemies detected')

            while in_dialogue:
                progress_log(f'Dialog: waiting...')
                zoom_out()
                no_overlay_grab_screen() # grab screen for this iteration
                debug_log_image(screen_img, 'dialogue-iteration-capture')
                dialogue_choice, buttons = detect_dialogue_buttons(screen_img)
                if dialogue_choice:
                    progress_log('Dialog choice detected!')
                    progress_log(f'Starting handler: {dialogue_mode}')
                    dialogue_handlers[dialogue_mode](buttons)
                    sleep(0.2)
                    continue

                # structural, _ = detect_structural(screen_img)
                # structural_rooms = detect_structural_rooms(structural)
                structural_match = np.sum(np.all(screen_img[structural.points[:,0], structural.points[:,1]] == structural_color, axis=1))
                structural_match /= structural.point_count
                progress_log(f'Dialog: match {100*structural_match:0.2f}%')
                # if not match_room_with_structural(structural_rooms, central_room):
                if structural_match >= 0.6:
                    progress_log(f'Dialog over!')
                    break
        
        if last_room_type != 'elevator':
            progress_log(f'Proceeding to loot collection!')
            loot_bounds = detect_loot(no_overlay_grab_screen())
            progress_log(f'Detected {len(loot_bounds)} loots')
            for loot in loot_bounds:
                mouse_click(*loot.pos)
                sleep(0.4)

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
        global last_key_pressed
        last_key_pressed = key

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
