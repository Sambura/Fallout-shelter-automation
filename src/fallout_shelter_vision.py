# Function specific to the game
# Most of the functions are used to detect certain in-game objects / ui elements

from .debug import *
from .vision import *
from .game_constants import *
from .drawing import *
from .visual_debug import *
from .util import *

import numpy as np
from scipy.signal import convolve2d
from time import perf_counter, sleep
from funcy import print_durations
from random import randrange

def get_room_type(room, loc):
    is_normal_room = room.width >= room.height
    if is_normal_room: return 'room'
    if loc != 'full': return 'unknown'
    return 'elevator'

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
            draw_border(get_do(), fragment.bounds, np.array([255, 0, 0, 255]), thickness=1)

        valid_room, location = analyze_room(fragment)
        if not valid_room: continue
        room_type = get_room_type(fragment.bounds, location)

        rooms_detected.append((room_type, location, fragment.bounds))
        detected_fragments.append(fragment)

        result_log(f'Room detected! {fragment.bounds} : {room_type}, location: {location}')
        if debug_show_result_visuals and np.all(pixels.shape[:2] == screen_shape):
            draw_border(get_do(), fragment.bounds, np.array([0, 0, 180, 255]), thickness=3)

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
        #     get_do()[mask == fragment.fragment_value] = get_debug_color(fragment.fragment_value)
        # progress_log(f'New fragment: ratio: {color_ratio * 100:0.2f}%; {clean_color_fraction * 100:0.2f}% clean colors')

        if fragment.point_count < healing_button_min_pixels: continue

        checkers = [stimpak_check, levelup_check, antirad_check]
        if abs(color_ratio - stimpak_color_ratio) > abs(color_ratio - antirad_color_ratio):
            checkers = checkers[::-1]

        for checker in checkers: 
            if checker(fragment, color_ratio, clean_color_fraction): break
        else: continue

        if debug_show_result_visuals:
            get_do()[mask == fragment.fragment_value] += get_debug_color(fragment.fragment_value)
    
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
            get_do()[mask == fragment.fragment_value] += get_debug_color(fragment.fragment_value)

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
            draw_border(get_do(), button_bounds, np.array([255, 255, 255, 80]), 6)

    return detected

@print_durations()
def detect_structural(pixels):
    """Detects the structural (the black-colored area that surrounds all rooms)
    Works by finding all fragments comprised of exactly black pixels and selecting the largest of them
    Fragments with small pixel count are filtered out
    Returns (fragment, obscured_directions): fragment is the found structural, and obscured_directions is
        a list of directions (strings) in which you can move camera to reveal obscured part of the detected
        structural
    """
    create_debug_frame()
    fragments_mask = match_color_exact(pixels, structural_color)
    fragments, _, _, mask = detect_fragments(pixels, fragments_mask)

    for fragment in fragments[:]: # copy list
        if fragment.point_count < struct_min_room_pixels: 
            fragments.remove(fragment); continue
        
        # progress_log(f'Detected structural fragment: {fragment.point_count} pixel count')
        if debug_show_progress_visuals:
            get_do()[mask == fragment.fragment_value] = get_debug_color(fragment.fragment_value)

    if len(fragments) == 0: return None, None

    best_candidate = max(fragments, key=lambda x: x.point_count)
    obscured_directions = []
    best_candidate.compute(patch_mask=True)
    if best_candidate.bounds.y_min == 0: obscured_directions.append('down')
    if best_candidate.bounds.y_max == pixels.shape[0] - 1: obscured_directions.append('up')
    if best_candidate.bounds.x_min == 0: obscured_directions.append('left')
    if best_candidate.bounds.x_max == pixels.shape[1] - 1: obscured_directions.append('right')

    return best_candidate, obscured_directions

# returns are_there_enemies, fragments
@print_durations()
def detect_enemies(pixels):
    create_debug_frame()
    fragments_mask = detect_blending_pixels(pixels, enemy_healthbar_border_color, enemy_healthbar_outline_color, 
        enemy_healthbar_detect_blending, enemy_healthbar_color_max_deviation)

    fragments, _, _, mask = detect_fragments(pixels, fragments_mask)
    detected = []

    for fragment in fragments:
        if debug_show_progress_visuals:
            get_do()[mask == fragment.fragment_value] += get_debug_color(fragment.fragment_value)

        if fragment.point_count < enemy_healthbar_min_border_pixel_count: continue

        if is_fragment_rectangular(fragment):
            detected.append(fragment)
            if debug_show_progress_visuals:
                get_do()[mask == fragment.fragment_value] += np.array([50, 255, 20, 255])

    return len(detected) > 0, detected

# [?] Accepts a structural fragment (from detect_structural) and returns the list of rooms
# in this structural (as an array of Bounds)
# ended up not using this :( 
# primary problem is that sometimes some text appears above characters
# and messes up with structural detection
@print_durations()
# TODO: small bumps on room bounds can lead to detection of too large room (e.g. text overlaps with structural)
# fix that at some point....
def detect_structural_rooms(structural: Fragment):
    if structural is None: return []
    create_debug_frame()
    if not hasattr(structural, 'patch_mask'): structural.compute(patch_mask=True)
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
            draw_border(get_do(), rooms[-1], get_debug_color(fragment.fragment_value), thickness=2)

    return rooms

@print_durations()
def detect_dialogue_buttons(pixels, screen_size):
    create_debug_frame()
    fragments_mask = match_color_fuzzy(pixels, primary_dialogue_button_color, dialogue_button_color_max_deviation)
    fragments, _, _, _ = detect_fragments(pixels, fragments_mask)
    detected = []

    for fragment in fragments:
        rel_width, rel_height = fragment.bounds.width / screen_size[0], fragment.bounds.height / screen_size[1]
        if rel_width < min_dialogue_button_rel_size[0] or rel_height < min_dialogue_button_rel_size[1]: continue
        rel_x = fragment.bounds.x / screen_size[0]
        if abs(rel_x - 0.5) > dialogue_button_max_center_deviation_rel: continue
        _, rect_fraction = is_fragment_rectangular(fragment, report_fraction=True)
        if rect_fraction < min_dialogue_button_border_fraction: continue

        # progress_log(f'Detected dialogue button: {fragment.bounds}')
        if debug_show_progress_visuals:
            draw_border(get_do(), fragment.bounds, np.array([200, 25, 10, 255], dtype=np.uint8), 4)

        detected.append(fragment)

    # just in case
    detected = sorted(detected, key=lambda x: x.bounds.y)

    return len(detected) > 0, detected

# TODO: split in two functions, one of which does not require grab_screen (?)
@print_durations()
def detect_loot(pixels, grab_screen_func):
    create_debug_frame()
    structural, _ = detect_structural(pixels)
    delete_debug_frame()
    rooms = detect_structural_rooms(structural)
    screen_shape = (np.array(pixels.shape[:2][::-1])).astype(int)
    screen_center = (screen_shape / 2).astype(int)

    central_room = next((x for x in rooms if x.contains_point(*screen_center)), None)
    if central_room is None:
        progress_log(f'!Critical: failed to detect central room! Fallback to whole screen')
        central_room = Bounds(0, 0, screen_shape[0] - 1, screen_shape[1] - 1)
    else:
        # TODO: when not using full-screen capture, coordinates are not properly compensated, fix that (or is it fine?)
        # central_room = Bounds(0, 0, screen_shape[0] - 1, screen_shape[1] - 1)
        progress_log(f'Detected main room for loot collection: {central_room}')
    
    search_iterations = 24
    search_interval = 1 / 8 # 8 fps
    progress_log(f'Loot collection start: {search_iterations} frames at {1 / search_interval} FPS')
    
    # collect frame data
    frames = []
    for x in slow_loop(interval=search_interval, max_iter_count=search_iterations):
        frames.append(grab_screen_func(central_room.get_bbox()))
        debug_log_image(frames[-1], f'loot-collection-frame-{x}')

    # for i in range(search_iterations):
    #     debug_log_image(frames[i], f'loot-frame-{i}')

    progress_log(f'Frames collected, computing...')

    color_diffs = [np.abs(x.astype(int) - y).astype(np.uint8) for x, y in zip(frames[:-1], frames[1:])] # colorful diffs
    any_changes = np.sum([np.any(x > 0, axis=2) for x in color_diffs], axis=0) > 0 # mask where at least one pixel changed
    # not `scores` anymore but ok
    scores = [match_color_tone(x, loot_particles_base_color, min_pixel_value=100, tolerance=10) for x in color_diffs] # scores for loot detection

    # debug snippet
    if False: # does not work for non-fullscreen capture
        get_do()[:, :, 3] = 255
        for i in range(len(color_diffs)):
            progress_log(f'showing diff #{i + 1}')
            get_do()[:, :, :3] = color_diffs[i]
            sleep(2.5)
            progress_log(f'showing scores')
            get_do()[:, :, 0] = scores[i] * 255
            get_do()[:, :, 1] = scores[i] * 255
            get_do()[:, :, 2] = 0
            sleep(2.5)
    # debug snippet

    matches = [match_cont_color(x, corpse_particles_base_color, corpse_color_detection_threshold) for x in color_diffs] # matches for corpse detection
    # pos_matches = np.sum(scores, axis=0) * any_changes > 0 # pixels that match loot
    pos_matches = np.sum(scores, axis=0) * any_changes > 0 # pixels that match loot
    neg_matches = any_changes ^ pos_matches # pixels that don't match loot
    s_scores = pos_matches.astype(int) - neg_matches.astype(int) # combination of last 2
    for x, y in zip(matches, color_diffs): x[np.all(y == 0, axis=2)] = 0 # filter matches (?)
    cum_match = np.sum(matches, axis=0) > 0 # cumulative matches
    # get_do()[:,:,0] = 255 * neg_matches
    # get_do()[:,:,2] = 255 * pos_matches
    # get_do()[:,:,3] = 255 * any_changes

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
            progress_log(f'Discarded corpse candidate: not enough pixels: {corpse_frags[i].bounds}')
            del corpse_frags[i]

    for fragment in loot_frags[:]: # copy list
        fragment_val = np.mean(fragment.masked_patch)
        if fragment_val < 0: 
            loot_frags.remove(fragment)
            continue
        # progress_log(f'Fragment: {fragment.bounds}, value: {fragment_val}')

        draw_border(get_do(), fragment.bounds.offset(central_room.low_pos), get_debug_color(randrange(10)), 1)

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
        #     get_do()[fragment.bounds.to_slice(offset=central_room.low_pos)] = get_debug_color(i)

        draw_border(get_do(), group_bounds[i].offset(central_room.low_pos), np.array([28, 125, 199, 255]), 3)

    for fragment in corpse_frags:
        if fragment.bounds.are_smaller_than(corpse_min_fragment_size):
            progress_log(f'Discarding corpse candidate: {fragment.bounds}')
            continue
        else:
            filtered_groups.append(fragment.bounds.offset(central_room.low_pos))

        draw_border(get_do(), fragment.bounds.offset(central_room.low_pos), np.array([255, 0, 0, 255]), 3)

    return filtered_groups

def is_room_click_diff(diff: np.ndarray):
    "Given the diff of the room, determine whether the room was clicked or not"
    border_width = int(room_click_diff_border_width * diff.shape[0])
    mask = np.ones_like(diff, dtype=bool)
    mask[border_width:-border_width, border_width:-border_width] = False
    hh, hw = mask.shape[0] // 2, mask.shape[1] // 2
    border = np.logical_and(diff, mask)

    # check there are pixels in border area in each of the 4 quadrants
    return np.any(border[:hh, :hw]) and np.any(border[:hh, hw:]) and np.any(border[hh:, :hw]) and np.any(border[hh:, hw:])

def detect_ui(pixels):
    "Returns true/false if the UI buttons are on screen (menu, screenshot, mission menu)"
    # look at corners of screen:
    #   - top-right: 6% square (mission button)
    #   - bottom-left: 6% square (screenshot button)
    #   - bottom-right: 11% square (menu button)
    shape = pixels.shape[:2]
    height6 = int(shape[0] * 0.06)
    width6 = int(shape[1] * 0.06)
    height11 = int(shape[0] * 0.11)
    width11 = int(shape[1] * 0.11)

    top_right_patch = pixels[:height6, shape[1] - width6:]
    bottom_left_patch = pixels[shape[0] - height6:, :width6]
    bottom_right_patch = pixels[shape[0] - height11:, shape[1] - width11:]

    have_patch1 = np.sum(match_color_exact(top_right_patch, ui_primary_color)) > ui_minimal_pixel_count
    have_patch2 = np.sum(match_color_exact(bottom_left_patch, ui_primary_color)) > ui_minimal_pixel_count
    have_patch3 = np.sum(match_color_exact(bottom_right_patch, ui_primary_color)) > ui_minimal_pixel_count

    return have_patch1 and have_patch2 and have_patch3
