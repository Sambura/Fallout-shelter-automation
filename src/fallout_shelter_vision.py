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
        if clearance_fraction < struct_min_room_clearance_fraction: 
            progress_log(f'Filtered out structural room: {clearance_fraction} clearance')
            continue

        # _, rect_fraction = is_fragment_rectangular(fragment, report_fraction=True)
        if not hasattr(fragment, 'patch_mask'): fragment.compute(patch_mask=True)

        # cut 10% of pixels from each side before rectangular check (to cut away some potential bumps)
        dw = int(fragment.bounds.width * 0.1)
        dh = int(fragment.bounds.height * 0.1)

        _, rect_fraction = _is_fragment_rectangular(fragment.patch_mask[dh:-dh, dw:-dw])
        if rect_fraction < struct_min_room_border_fraction: continue

        rooms.append(fragment.bounds.offset(structural.bounds.low_pos))
        
        # progress_log(f'Detected structural room ({fragment.bounds}) : border fraction {rect_fraction*100:0.2f}%, clearance: {clearance_fraction*100:0.2f}%')
        if debug_show_progress_visuals:
            draw_border(get_do(), rooms[-1], get_debug_color(fragment.fragment_value), thickness=2)

    return rooms

static_loot_general_var = 0
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

@print_durations()
def new_detect_loot_general_debug(scan_bounds, grab_screen_func):
    global static_loot_general_var
    result_log('[Debug general loot detection]')

    static_loot_general_var += 1
    
    # collect frame data
    frames = []
    interval = 0 # 0.5
    reset_mock_frame_index()
    for x in slow_loop(interval=interval, max_iter_count=2):
        frames.append(grab_screen_func(scan_bounds.get_bbox()))

    frame1, frame2 = frames
    color_diff = frame1.astype(int) - frame2 # colorful signed diffs
    # doesn't really make sense, eh?
    # pos_diff = np.clip(color_diff, 0, 255).astype(np.uint8) # only positive changes
    # neg_diff = -np.clip(color_diff, -255, 0).astype(np.uint8) # only negative changes
    unsigned_color_diff = np.abs(color_diff).astype(np.uint8) # color_diffs but unsigned
    any_changes = np.any(color_diff != 0, axis=2) # all pixels that changed

    # std match, 115/1.5 #ebda78 (score: 1938)
    color_match = match_color_grades_std(color_diff, np.array([235, 218, 120]), min_pixel_value=105, tolerance=1.5)
    color_score = np.mean(color_diff, axis=2) * color_match
    color_score /= np.max(color_score)
    
    progress_log(f'Convolving...')
    convolved_match = box_blur(color_score, 3).squeeze()
    threshold = 0.17
    filtered_match = convolved_match > threshold

    color_match_old = match_color_grades(color_diff, loot_particles_base_color, min_pixel_value=120, tolerance=10)
    progress_log(f'Convolving...')
    threshold = 0.5
    filtered_match_old = box_blur(color_match_old.astype(float), 5).squeeze() > threshold

    # if static_loot_general_var % 2 == 1:
    #     color_match_pos = match_color_grades(pos_diff, loot_particles_base_color, min_pixel_value=120, tolerance=10)
    #     color_match_neg = match_color_grades(neg_diff, loot_particles_base_color, min_pixel_value=120, tolerance=10)
    # else:
    #     color_match_pos = match_color_grades(pos_diff, general_loot_diff_color, min_pixel_value=1, tolerance=10000)
    #     color_match_neg = match_color_grades(neg_diff, general_loot_diff_color, min_pixel_value=1, tolerance=10000)
     #color_match_cum = color_match_pos | color_match_neg

    # color_score_pos = (np.mean(pos_diff, axis=2) * color_match_pos).astype(np.uint8)
    # color_score_neg = (np.mean(neg_diff, axis=2) * color_match_neg).astype(np.uint8)

    # color_score = (color_score_pos.astype(float) + color_score_neg.astype(float)) / 2
    # color_score = color_score ** 2

    while True:
        get_do()[:,:,:3] = frame1
        get_do()[:,:,3] = 255
        sleep(0.8)
        progress_log('Cumulative diffs:')
        get_do()[:,:,:3] = unsigned_color_diff
        get_do()[:,:,3] = 255
        sleep(1.5)
        progress_log('Color match results (new):')
        get_do()[:,:,:3] = frame1 # overwrite on top of original frame for nice visuals ^^
        get_do()[:,:,1] = 255 * color_match
        # get_do()[:,:,0] = 255 * (any_changes[0] & ~color_matches_cum[0])
        get_do()[:,:,0] //= 2
        get_do()[:,:,2] = 0
        get_do()[:,:,3] = 255
        sleep(1.5)
        progress_log('Matches scores:')
        get_do()[:,:,0] = (255 * color_score).astype(np.uint8)
        get_do()[:,:,1] = (255 * color_score).astype(np.uint8)
        get_do()[:,:,2] = (255 * color_score).astype(np.uint8)
        sleep(3.5)
        # break
        #           progress_log('Filtered matches:')
        #           get_do()[:,:,1] = 255 * filtered_match
        #           get_do()[:,:,0] = 255 * (color_match_cum & ~filtered_match)
        #           get_do()[:,:,3] = 255
        #           sleep(2.5)

        progress_log('Filtered results (discrete):')
        get_do()[:,:,:3] = frame1 # overwrite on top of original frame for nice visuals ^^
        get_do()[:,:,1] = 255 * filtered_match_old
        get_do()[:,:,0] //= 2
        get_do()[:,:,2] = 0
        get_do()[:,:,3] = 255
        sleep(2.5)
        progress_log('Filtered results (float):')
        get_do()[:,:,:3] = frame1 # overwrite on top of original frame for nice visuals ^^
        get_do()[:,:,1] = 255 * filtered_match
        get_do()[:,:,0] //= 2
        get_do()[:,:,2] = 0
        get_do()[:,:,3] = 255
        sleep(2.5)

        # break

@print_durations()
def new_detect_loot_corpse_debug(scan_bounds, grab_screen_func):
    search_iterations = 2
    search_interval = 0.000001 # 1 / 8 # 8 fps
    progress_log(f'Loot collection start: {search_iterations} frames at {1 / search_interval} FPS')
    if True:
        reset_mock_frame_index()
    
    # collect frame data
    frames = []
    for x in slow_loop(interval=search_interval, max_iter_count=search_iterations):
        frames.append(grab_screen_func(scan_bounds.get_bbox()))
        # debug_log_image(frames[-1], f'loot-collection-frame-{x}')

    # for i in range(search_iterations):
    #     debug_log_image(frames[i], f'loot-frame-{i}')

    progress_log(f'Frames collected, computing...')
    # frames = frames[::4] # space out frames in time
    progress_log(f'Stripping frames...')



    color_diffs = [x.astype(int) - y for x, y in zip(frames[:-1], frames[1:])] # colorful signed diffs
    pos_diffs = [np.clip(x, 0, 255).astype(np.uint8) for x in color_diffs] # only positive changes
    neg_diffs = [(- np.clip(x, -255, 0)).astype(np.uint8) for x in color_diffs] # only negative changes
    unsigned_color_diffs = [np.abs(x).astype(np.uint8) for x in color_diffs] # color_diffs but unsigned
    any_changes = [np.any(x != 0, axis=2) for x in color_diffs] # all pixels that changed

    # color_matches = [match_cont_color(x, corpse_particles_base_color, corpse_color_detection_threshold) for x in unsigned_color_diffs]
    # color_matches_pos = [match_cont_color(x, corpse_particles_base_color, corpse_color_detection_threshold) for x in pos_diffs]
    # color_matches_neg = [match_cont_color(x, corpse_particles_base_color, corpse_color_detection_threshold) for x in neg_diffs]
    color_matches = [match_color_grades(x, corpse_particles_base_color, min_pixel_value=80, tolerance=15) for x in unsigned_color_diffs]
    # color_matches_pos = [match_color_grades(x, corpse_particles_base_color, min_pixel_value=80, tolerance=15) for x in pos_diffs]
    # color_matches_neg = [match_color_grades(x, corpse_particles_base_color, min_pixel_value=80, tolerance=15) for x in neg_diffs]
    # color_matches_cum = [x | y for x, y in zip(color_matches_pos, color_matches_neg)]

    progress_log(f'Convolving...') # NOT MATCHES, IT IS A SINGLE ONE RIGHT NOW FIXXXX
    convolved_matches = box_blur(color_matches[0].astype(float), 5).squeeze()
    threshold = 0.9
    filtered_matches = convolved_matches > threshold

    while True:
        get_do()[:,:,:3] = frames[0]
        get_do()[:,:,3] = 255
        sleep(1)
        get_do()[:,:,:3] = frames[1]
        get_do()[:,:,3] = 255
        sleep(1)
        progress_log('Positive diffs:')
        get_do()[:,:,:3] = pos_diffs[0]
        get_do()[:,:,3] = 255
        sleep(1.5)
        progress_log('Negative diffs:')
        get_do()[:,:,:3] = neg_diffs[0]
        get_do()[:,:,3] = 255
        sleep(1.5)
        progress_log('Cumulative diffs:')
        get_do()[:,:,:3] = unsigned_color_diffs[0]
        get_do()[:,:,3] = 255
        sleep(0.8)
        progress_log('Color match results:')
        get_do()[:,:,:3] = frames[0] # overwrite on top of original frame for nice visuals ^^
        get_do()[:,:,0] //= 2
        get_do()[:,:,1] = 255 * color_matches[0]
        get_do()[:,:,2] = 0
        get_do()[:,:,3] = 255
        sleep(4)
        progress_log('Color match results (filtered):')
        get_do()[:,:,1] = 255 * filtered_matches
        sleep(4)

@print_durations()
def detect_loot(scan_bounds, grab_screen_func=None, frames=None):
    "provide either screen grab function or list of 2 frames"
    if frames is None:
        search_iterations = 2
        search_interval = 0.5 # 1 / 8 # 8 fps
        progress_log(f'Loot collection start: {search_iterations} frames at {1 / search_interval} FPS')

        # collect frame data
        frames = []
        for x in slow_loop(interval=search_interval, max_iter_count=search_iterations):
            frames.append(grab_screen_func(scan_bounds.get_bbox()))
            debug_log_image(frames[-1], f'loot-collection-frame-{x}') # do NOT disable counter increment
        progress_log(f'Frames collected, computing...')
    
    frame1, frame2 = frames

    # base computations
    color_diff = frame1.astype(int) - frame2 # colorful signed diffs
    unsigned_color_diff = np.abs(color_diff).astype(np.uint8) # color_diffs but unsigned
    # any_changes = np.any(color_diff != 0, axis=2)

    # general loot
    color_match = match_color_grades_std(color_diff, np.array([235, 218, 120]), min_pixel_value=105, tolerance=1.5)
    color_score = np.mean(color_diff, axis=2) * color_match
    color_score /= np.max(color_score)
    
    progress_log(f'Convolving...')
    convolved_match_general = box_blur(color_score, 3).squeeze()
    threshold = 0.17
    filtered_match_general = convolved_match_general > threshold

    # corpse loot
    color_match_corpse = match_color_grades(unsigned_color_diff, corpse_particles_base_color, min_pixel_value=80, tolerance=15)

    progress_log(f'Convolving...') # NOT MATCHES, IT IS A SINGLE ONE RIGHT NOW FIXXXX
    convolved_match_corpse = box_blur(color_match_corpse.astype(float), 5).squeeze()
    threshold = 0.9
    filtered_match_corpse = convolved_match_corpse > threshold

    # collection (?)
    filtered_matches = filtered_match_general | filtered_match_corpse

    loot_coords = []
    grid_size = 10 # let's say 1080p => 10x10 grid (~20k pixels per cell)
    while scan_bounds.area / (grid_size * grid_size) < 30000: grid_size -= 1
    point_color = get_debug_color(randrange(10))
    for y in range(grid_size):
        for x in range(grid_size):
            start_pos = (int(x * scan_bounds.width / grid_size), int(y * scan_bounds.height / grid_size))
            end_pos = (int((x + 1) * scan_bounds.width / grid_size), int((y + 1) * scan_bounds.height / grid_size))
            cell_bounds = Bounds(*start_pos, *end_pos)
            cell_patch = filtered_matches[cell_bounds.to_slice()]
            ys, xs = cell_patch.nonzero()
            if len(xs) == 0: continue

            draw_border(get_do(), cell_bounds.offset(scan_bounds.low_pos), np.array([0, 200, 40, 255]), thickness=3)

            click_index = randrange(0, len(xs))
            target_pos = np.array([xs[click_index], ys[click_index]]) + scan_bounds.low_pos + start_pos

            draw_disk(get_do(), np.array(target_pos), 7, point_color)
            loot_coords.append(target_pos)
            result_log(f'Detected loot candidate at cell {x + 1}; {y + 1}')

    return loot_coords

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

def detect_character_panel(pixels):
    "detect the character panel that pops up on the left when you click a character"
    dh = int(pixels.shape[0] * 0.1) # get pixels from 10% to 90% vertically
    width = int(pixels.shape[1] * 0.25) # get pixels from 0% to 25% horizontally

    panel_pixels = pixels[dh:-dh, 0:width]
    low_color_count = np.sum(match_color_exact(panel_pixels, character_panel_low_color))
    high_color_count = np.sum(match_color_exact(panel_pixels, ui_primary_color))

    return low_color_count >= character_panel_min_low_pixels and high_color_count >= character_panel_min_high_pixels
