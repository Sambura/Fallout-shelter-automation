from src.game_constants import *
from src.vision import *
from src.util import *
from src.drawing import *
from src.debug import *
from src.fallout_shelter_vision import *
from src.visual_debug import *

import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import tkinter as tk
import threading
import io
from pynput import keyboard, mouse
from time import sleep, perf_counter
import traceback
from random import randrange

def visualize_fragment_grouping(fragments_mask, radius=40):
    create_debug_frame()
    fragments, _, _ = detect_fragments(fragments_mask, fragments_mask, patch_mask=True)

    for fragment in fragments:
        draw_circle(get_do(), fragment.bounds.pos, radius, np.array([255,255,255,255]))
    
    if len(fragments) == 0: return
    start_fragment = fragments[randrange(len(fragments))]
    draw_circle(get_do(), start_fragment.bounds.pos, radius, np.array([255,0,0,255]))
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
                        draw_circle(get_do(), fragment.bounds.pos, radius, np.array([0,255,0,255]))
                        break_flag = True
                        break
                if break_flag: break

# down and up are switched for stupid reasons (down means smaller coordinate, screen top is 0)
CAMERA_PAN_KEYS = { 'left': 'a', 'right': 'd', 'down': 'w', 'up': 's' }

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

###
### ======================== MAIN ========================
### 

class FalloutShelterAutomationApp:
    # constants (?)
    version = 'v0.4.3'
    app_update_interval_ms: int = 20
    "Controls how often does app make regular update iterations (e.g. updating displayed log, starting async tasks, etc.)"
    chord_start_sequence: str = '<ctrl>+f'

    # mission script parameters (game related)
    camera_pan_deadzone_size: float = 0.05
    "Precision of camera panning - the center of the room should end up in a rectangle with dimensions camera_pan_deadzone_size * screen_shape"
    camera_pan_initial_duration = 0.1
    crit_wait_count = 4
    dialogue_mode: str = 'manual' # random / manual
    "Controls which dialog handler is used when selecting dialog option"

    # state attributes
    tick_counter = 0
    "Counts the number of times FalloutShelterAutomationApp.update() was called"
    current_execution_target = None
    repeat_current_execution_target = False
    terminate_pending = False
    keyboard_chord_pending = False
    task_in_progress = False
    script_running = False
    last_key_pressed = None
    overlay_on = True
    show_log = True
    state_overlay: np.ndarray = None
    "Current overlay image that represents app state (normal / error) (currently as a red/yellow border)"

    ### 
    ### Game automation functions
    ###

    def direction_to_screen_center(self, loc: Bounds, blocked: str):
        "Which way should camera be panned for `loc` to move towards screen center? `blocked` specifies direction along which panning is not possible"
        # note that if loc is on the left of the screen, we should move right for it to be centered
        if not is_horizontal(blocked):
            if loc.x < self.camera_deadzone.x_min: return 'left'
            if loc.x > self.camera_deadzone.x_max: return 'right'

        if not is_vertical(blocked):
            if loc.y < self.camera_deadzone.y_min: return 'down'
            if loc.y > self.camera_deadzone.y_max: return 'up'

        return None

    def get_panning_bbox(self, bounds, direction):
        "Returns bbox to be used for screen grab when panning in `direction` and keeping `bounds` in view after panning"
        if direction == 'left':
            return Bounds(bounds.x_min, bounds.y_min, self.screen_shape_xy[0] - 1, bounds.y_max).get_bbox()
        elif direction == 'right':
            return Bounds(0, bounds.y_min, bounds.x_max, bounds.y_max).get_bbox()
        elif direction == 'up':
            return Bounds(bounds.x_min, 0, bounds.x_max, bounds.y_max).get_bbox()
        elif direction == 'down':
            return Bounds(bounds.x_min, bounds.y_min, bounds.x_max, self.screen_shape_xy[1] - 1).get_bbox()

    def dialogue_random_handler(self, buttons):
        button_index = randrange(len(buttons))

        self.mouse_click(*buttons[button_index].bounds.pos)
        progress_log(f'Clicking button #{button_index + 1}...')

    def dialogue_manual_handler(self, buttons):
        progress_log(f'Waiting input: 1-{len(buttons)}...')

        # technically it is *kind of* a race condition, but we don't care right?
        self.last_key_pressed = None
        while self.last_key_pressed is None or not hasattr(self.last_key_pressed, 'char') or not ('1' <= self.last_key_pressed.char <= '9'):
            sleep(0.1)

        button_index = int(self.last_key_pressed.char) - 1

        progress_log(f'Clicking button #{button_index + 1}...')
        self.mouse_click(*buttons[button_index].bounds.pos)

    def navigate_to_room(self, room_bounds: Bounds, click=True):
        progress_log(f'Navigating to the room... : {room_bounds}')

        pan_duration = self.camera_pan_initial_duration
        last_direction = None
        last_bounds = Bounds(*(np.ones(4)*(-10000000000))) # idk lol
        blocked_direction = None

        room_anchors = room_bounds.get_corners(get_8=True)
        furthest_anchor_i = np.argmax([np.linalg.norm(x) for x in np.array(room_anchors) - self.screen_center_xy])
        # cursor at the edge of the screen will pan camera, make sure to clip position
        zoom_point = np.clip(room_anchors[furthest_anchor_i], [40, 40], self.screen_shape_xy - 41)
        self.zoom_in(*zoom_point)
        progress_log('zoomed in, rediscovering room...')
        sleep(1.5)
        self.no_overlay_grab_screen() # will need to find room again after zoom-in
        rooms = detect_rooms(self.screen_img)
        if len(rooms) == 0:
            progress_log('Navigation: target lost')
            return False, None, None

        # find room closest to where we had cursor for zooming in
        room_bounds = rooms[np.argmin([np.linalg.norm(zoom_point - x.pos) for z, y, x in rooms])][2]

        direction = self.direction_to_screen_center(room_bounds, blocked_direction)
        while direction is not None:
            if are_opposite_directions(last_direction, direction): pan_duration /= 2

            progress_log(f'Panning now: {direction}')    
            self.pan_camera(direction, pan_duration)

            last_bounds = room_bounds

            bbox = self.get_panning_bbox(room_bounds, direction)
            start_time = perf_counter()
            progress_log(f'Finding room again...')
            sleep(0.1) # post-pan delay
            while True: # put iteration limit?
                screen_crop = self.no_overlay_grab_screen(bbox) # how can this be not assigned + the outer loop is terminated??
                debug_log_image(screen_crop, 'navigation-rescan')
                rooms = detect_rooms(screen_crop)
                if perf_counter() - start_time > 2:
                    progress_log('Panning failed: timeout')
                    return False, None, None # timeout
                if len(rooms) > 0: break

            room_bounds = rooms[0][2].offset(bbox[:2])

            # progress_log(f'New bounds: {room_bounds}')
            if last_bounds.x == room_bounds.x and last_bounds.y == room_bounds.y:
                progress_log(f'Panning blocked... `{direction}`')
                if blocked_direction is not None and blocked_direction != direction: break
                blocked_direction = direction

            last_direction = direction
            direction = self.direction_to_screen_center(room_bounds, blocked_direction)

        progress_log('Panning complete')
        if click:
            sleep(0.1) # post panning delay (camera seems to be panning a bit after a key is released)
            self.no_overlay_grab_screen()
            debug_log_image(self.screen_img, 'navigation-end-grab')
            rooms = detect_rooms(self.screen_img)
            filtered_rooms = [x for _, _, x in rooms if x.shape == room_bounds.shape]
            if len(filtered_rooms) == 0:
                progress_log('Navigation: target lost')
                return False, None, None

            room_bounds = min(filtered_rooms, key=lambda x: np.linalg.norm(x.pos - room_bounds.pos))

            for _ in range(5):
                pre_click_room = self.no_overlay_grab_screen(room_bounds.get_bbox())

                progress_log(f'Clicking: {room_bounds.x, room_bounds.y}')
                self.mouse_click(room_bounds.x, room_bounds.y)
                sleep(0.5) # click diff wait
                post_click_room = self.no_overlay_grab_screen(room_bounds.get_bbox())

                click_diff = compute_img_diff_ratio(post_click_room, pre_click_room)

                progress_log(f'Post-click pixel diff: {click_diff*100:0.2f}%')

                diff_mask = np.any(post_click_room != pre_click_room, axis=2)
                log_image(pre_click_room, 'preclick', increment_counter=False)
                log_image(post_click_room, 'postclick', increment_counter=False)
                log_image(diff_mask, 'clickdiff')

                ## !!!!
                ## TODO: not enough validation for click diff detection: add checking of diff colors
                ## !!!!

                if True:
                    temp_test_rooms = detect_rooms(pre_click_room, return_fragments=True)
                    if len(temp_test_rooms) != 1 or not is_fragment_rectangular(temp_test_rooms[0]):
                        progress_log('ALERT: room_focus assertion failed')
                        log_image(pre_click_room, 'assertion-failed', increment_counter=False)

                # room click diff threshold
                if click_diff >= 0.01: 
                    if is_room_click_diff(diff_mask):
                        return True, room_bounds, pre_click_room
                    progress_log(':: Click diff detected, flagged as false positive')
                progress_log(f'Click failed, repeat:')

            progress_log('Gave up on clicking...')
            return False, None

        return True, room_bounds, pre_click_room

    def make_critical_strike(self, crit_bounds):
        grab_size = min(crit_bounds.width, crit_bounds.height) // 3
        grab_bounds = Bounds.from_center(crit_bounds.x, crit_bounds.y, grab_size, grab_size)
        grab_bbox = grab_bounds.get_bbox()
        hit_times = []
        debug_grabs = []
    
        def get_crit_stats():
            ts = perf_counter()
            img = self.grab_screen(grab_bbox)
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
    
            if len(hit_times) >= self.crit_wait_count: break
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
    
        self.mouse_input.press(mouse.Button.left)
        sleep(0.08)
        self.mouse_input.release(mouse.Button.left)
    
        progress_log(f'Crit diff info gathered: {[f"{x:0.2f}" for x in diffs]}, avg: {avg_diff:0.2f}s, std: {std_diff:0.4f}')
    
        for img in debug_grabs: debug_log_image(img, 'crit-scan')
        sleep(1.2) # wait for crit message to disappear

    def battle_iteration(self):
        meds = detect_med_buttons(self.screen_img)
        for name, bounds in meds:
            progress_log('Clicking meds...')
            self.mouse_click(bounds.x, bounds.y)
            sleep(0.05)
    
        crits = detect_critical_button(self.screen_img)
        for bounds in crits:
            progress_log('Clicking crits...')
            self.mouse_click(bounds.x, bounds.y)
            sleep(0.3)
            self.make_critical_strike(bounds)
            sleep(0.8)
    
        return len(meds) > 0, len(crits) > 0

    def zoom_out(self):
        self.mouse_input.position = self.screen_center_xy
        init_screen = self.no_overlay_grab_screen()
        for _ in range(20):
            self.mouse_input.scroll(0, -1)
        sleep(0.15) # zoom initial delay
        new_screen = self.no_overlay_grab_screen()
        if np.sum(new_screen != init_screen) / np.prod(new_screen.shape) > 0.5:
            sleep(0.5) # zoom delay
            progress_log('Zoom-out: full delay')

    def zoom_in(self, x, y):
        # TODO: replace with `q` and `e` keys for finer zoom control (largest room should fit on screen)
        self.mouse_input.position = (x, y)
        for _ in range(5): self.mouse_input.scroll(0, 1)
        sleep(1)

    def look_for_room_using_structural(self):
        self.zoom_out()
        reached_left, reached_right = False, False
        direction = 'down' # camera actually goes up
    
        def scan_primary(directions):
            nonlocal reached_left, reached_right
            if direction in directions: 
                self.pan_camera(direction, self.camera_pan_initial_duration)
                reached_left = False
                reached_right = False
                return True
            
            return False
    
        def scan_secondary():
            if not reached_left:
                self.pan_camera('left', self.camera_pan_initial_duration)
                return True
            if not reached_right:
                self.pan_camera('right', self.camera_pan_initial_duration)
                return True
            
            return False
    
        while True:
            progress_log('Structural room scanning iteration...')
            self.no_overlay_grab_screen()
            rooms = detect_rooms(self.screen_img)
            if len(rooms) > 0: 
                progress_log(f'Structural scan found room: {rooms[0][2]}')
                return True
            
            structural, directions = detect_structural(self.screen_img)
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

    def mission_script(self):
        self.update_overlay()
        progress_log('>>> Starting mission script')
        progress_log('Looking for starting room...')
        self.script_running = True

        last_rooms = None
        last_time_room_detected = perf_counter()
        last_iteration_room_detected = True
        while True:
            progress_log('General mission iteration...')

            if perf_counter() - last_time_room_detected >= structural_scan_begin_timeout and not last_iteration_room_detected:
                progress_log('General iteration timeout, engaging structure-based scanning...')
                if not self.look_for_room_using_structural():
                    progress_log('Failed to detect new rooms, aborting execution')
                    break

            last_iteration_room_detected = False

            # new 
            in_battle = False
            new_room_discovered = False
            in_dialogue = False

            self.zoom_out() # general iteration on max zoom out for simplicity
            self.no_overlay_grab_screen() # grab screen for this iteration
            debug_log_image(self.screen_img, 'iteration-capture')
            clear_debug_canvas()
            self.battle_iteration() # take care of meds / crits (if any)

            # our next action - go to the new room, detect some..
            rooms = detect_rooms(self.screen_img)

            if len(rooms) == 0:
                progress_log('No rooms detecting, skipping iteration...')
                continue

            last_time_room_detected = perf_counter()
            last_iteration_room_detected = True

            progress_log(f'Found room: {rooms[0][2]}, type: {rooms[0][0]}')
            last_room_type = rooms[0][0]
            navigate_successful, room_bounds, pre_click_img = self.navigate_to_room(rooms[0][2], click=True)
            if not navigate_successful: continue
            progress_log(f'Room navigation successful, waiting for walk complete...')
            last_rooms = detect_rooms(self.no_overlay_grab_screen())

            # wait until the room is reached
            while True:
                start_time = perf_counter()
                current_room = self.no_overlay_grab_screen(room_bounds.get_bbox())
                diff = compute_img_diff_ratio(pre_click_img, current_room)
                debug_log_image(current_room, f'walk-wait-{diff*100:0.2f}-diff')
                progress_log(f'Waiting for walk completion: current diff {100*diff:0.2f}%')

                # min room entered diff threshold
                if diff >= room_reached_min_diff: break # room reached!
                elapsed = perf_counter() - start_time
                if elapsed < walk_wait_min_interval: sleep(walk_wait_min_interval - elapsed)

            # we reached room - wait until enemies are detected OR a new room is detected OR 4 seconds pass
            progress_log(f'Walk complete! Analyzing situation...')
            post_walk_screen_img = self.no_overlay_grab_screen()
            debug_log_image(post_walk_screen_img, f'post-walk-structural')

            proceed = False
            while not proceed:
                start_time = perf_counter()
                while perf_counter() - start_time <= room_analysis_timeout:
                    self.no_overlay_grab_screen()
                    debug_log_image(self.screen_img, f'wait-iteration')
                    have_enemies, _ = detect_enemies(self.screen_img)
                    have_crits = detect_critical_button(self.screen_img)
                    if have_enemies or have_crits:
                        in_battle = True
                        have_enemies = True
                        progress_log('Enemies detected!')
                        proceed = True # enemies, no need to stay
                        break
                    
                    rooms = detect_rooms(self.screen_img)
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
                    self.no_overlay_grab_screen()
                    debug_log_image(self.screen_img, f'structural-room-match')
                    structural, _ = detect_structural(post_walk_screen_img)
                    structural_match = np.sum(np.all(self.screen_img[structural.points[:,0], structural.points[:,1]] == structural_color, axis=1))
                    structural_match /= structural.point_count
                    # structural_rooms = detect_structural_rooms(structural)
                    # in_dialogue = not match_room_with_structural(structural_rooms, room_bounds)
                    in_dialogue = structural_match < 0.6
                    progress_log(f'Structural match: {100*structural_match:0.2f}%')
                    # central_room = next((x for x in structural_rooms if x.contains_point(*self.screen_center_xy)), None)
                    if in_dialogue: 
                        progress_log(f'Dialogue detected!')
                    elif not new_room_discovered:
                        progress_log('Structural matched, nothing to do - aborting iteration')
                        proceed = True # nothing to do - proceeding
                        continue
                    else:
                        proceed = True # unconfirmed, but new room - no need to stay

                while have_enemies:
                    self.no_overlay_grab_screen() # grab screen for this iteration
                    debug_log_image(self.screen_img, 'battle-iteration-capture')
                    had_meds, had_crits = self.battle_iteration() # take care of meds / crits
                    if had_crits: continue # critical hit marker can obscure enemies and they go undetected

                    have_enemies, enemies = detect_enemies(self.screen_img)
                    progress_log(f'Battle iteration: {len(enemies)} enemies detected')

                while in_dialogue:
                    ## !!!!
                    ## TODO: optimize dialogue waiting iteration by only analyzing central room pixels
                    ## TODO: there are `dialogs` which zoom in, but then just proceed to battle, handle that
                    ## !!!!

                    progress_log(f'Dialog: waiting...')
                    self.zoom_out()
                    self.no_overlay_grab_screen() # grab screen for this iteration
                    debug_log_image(self.screen_img, 'dialogue-iteration-capture')
                    dialogue_choice, buttons = detect_dialogue_buttons(self.screen_img, self.screen_shape_xy)
                    if dialogue_choice:
                        progress_log('Dialog choice detected!')
                        progress_log(f'Starting handler: {self.dialogue_mode}')
                        self.dialogue_handlers[self.dialogue_mode](self, buttons)
                        sleep(0.2)
                        continue

                    # structural, _ = detect_structural(self.screen_img)
                    # structural_rooms = detect_structural_rooms(structural)
                    structural_match = np.sum(np.all(self.screen_img[structural.points[:,0], structural.points[:,1]] == structural_color, axis=1))
                    structural_match /= structural.point_count
                    progress_log(f'Dialog: match {100*structural_match:0.2f}%')
                    # if not match_room_with_structural(structural_rooms, central_room):
                    if structural_match >= 0.6:
                        progress_log(f'Dialog over!')
                        break
       
            if last_room_type != 'elevator':
                progress_log(f'Proceeding to loot collection!')
                loot_bounds = detect_loot(self.no_overlay_grab_screen(), self.no_overlay_grab_screen)
                progress_log(f'Detected {len(loot_bounds)} loots')
                for loot in loot_bounds:
                    self.mouse_click(*loot.pos)
                    sleep(0.4)

            progress_log('Noting to do, proceeding to next iteration')
            continue

        progress_log('>>> Mission script complete')
        self.script_running = False

    ### 
    ### App initialization
    ###

    def __init__(self):
        self.dialogue_handlers = {
            'random': self.dialogue_random_handler,
            'manual': self.dialogue_manual_handler
        }
        
        # Entries format:  <character>: (function, name/title, repeat_execution)
        self.execution_target_map = {
            'm': (lambda x: detect_med_buttons(x.no_overlay_grab_screen()), 'meds detection', True),
            'c': (lambda x: detect_critical_button(x.no_overlay_grab_screen()), 'critical cue detection', True),
            'r': (lambda x: detect_rooms(x.no_overlay_grab_screen()), 'rooms detection', True),
            'b': (FalloutShelterAutomationApp.debug_start_battle_detect, 'battle cues detection', True),
            'p': (FalloutShelterAutomationApp.debug_start_visualizing_diff, 'diff detection', True),
            'f': (lambda x: detect_structural(x.no_overlay_grab_screen()), 'structural detection', True),
            'e': (lambda x: detect_enemies(x.no_overlay_grab_screen()), 'enemies detection', True),
            'h': (FalloutShelterAutomationApp.debug_start_detect_structural_rooms, 'structural room detection', True),
            'n': (lambda x: detect_dialogue_buttons(x.no_overlay_grab_screen(), x.screen_shape_xy), 'dialogue button detection', True),
            '/': (lambda x: detect_loot(x.no_overlay_grab_screen(), x.no_overlay_grab_screen), 'loot detection', False), # no repeat
        }
        "Maps characters on keyboard to functions to start upon chord completion"

        self.keyboard_input = keyboard.Controller()
        self.mouse_input = mouse.Controller()

    def run(self):
        print('Welcome to FSA! Initializing...')
        
        if False: # set to true to enable mock screen capture
            self.grab_screen, self.native_grab_screen = init_screen_capture(mode='mock', mock_directory='_refs/general_loot_frames2/')
            set_max_debug_frames(10)
            create_debug_frame()
            get_do()[:, :, :3] = get_mock_frames()[0]
            get_do()[:, :, 3] = 255
            create_debug_frame()
        else:
            self.grab_screen, self.native_grab_screen = init_screen_capture(mode='real', window_title='Fallout Shelter')
        
        self.screen_img = self.grab_screen(None)
        self.screen_shape = np.array(self.screen_img.shape[:2])
        self.screen_shape_xy = self.screen_shape[::-1]
        self.screen_center_xy = self.screen_shape_xy // 2
        self.state_overlay_ok = np.ones((*self.screen_shape, 4), dtype=np.uint8) * np.array([255, 0, 0, 255])
        self.state_overlay_ok[1:-1,1:-1] = 0
        self.state_overlay_error = np.ones((*self.screen_shape, 4), dtype=np.uint8) * np.array([180, 255, 0, 255])
        self.state_overlay_error[2:-2,2:-2] = 0
        self.state_overlay = self.state_overlay_ok

        create_debug_frame()

        sx, sy = self.screen_center_xy
        sw, sh = (self.screen_shape_xy * self.camera_pan_deadzone_size / 2).astype(int)
        self.camera_deadzone = Bounds(sx - sw, sy - sh, sx + sw, sy + sh)

        if self.native_grab_screen:
            self.no_overlay_grab_screen = self.no_overlay_grab_screen_native
            result_log('Native screen capture initialized')

        init_output_directory()
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-transparentcolor','#f0f0f0')
        self.root.attributes("-topmost", True)   

        log(f'\n' \
        f'Fallout Shelter Automation: {self.version}\n' \
        f'Start mission script: Ctrl + F; Enter (Esc to terminate)\n' \
        f'Toggle log display: Ctrl + F; L\n' \
        f'Shutdown: Ctrl + F; Esc \n')

        self.panel = tk.Label(self.root)
        self.log_label = tk.Label(self.root, text='Error text...', background='#000000', foreground='#44dd11', justify='left')
        self.log_label.pack(side='left')

        with Image.open("./resources/test_overlay.png") as img:
            self.update_overlay(np.array(img))
        self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_listener_thread.start()

        self.root.after(self.app_update_interval_ms, self.update)
        self.root.mainloop()

    ####
    #### Technical util functions
    ####

    def show_overlay(self):
        self.panel.place(relheight=1, relwidth=1)
        if self.show_log: self.log_label.pack(side='left')
        self.overlay_on = True

    def hide_overlay(self): 
        self.panel.place_forget()
        self.log_label.pack_forget()
        self.overlay_on = False

    def no_overlay_grab_screen(self, bbox=None):
        self.hide_overlay()
        sleep(0.06) # last tried: 0.05, sometimes overlay still gets captured
        screen = self.grab_screen(bbox)
        if bbox is None: self.screen_img = screen
        self.show_overlay()
        return screen

    def no_overlay_grab_screen_native(self, bbox=None):
        screen = self.grab_screen(bbox)
        if bbox is None: self.screen_img = screen
        return screen

    def update_overlay(self, image=0, autoshow=True):
        image_data = np.clip(image + self.state_overlay, 0, 255).astype(np.uint8)

        with io.BytesIO() as output:
            Image.fromarray(image_data).save(output, format='PNG')
            new_overlay = tk.PhotoImage(data=output.getvalue(), format='png')
            self.panel.configure(image=new_overlay)
            self.panel.image = new_overlay

        if autoshow: self.show_overlay()
    
    def display_debug(self): self.update_overlay(combined_debug_frames(), autoshow=False)
    
    def mouse_click(self, x, y):
        restore_overlay = self.overlay_on
        self.hide_overlay() # so we don't click overlay instead :)
        self.mouse_input.position = (x, y)
        sleep(0.07)
        self.mouse_input.press(mouse.Button.left)
        sleep(0.12)
        self.mouse_input.release(mouse.Button.left)
        if restore_overlay: self.show_overlay()

    def pan_camera(self, direction, duration=0.05):
        self.keyboard_input.press(CAMERA_PAN_KEYS[direction])
        sleep(duration)
        self.keyboard_input.release(CAMERA_PAN_KEYS[direction])

    ####
    #### App internals functions
    ####

    def update(self):
        if self.terminate_pending: 
            print('Terminating...')
            finish_screen_capture()
            quit()

        self.root.after(self.app_update_interval_ms, self.update)
        self.tick_counter += 1
        if self.tick_counter % 25 == 0: self.display_debug()

        self.log_label.config(text=get_current_log())
        if self.current_execution_target is not None and not self.task_in_progress:
            self.task_in_progress = True
            self.task = threading.Thread(target=self.make_async_task(self.current_execution_target), daemon=True)
            if not self.repeat_current_execution_target:
                self.current_execution_target = None

            self.task.start()

    def handle_keyboard_chord(self, key):
        self.keyboard_chord_pending = False

        if key == keyboard.Key.esc: 
            self.terminate(); 
            return
        if key == keyboard.Key.backspace:
            self.update_overlay()
            self.current_execution_target = None
            return
        if key == keyboard.Key.enter:
            self.current_execution_target = FalloutShelterAutomationApp.mission_script
            self.repeat_current_execution_target = False
            return

        if not hasattr(key, 'char'): progress_log(f'Unknown chord: {key}'); return
        if key.char == '\r': # ignore this character (idk why)
            self.keyboard_chord_pending = True
            return

        if key.char in self.execution_target_map:
            func, message, repeat = self.execution_target_map[key.char]
            if message is not None: result_log(f'Starting {message}...')
            def _exec_target(self):
                create_debug_frame()
                func(self)
                self.display_debug()

            self.current_execution_target = _exec_target
            self.repeat_current_execution_target = repeat
        elif key.char == 'l': # toggle log display
            self.show_log = not self.show_log
            if self.overlay_on:
                if self.show_log:
                    self.log_label.pack(side='left')
                else:
                    self.log_label.pack_forget()
        else: progress_log('Unknown chord...')

    def terminate(self):
        # Flag is read both by keyboard listener and app update function, ensuring both threads terminate
        self.terminate_pending = True

    def make_async_task(self, func):
        "Async tasks change overlay to indicate error, and reset `task_in_progress` flag on completion"
        def _func():
            try:
                func(self)
            except:
                print('Async task exception:')
                traceback.print_exc()
                self.state_overlay = self.state_overlay_error
                self.update_overlay()

            self.task_in_progress = False

        return _func

    def keyboard_listener(self):
        "Continuously listen to keyboard and call chord handler when appropriate. Ideally should not be modified"
        def begin_chord():
            self.keyboard_chord_pending = True
            progress_log('Waiting for chord...')

        chord_start_hotkey = keyboard.HotKey(keyboard.HotKey.parse(self.chord_start_sequence), begin_chord)

        def for_canonical(f):
            def _func(k):
                f(l.canonical(k))
    
            return _func
        
        def keyboard_on_press(key):
            if self.script_running and key == keyboard.Key.esc: 
                self.terminate() # esc during mission script should abort execution (kill-switch)
    
            self.last_key_pressed = key

            if self.keyboard_chord_pending:
                try:
                    self.handle_keyboard_chord(key) 
                except:
                    progress_log('Keyboard listener thread encountered exception:')
                    traceback.print_exc()
            
            if self.terminate_pending: return False # should be AFTER chord handler
            for_canonical(chord_start_hotkey.press)(key)

        def keyboard_on_release(key):
            for_canonical(chord_start_hotkey.release)(key)
    


        with keyboard.Listener(on_press=keyboard_on_press, on_release=keyboard_on_release) as l:
            l.join()

    ####
    #### Debug functions
    ####

    def debug_start_battle_detect(self):
        detect_critical_button(self.no_overlay_grab_screen())
        detect_med_buttons(self.no_overlay_grab_screen())
    
    def debug_start_detect_structural_rooms(self):
        structural, _ = detect_structural(self.no_overlay_grab_screen())
        clear_debug_canvas()
        detect_structural_rooms(structural)
    
    def debug_start_visualizing_diff(self):
        # self.diff_image is used by this function exclusively
        image2 = self.no_overlay_grab_screen(None)
        if not hasattr(self, 'diff_image'): self.diff_image = image2

        output = np.zeros((*image2.shape[:2], 4), dtype=int)
        diff = np.any(self.diff_image != image2, axis=2)
        output[:, :, 3] = output[:, :, 0] = diff * 255
        self.diff_image = image2

        self.update_overlay(output)
        sleep(0.2)

###
### static void main string args
###

if __name__ == '__main__':
    app = FalloutShelterAutomationApp()
    app.run()
