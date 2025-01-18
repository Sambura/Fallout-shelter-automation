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
import cv2
import itertools
import argparse

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
    mission_paused = False
    mask_escape_key = False
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

    def get_panning_bounds(self, bounds, direction):
        "Returns Bounds to be used for screen grab when panning in `direction` and keeping `bounds` in view after panning"
        if direction == 'left':
            return Bounds(bounds.x_min, bounds.y_min, self.screen_shape_xy[0] - 1, bounds.y_max)
        elif direction == 'right':
            return Bounds(0, bounds.y_min, bounds.x_max, bounds.y_max)
        elif direction == 'up':
            return Bounds(bounds.x_min, 0, bounds.x_max, bounds.y_max)
        elif direction == 'down':
            return Bounds(bounds.x_min, bounds.y_min, bounds.x_max, self.screen_shape_xy[1] - 1)

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

    def navigate_to_room(self, room_bounds: Bounds, click=True, zoom_in=True):
        progress_log(f'Navigating to the room... : {room_bounds}')

        pan_duration = self.camera_pan_initial_duration
        last_direction = None
        last_bounds = Bounds(*(np.ones(4)*(-10000000000))) # idk lol
        blocked_direction = None

        if zoom_in:
            room_anchors = room_bounds.get_corners(get_8=True)
            furthest_anchor_i = np.argmax([np.linalg.norm(x) for x in np.array(room_anchors) - self.screen_center_xy])
            zoom_point = self.filter_mouse_coords(*room_anchors[furthest_anchor_i])
            self.zoom_in(*zoom_point)
            progress_log('zoomed in, rediscovering room...')
            rooms = detect_rooms(self.latest_frame)
            if len(rooms) == 0:
                progress_log('Navigation: target lost')
                return False, None, None

            # find room closest to where we had cursor for zooming in
            vectors_norms = [[np.linalg.norm(np.array(corner) - zoom_point) for corner in z.get_corners()] for x, y, z in rooms]
            room_bounds = rooms[np.argmin([np.min(norms) for norms in vectors_norms])][2]

        direction = self.direction_to_screen_center(room_bounds, blocked_direction)
        while direction is not None:
            if are_opposite_directions(last_direction, direction): pan_duration /= 2

            progress_log(f'Panning now: {direction}')    
            self.pan_camera(direction, pan_duration)

            last_bounds = room_bounds

            pan_bounds = self.get_panning_bounds(room_bounds, direction)
            start_time = perf_counter()
            progress_log(f'Finding room again...')
            sleep(0.1) # post-pan delay
            while True: # put iteration limit?
                screen_crop = self.latest_frame[pan_bounds.to_slice()] # how can this be not assigned + the outer loop is terminated??
                debug_log_image(screen_crop, 'navigation-rescan')
                rooms = detect_rooms(screen_crop)
                if perf_counter() - start_time > 2:
                    progress_log('Panning failed: timeout')
                    return False, None, None # timeout
                if len(rooms) > 0: break

            room_bounds = rooms[0][2].offset(pan_bounds.low_pos)

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
            debug_log_image(self.freeze_frame(), 'navigation-end-grab')
            rooms = detect_rooms(self.fixed_frame)
            filtered_rooms = [x for _, _, x in rooms if x.shape == room_bounds.shape]
            if len(filtered_rooms) == 0:
                progress_log('Navigation: target lost')
                return False, None, None

            room_bounds = min(filtered_rooms, key=lambda x: np.linalg.norm(x.pos - room_bounds.pos))

            for _ in range(5):
                pre_click_room = self.latest_frame[room_bounds.to_slice()]

                progress_log(f'Clicking: {room_bounds.x, room_bounds.y}')
                self.mouse_click(room_bounds.x, room_bounds.y)
                sleep(0.5) # click diff wait
                post_click_room = self.latest_frame[room_bounds.to_slice()]

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

        progress_log(f'Waiting for data...')
        sleep(1.5)
        had_overlay = self.overlay_on
        self.hide_overlay()
        sleep(0.5)
        captured_frames = self.copy_frames_buffer()
        hit_timings = []

        last_i = -2 # does not have to be -2, any negative (except -1) will do the same
        for i, (frame, abs_time, delta) in enumerate(captured_frames):
            crop = crop_image(frame, grab_bbox)
            crit_pixels = np.sum(match_color_exact(crop, critical_hit_color))
            if crit_pixels >= min_critical_pixels:
                if last_i + 1 == i: # if several in a row: take average (yes its not ideal for more than 2 frames)
                    hit_timings[-1] = (hit_timings[-1] + abs_time) / 2
                else:
                    hit_timings.append(abs_time)
                last_i = i

        if len(hit_timings) < 2:
            progress_log('Crit failed...')
            if had_overlay: self.show_overlay()
            return

        diffs = np.diff(hit_timings)
        avg_diff = np.mean(diffs) # interval between crits
        std_diff = np.std(diffs)
        next_crit = hit_timings[-1]
        for _ in range(10): # limit the iteration number
            if next_crit < perf_counter(): 
                next_crit += avg_diff

        next_crit -= 0.03 # bias
        while perf_counter() - 3 * std_diff < next_crit: sleep(0.0001)
    
        self.mouse_input.press(mouse.Button.left)
        sleep(0.08)
        self.mouse_input.release(mouse.Button.left)
    
        progress_log(f'Crit diff info gathered: {[f"{x:0.2f}" for x in diffs]}, avg: {avg_diff:0.2f}s, std: {std_diff:0.4f}')

        if had_overlay: self.show_overlay()
        sleep(2) # wait for crit message to disappear

    def battle_iteration(self):
        "Freeze frame before calling"
        meds = detect_med_buttons(self.fixed_frame)
        for name, bounds in meds:
            progress_log('Clicking meds...')
            self.mouse_click(bounds.x, bounds.y)
            sleep(0.05)
    
        crits = detect_critical_button(self.fixed_frame)
        for bounds in crits:
            progress_log('Clicking crits...')
            self.mouse_click(bounds.x, bounds.y)
            sleep(0.3)
            self.make_critical_strike(bounds)
    
        return len(meds) > 0, len(crits) > 0

    def zoom_out(self):
        full_delay_threshold = 0.2
        self.mouse_input.position = self.screen_center_xy
        init_screen = self.latest_frame
        for _ in range(20): self.mouse_input.scroll(0, -1)
        sleep(0.15) # zoom initial delay
        diff_value = np.sum(self.latest_frame != init_screen) / np.prod(init_screen.shape)
        if diff_value > full_delay_threshold:
            sleep(1) # zoom delay
            progress_log('Zoom-out: full delay')
        self.zoomed_out = True

    def zoom_in(self, x, y):
        "Zooms in while focusing on specified point. After zoom-in a 3-slot room should still fit on screen + a bit of extra space around"
        zoom_in_duration = 0.75
        self.mouse_input.position = self.filter_mouse_coords(x, y)
        self.keyboard_input.press('e')
        sleep(zoom_in_duration)
        self.keyboard_input.release('e')
        sleep(0.1)
        self.zoomed_out = False

    def look_for_room_using_structural(self):
        "Returns true if at least one unvisited room has been found and is on screen right now"
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
            rooms = detect_rooms(self.freeze_frame())
            if len(rooms) > 0: 
                progress_log(f'Structural scan found room: {rooms[0][2]}')
                return True
            
            structural, directions = detect_structural(self.fixed_frame)
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

    def estimate_movement_direction(self, frames):
        if len(frames) < 3: # realistically should not happen
            result_log(f'Could not estimate movement: {len(frames)}/{len(self.screen_frames)}')
            return None
        else:
            result_log(f'Estimating movement: @ {len(frames)}')

        motion_threshold = 15
        diff_maps = [compute_motion_diff(x[0], y[0], z[0], diff_threshold=motion_threshold) for x, y, z in zip(frames, frames[1:], frames[2:])]
        cumulative_map = (np.mean(diff_maps, axis=0) * 255).astype(np.uint8)

        for i, diff_map in enumerate(diff_maps):
            debug_log_image((diff_map * 255).astype(np.uint8), f'motion-diff-map-f-{i}', increment_counter=(i==len(diff_maps)-1))

        rg_cmp_map = cumulative_map[:,:,0] == cumulative_map[:,:,1]
        gb_cmp_map = cumulative_map[:,:,1] == cumulative_map[:,:,2]
        rb_cmp_map = cumulative_map[:,:,0] == cumulative_map[:,:,2]
        neutral_map = rg_cmp_map | gb_cmp_map | rb_cmp_map
        cumulative_map[neutral_map] = 0

        centroids = [np.argwhere(x).mean(axis=0).astype(int)[::-1] for x in [np.squeeze(x) for x in np.split(cumulative_map, 3, axis=2)]]

        draw_arrow_line(get_do(), centroids[0], centroids[2], np.array([255, 90, 90, 255]), width=3, arrow_abs_length=100)
        draw_arrow_line(get_do(), centroids[2], centroids[1], np.array([90, 255, 90, 255]), width=3, arrow_abs_length=100)

        direction_value = centroids[1][0] - centroids[0][0]
        result_log(f'Direction value : {direction_value:0.4f}')
        if abs(direction_value) < 0.1: result_log('Warning: direction value too small')
        return 'left' if direction_value < 0 else 'right'

    def room_scan_state(self):
        "Finds and select the next room to go to, or terminates script if all rooms are visited"
        progress_log('General mission iteration...')

        self.hide_character_panel() # always good to remove that
        # if we didn't see a new room in a while, start structural scan
        if perf_counter() - self.last_time_room_detected >= structural_scan_begin_timeout and not self.last_iteration_room_detected:
            progress_log('General iteration timeout, engaging structure-based scanning...')
            if not self.look_for_room_using_structural():
                progress_log('Failed to detect new rooms, aborting execution')
                self.script_complete = True # terminate mission script
                return

        self.last_iteration_room_detected = False
        if self.need_zoom_out and not self.zoomed_out:
            self.zoom_out()
        debug_log_image(self.freeze_frame(), 'iteration-capture')
        self.battle_iteration() # meds / level-ups should be clicked

        rooms = detect_rooms(self.fixed_frame) # scan for unvisited rooms
        if len(rooms) == 0:
            progress_log('No rooms detected, retrying...')
            self.need_zoom_out = True
            return

        self.last_time_room_detected = perf_counter()
        self.last_iteration_room_detected = True
        self.need_zoom_out = False

        self.target_room = rooms[0] # room we will navigate to next
        progress_log(f'Found room: {self.target_room[2]}, type: {self.target_room[0]}')
        self.script_state = self.room_navigation_state

    def room_navigation_state(self):
        "Pans camera over to target room, clicks, and waits until room is reached"
        _, _, target_bounds = self.target_room
        navigate_successful, self.current_room_bounds, pre_click_img = self.navigate_to_room(target_bounds, click=True, zoom_in=self.zoomed_out)
        if not navigate_successful:
            self.script_state = self.room_scan_state
            return

        progress_log(f'Room navigation successful, waiting for walk complete...')
        # this is used to later detect if any new rooms appear after we reach the room
        self.last_detected_rooms = detect_rooms(self.next_frame())
        self.current_room_type = get_room_type(self.current_room_bounds, 'full')

        # wait until the room is reached
        for _ in slow_loop(interval=walk_wait_min_interval):
            start_time = perf_counter()
            current_room = self.latest_frame[self.current_room_bounds.to_slice()]
            diff = compute_img_diff_ratio(pre_click_img, current_room)
            debug_log_image(current_room, f'walk-wait-{diff*100:0.2f}-diff')
            progress_log(f'Waiting for walk completion: current diff {100*diff:0.2f}%')

            if diff >= room_reached_min_diff: break # room reached!

        progress_log(f'Walk complete! Analyzing situation...') # room reached, wrap up and switch state
        self.post_walk_screen_img = self.latest_frame
        debug_log_image(self.post_walk_screen_img, f'post-walk-structural')

        self.script_state = self.room_analysis_state

    def structural_detection_task(self, delay=0):
        sleep(delay)
        frame = self.latest_frame # do not use freeze frame from async tasks
        structural, _ = detect_structural(frame)
        structural_rooms = detect_structural_rooms(structural)

        loot_bounds = None
        if len(structural_rooms) > 0:
            central_rooms = [room for room in structural_rooms if room.contains_point(*self.screen_center_xy)]
            if len(central_rooms) > 0:
                debug_log_image(frame, 'structural-analysis')
                loot_bounds = central_rooms[0]
        
        if loot_bounds is None:
            loot_bounds = self.screen_bounds
            debug_log_image(frame, 'structural-analysis-no-central')

        self.current_room_bounds = loot_bounds
        self.loot_bound_detected = True

    # some notes for reference
    #  - if the room is empty, the adjacent rooms are revealed pretty much immediately as the first dweller enters it
    #  - if the room has enemies, the fight (and enemy healthbar) only starts/appear once ALL dwellers reach the room
    #  - same applies to dialogue - only starts when all dwellers reach the room. The dialogue zoom also happens with same rules
    #   + dialogue *may* transition to fight with no dialogue options to pick from
    #   + dialogue ALSO may have neither fight nor choices
    #   + when in dialog, UI buttons disappear!
    #   + dialogue can be skipped by clicking on screen
    #  - (i think) dialogue *with* options *may* also transition to a fight
    #  - fun fact: if your dweller never took damage, their healthbars are hidden (unless in fight)
    def room_analysis_state(self):
        "After a new room has been reached, we need to figure out what to do next. This happens here"
        room_entry_frames = self.copy_frames_buffer(min_frame_delta=0.25) # get frames of how we enter (for later)
        if self.current_room_type == 'elevator': # probably nothing interesting in an elevator, skip
            result_log('At elevator: skip analysis')
            self.script_state = self.room_scan_state
            return

        self.next_frame() # wait a bit for room to appear
        debug_log_image(self.freeze_frame(), "room-analysis-start")
        self.early_loot_coords = None
        self.enemies_detected = False
        self.dialogue_detected = False
        self.loot_bound_detected = True

        # first thing we do - check if there is a new room appeared. If so, collect loot in current room
        new_rooms = detect_rooms(self.fixed_frame)
        if has_new_room(new_rooms, self.last_detected_rooms):
            progress_log('Room analysis: new room')
            self.script_state = self.loot_collection_state
            return

        # No new room detected, must be something else.
        # probably the next best thing to do is figure out where dwellers came from into the room
        # (ended up not using direction (and it is not accurate whatsoever))
        # entry_direction = self.estimate_movement_direction(room_entry_frames) # ~1 second? idk

        # the enemies can appear at any point, just look for them in background
        self.run_enemy_detection = True
        def enemy_detection_task():
            for _ in slow_loop(interval=0.5): # scan at most twice a second
                self.enemies_detected, self.current_enemies = detect_enemies(self.latest_frame)
                if not self.run_enemy_detection or self.enemies_detected: 
                    if self.enemies_detected: progress_log('Interrupting: enemies detected')
                    break

        enemy_detection_thread = threading.Thread(target=enemy_detection_task)
        enemy_detection_thread.start()
        self.hide_character_panel() # why not

        # I guess we just wait for dialog to start or whatever
        def dialog_detection_task():
            for _ in slow_loop(interval=0.5, max_duration=10):
                if not self.run_dialogue_detection:
                    progress_log(f'Stopped dialogue detection')
                    break
                
                frame = self.latest_frame
                if not detect_ui(frame):
                    progress_log(f'Dialogue detected')
                    debug_log_image(frame, 'dialog-detection-frame')
                    self.dialogue_detected = True
                    break
                    
        progress_log(f'Detecting dialogue...')
        self.run_dialogue_detection = True
        dialogue_detection_thread = threading.Thread(target=dialog_detection_task)
        dialogue_detection_thread.start()

        # we start loot detection as soon as structural is detected, wait for now...
        loot_detection_thread = None
        def loot_detection_task():
            sleep(0.3) # wait for room fade-in
            result_log('>> background loot detection')
            frames = [x[0] for x in self.copy_frames_buffer(min_frame_delta=0.5)[-2:]]
            for i, frame in enumerate(frames):
                debug_log_image(frame, f'precapture-loot-detection-frame-{i}')
            self.early_loot_coords = detect_loot(self.current_room_bounds, frames=frames)
            result_log('<< background loot detection')

        # ~~ EveNT lOoP ~~
        progress_log(f'Starting event loop...')
        loop_start = perf_counter()
        for _ in slow_loop(interval=0.1):
            if self.enemies_detected:
                self.run_dialogue_detection = False
                self.script_state = self.battle_state
                return
            if self.dialogue_detected: # leave enemy detection on
                self.script_state = self.dialogue_state
                # bounds change when entering the dialogue
                self.loot_bound_detected = False
                progress_log(f'Detecting structural...')
                structural_detection_thread = threading.Thread(target=lambda: self.structural_detection_task(delay=1.5))
                structural_detection_thread.start()
                return
            if self.loot_bound_detected and loot_detection_thread is None:
                loot_detection_thread = threading.Thread(target=loot_detection_task)
                loot_detection_thread.start()

            if self.early_loot_coords is not None and perf_counter() - loop_start > 6.5:
                progress_log(f'Event loop timeout, proceed to looting')
                break # wait up to 6.5 seconds for something, otherwise proceed to looting
        
        # stop threads and switch state
        self.run_enemy_detection = False
        self.run_dialogue_detection = False
        self.script_state = self.loot_collection_state

    def dialogue_state(self):
        result_log('Dialog: waiting...')
        dialogue_complete = False
        for _ in slow_loop(interval=0.2):
            dialogue_choice, buttons = detect_dialogue_buttons(self.freeze_frame(), self.screen_shape_xy)
            # debug_log_image(self.fixed_frame, 'dialogue-iteration-capture')
            if dialogue_choice:
                progress_log('Dialog choice detected!')
                progress_log(f'Starting handler: {self.dialogue_mode}')
                debug_log_image(self.fixed_frame, 'dialogue-choice-capture')
                self.dialogue_handlers[self.dialogue_mode](buttons)
            if self.enemies_detected:
                self.script_state = self.battle_state
                return
            if detect_ui(self.fixed_frame):
                progress_log('Dialogue finished')
                self.script_state = self.loot_collection_state
                self.run_enemy_detection = False # stop enemy detection now
                return
            else: # mouse click to skip dialogue faster!
                self.mouse_click(*self.screen_center_xy)

    def battle_state(self):
        have_enemies = True
        while have_enemies:
            debug_log_image(self.freeze_frame(), 'battle-iteration-capture')
            had_meds, had_crits = self.battle_iteration() # take care of meds / crits
            if had_crits: 
                self.hide_character_panel()
                continue # critical hit marker can obscure enemies and they go undetected

            have_enemies, enemies = detect_enemies(self.fixed_frame)
            progress_log(f'BIT: {len(enemies)} enemies detected')
        
        sleep(2.5) # wait for enemy death animation (idk how accurate is this number)
        self.script_state = self.loot_collection_state

    def hide_character_panel(self):
        if detect_character_panel(self.latest_frame):
            self.mask_escape_key = True
            self.keyboard_input.press(keyboard.Key.esc)
            sleep(0.03)
            self.keyboard_input.release(keyboard.Key.esc)
            self.mask_escape_key = False
            sleep(0.2) # character panel hide animation (is probably longer)

    def loot_collection_state(self):
        progress_log(f'Proceeding to loot collection!')
        scan_attempts = 2

        if self.dialogue_detected or self.enemies_detected:
            scan_attempts += 1

        draw_border(get_do(), self.current_room_bounds, np.array([255, 0, 0, 255]), thickness=5)
        for attempt in range(scan_attempts):
            progress_log(f'Scan attempt: #{attempt + 1}')
            self.hide_character_panel()

            if self.early_loot_coords:
                progress_log(f'Using precomputed loot data...')
                loot_coords = self.early_loot_coords
                self.early_loot_coords = None # do not reuse them twice!
            else:
                loot_coords = detect_loot(self.current_room_bounds, lambda bbox: crop_image(self.latest_frame, bbox))

            progress_log(f'Detected {len(loot_coords)} loots')
            for loot in loot_coords:
                self.mouse_click(*loot)
                sleep(0.35)
            sleep(2.5) # loot collection animations should finish
        
        self.script_state = self.room_scan_state
    
    def mission_script(self):
        progress_log('>>> Starting mission script')

        # state initialization
        self.script_running = True                      # is game automation script running now
        self.script_complete = False                    # is script complete and should be terminated? (usually inverse of script_running)
        self.last_time_room_detected = perf_counter()   # last timestamp (seconds) when an unvisited room was detected
        self.last_iteration_room_detected = False       # has an unvisited room been detected on previous iteration?
        self.script_state = self.room_scan_state        # current state function of the mission script
        self.mission_paused = False

        progress_log('>>> Starting capture thread')
        self.start_capture_thread()

        self.dialogue_detected = True # don't question it
        self.zoomed_out = False
        self.need_zoom_out = False

        progress_log('>>> Mission loop starts')
        while not self.script_complete:
            if self.mission_paused:
                progress_log('>>> Script is paused now')
                while self.mission_paused: sleep(0.1)
                progress_log('>>> Resuming the mission script')
                
            self.script_state()

        progress_log('>>> Mission script complete')

    def toggle_mission_pause(self):
        self.mission_paused = not self.mission_paused

    def start_capture_thread(self):
        if self.capture_thread is not None: return
        self.screen_frames_lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self.capture_worker)
        self.capture_thread.start()
        while not hasattr(self, 'latest_frame'): pass # get capture thread started

    def capture_worker(self):
        "to be launched in a separate thread - continually captures the screen"
        self.screen_frames = [] # (frame, abs_time, delta_time)
        last_timestamp = perf_counter()

        while True:
            timestamp = perf_counter()
            self.latest_frame = self.no_overlay_grab_screen()
            with self.screen_frames_lock:
                self.screen_frames.append((self.latest_frame, timestamp, timestamp - last_timestamp))
                while len(self.screen_frames) > self.max_screen_frames:
                    del self.screen_frames[0]
                while timestamp - self.screen_frames[0][1] > self.screen_frame_max_age:
                    del self.screen_frames[0]
            last_timestamp = perf_counter()
            if self.capture_thread is None: return

    def freeze_frame(self):
        "Captures the screen into self.fixed_frame and returns it"
        self.fixed_frame = self.latest_frame
        return self.fixed_frame

    def next_frame(self):
        img = self.latest_frame
        while img is self.latest_frame: pass
        return self.latest_frame

    def copy_frames_buffer(self, min_frame_delta=0, max_total_duration=-1):
        "Copies the buffer of captured screen frames for later use. Specify min_frame_delta to reduce number of returned frames"
        self.screen_frames_lock.acquire()
        buffer_copy = self.screen_frames[:]
        self.screen_frames_lock.release()

        # TODO : this should go from last frame to first (so that the first frame is always retained)
        i = 1
        while i < len(buffer_copy):
            while i < len(buffer_copy) and buffer_copy[i][1] - buffer_copy[i - 1][1] < min_frame_delta:
                del buffer_copy[i]
            i += 1

        if max_total_duration < 0:
            return buffer_copy

        end_time = buffer_copy[-1]
        for i in range(len(buffer_copy)):
            if end_time[1] - buffer_copy[i][1] <= max_total_duration:
                return buffer_copy[i:]
        
        raise Exception('I think this should not be reached? idk')

    ### 
    ### App initialization
    ###

    def __init__(self, force_pil_capture=False, mock_frames_path=None):
        self.dialogue_handlers = {
            'random': self.dialogue_random_handler,
            'manual': self.dialogue_manual_handler
        }
        
        # Entries format:  <character>: (function, name/title, repeat_execution, [no_new_thread])
        self.execution_target_map = {
            '`': (lambda x: x.zoom_in(500, 500), 'temp debug function', False), # replace with whatever you need at the time
            'm': (lambda x: detect_med_buttons(x.no_overlay_grab_screen()), 'meds detection', True),
            'c': (lambda x: detect_critical_button(x.no_overlay_grab_screen()), 'critical cue detection', True),
            'r': (lambda x: detect_rooms(x.no_overlay_grab_screen()), 'rooms detection', True),
            'b': (FalloutShelterAutomationApp.debug_start_battle_detect, 'battle cues detection', True),
            'p': (FalloutShelterAutomationApp.debug_start_visualizing_diff, 'diff detection', True),
            'f': (lambda x: detect_structural(x.no_overlay_grab_screen()), 'structural detection', True),
            'e': (lambda x: detect_enemies(x.no_overlay_grab_screen()), 'enemies detection', True),
            'h': (FalloutShelterAutomationApp.debug_start_detect_structural_rooms, 'structural room detection', False), # return
            'n': (lambda x: detect_dialogue_buttons(x.no_overlay_grab_screen(), x.screen_shape_xy), 'dialogue button detection', True),
            '/': (lambda x: new_detect_loot_general_debug(x.screen_bounds, x.no_overlay_grab_screen), 'loot detection', False), # no repeat
            't': (FalloutShelterAutomationApp.debug_start_visualizing_motion_diff, 'motion diff detection', True),
            '.': (FalloutShelterAutomationApp.debug_toggle_capture_thread, 'thread toggle', False),
            '+': (FalloutShelterAutomationApp.debug_dump_motion_diff_data, 'motion data dump', False),
            '-': (FalloutShelterAutomationApp.debug_test_iou, 'iou test', True),
            '=': (FalloutShelterAutomationApp.toggle_mission_pause, 'Pause/resume mission script', False, True)
        }
        "Maps characters on keyboard to functions to start upon chord completion"

        self.keyboard_input = keyboard.Controller()
        self.mouse_input = mouse.Controller()

        self.max_screen_frames = 250                     # kind of arbitrary (~1500 MB for 1920x1080 frames)
        self.screen_frame_max_age = 3                    # (seconds) (probably can be reduced, 3 seconds for debug reasons)
        self.capture_thread = None
        self.force_pil_capture = force_pil_capture
        self.mock_frames_path = mock_frames_path

    def run(self):
        print('Welcome to FSA! Initializing...')
        
        native_capture = not self.force_pil_capture
        if self.mock_frames_path is not None: # set to true to enable mock screen capture
            self.grab_screen, self.native_grab_screen = init_screen_capture(mode='mock', mock_directory=self.mock_frames_path)
            self.screen_shape = np.array(get_mock_frames()[0].shape[:2])
            override_debug_frame_shape(self.screen_shape)
            set_max_debug_frames(10)
            create_debug_frame()
            get_do()[:, :, :3] = get_mock_frames()[0]
            get_do()[:, :, 3] = 255
            create_debug_frame()
        else:
            self.grab_screen, self.native_grab_screen = init_screen_capture(mode='real', window_title='Fallout Shelter', use_native=native_capture)
            self.screen_shape = np.array(self.grab_screen(None).shape[:2])
            override_debug_frame_shape(self.screen_shape)
        
        self.screen_shape_xy = self.screen_shape[::-1]
        progress_log(f'Using screen shape: {self.screen_shape_xy}')
        self.screen_center_xy = self.screen_shape_xy // 2
        self.screen_bounds = Bounds.from_rect(0, 0, *self.screen_shape_xy)
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
            self.no_overlay_grab_screen = self.grab_screen
            result_log('Native screen capture initialized')
        elif not native_capture:
            result_log('Forced PIL screen capture')

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
            self.update_overlay(np.array(img.resize(self.screen_shape_xy)))
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
        self.show_overlay()
        return screen

    def no_overlay_grab_screen_native(self, bbox=None):
        screen = self.grab_screen(bbox)
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
    
    def filter_mouse_coords(self, x, y, bool_mode=False):
        """Filters coordinates for mouse action to prevent mouse getting close to screen edges,
        which triggers camera movement. Use bool_mode to return whether the coords *should* be
        filtered rather than returning new coords
        """
        rel_border_thickness = 0.03
        dw = int(self.screen_shape_xy[0] * rel_border_thickness)
        dh = int(self.screen_shape_xy[1] * rel_border_thickness)

        filtered_x = max(min(x, self.screen_shape_xy[0] - dw - 1), dw)
        filtered_y = max(min(y, self.screen_shape_xy[1] - dh - 1), dh)

        if bool_mode:
            return filtered_x != x or filtered_y != y

        return filtered_x, filtered_y

    def mouse_click(self, x, y, protect_borders=True):
        if protect_borders and self.filter_mouse_coords(x, y, bool_mode=True):
            result_log(f'Warning: click at {x}, {y} aborted')
            return

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
            if len(self.execution_target_map[key.char]) == 4 and self.execution_target_map[key.char][3]:
                func, message, repeat, _ = self.execution_target_map[key.char]
                if message is not None: result_log(f'Executing {message}...')
                create_debug_frame()
                func(self)
                self.display_debug()
                return

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
        self.capture_thread = None # this stops the capture thread
        sleep(0.05) # wait for thread to die
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
            if self.script_running and key == keyboard.Key.esc and not self.mask_escape_key: 
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

    def debug_toggle_capture_thread(self):
        if self.capture_thread is None:
            self.start_capture_thread()
            result_log("Started capture thread")
        else:
            self.capture_thread = None
            result_log("Stopping capture thread...")

    def debug_start_visualizing_motion_diff(self):
        # self.debug_motion_frames is used by this function exclusively
        curr_frame = self.no_overlay_grab_screen()
        if not hasattr(self, 'debug_motion_frames'): self.debug_motion_frames = []
        self.debug_motion_frames.append(curr_frame)
        while len(self.debug_motion_frames) > 3:
            del self.debug_motion_frames[0]

        if len(self.debug_motion_frames) < 3: return

        diff_map = compute_motion_diff(*self.debug_motion_frames)
        self.update_overlay(np.dstack((255 * diff_map, 255 * np.any(diff_map > 0, axis=2))))

    def debug_dump_motion_diff_data(self):
        if not hasattr(self, 'screen_frames'):
            result_log('Failure: no frames captured')
            return

        self.estimate_movement_direction(self.copy_frames_buffer(min_frame_delta=0.25, max_total_duration=2))
        return

        frames = self.copy_frames_buffer(min_frame_delta=0.25, max_total_duration=2)

        if len(frames) < 3:
            result_log('Failure: need at least 3 frames')
            return
        else:
            result_log(f'Ok: processing {len(frames)} frames...')

        diff_maps = [compute_motion_diff(x[0], y[0], z[0], diff_threshold=15) for x, y, z in zip(frames, frames[1:], frames[2:])]
        debug_log_image((diff_maps[0] * 255).astype(np.uint8), 'first-motion-diff')
        cumulative_map = (np.mean(diff_maps, axis=0) * 255).astype(np.uint8)

        rg_cmp_map = cumulative_map[:,:,0] == cumulative_map[:,:,1]
        gb_cmp_map = cumulative_map[:,:,1] == cumulative_map[:,:,2]
        rb_cmp_map = cumulative_map[:,:,0] == cumulative_map[:,:,2]
        neutral_map = rg_cmp_map | gb_cmp_map | rb_cmp_map
        cumulative_map[neutral_map] = 0

        centroids = [np.argwhere(x).mean(axis=0).astype(int)[::-1] for x in [np.squeeze(x) for x in np.split(cumulative_map, 3, axis=2)]]

        for i in itertools.count(start=1):
            df = get_debug_frame(-i)
            if df is None: break
            df //= 2

        draw_arrow_line(get_do(), centroids[0], centroids[2], np.array([255, 90, 90, 255]), width=3, arrow_abs_length=100)
        draw_arrow_line(get_do(), centroids[2], centroids[1], np.array([90, 255, 90, 255]), width=3, arrow_abs_length=100)

        draw_point(cumulative_map, centroids[0], 7, np.array([255, 0, 0]), outline_color=np.array([200, 200, 200]))
        draw_point(cumulative_map, centroids[1], 15, np.array([0, 255, 0]), outline_color=np.array([200, 200, 200]), outline_width=5)
        draw_point(cumulative_map, centroids[2], 10, np.array([0, 0, 255]), outline_color=np.array([200, 200, 200]))
        draw_arrow_line(cumulative_map, centroids[0], centroids[2], np.array([180, 180, 180]), width=4, arrow_abs_length=100)
        draw_arrow_line(cumulative_map, centroids[2], centroids[1], np.array([180, 180, 180]), width=4, arrow_abs_length=100)
        debug_log_image(cumulative_map, 'cum-motion-map')
        result_log(f'Motion diff dump success')

        # return
        # # video dump version
        # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # path = make_output_filename(f'motion-diff-dump.avi')
        # video_writer = cv2.VideoWriter(path, fourcc, fps=30, frameSize=(frames[0][0].shape[1], frames[0][0].shape[0]))
        # video_writer.set(cv2.CAP_PROP_BITRATE, 50000000)
        # for x, y, z in zip(frames, frames[1:], frames[2:]):
        #     diff_map = compute_motion_diff(x[0], y[0], z[0])
        #     maps = [np.squeeze(x) for x in np.split(diff_map, 3, axis=2)]
        #     # fragments = []
        #     # for diff_channel in maps:
        #     #     diff_fragments, _, _, mask = detect_fragments(diff_channel, diff_channel)
        #     #     for fragment in diff_fragments:
        #     #         if fragment.point_count < 150:
        #     #             fragment.compute(patch_mask=True)
        #     #             diff_channel[fragment.fragment_value == mask] = 0
        #     #             continue
        #     #         # filter out too large fragments
        #     #         # !! not the same as fragment.are_smaller_than(... (both sides larger vs one of them)
        #     #         if not Bounds.from_rect(0, 0, 400, 400).are_smaller_than(*fragment.bounds.shape):
        #     #             continue
        #     #         
        #     #         fragment.compute(patch_mask=True)
        #     #         diff_channel[fragment.fragment_value == mask] = 0
        #     #         result_log(f'Filtered out fragment: {fragment.bounds}')
        #     
        #     centroids = [np.argwhere(x).mean(axis=0).astype(int)[::-1] for x in maps]
        #     
        #     diff_img = (255 * diff_map).astype(np.uint8)
        #     draw_line(diff_img, centroids[0], centroids[2], np.array([110, 110, 110]), width=5)
        #     draw_line(diff_img, centroids[2], centroids[1], np.array([110, 110, 110]), width=5)
        #     draw_point(diff_img, centroids[0], 7, np.array([255, 0, 0]), outline_color=np.array([200, 200, 200]))
        #     draw_point(diff_img, centroids[1], 15, np.array([0, 255, 0]), outline_color=np.array([200, 200, 200]), outline_width=5)
        #     draw_point(diff_img, centroids[2], 10, np.array([0, 0, 255]), outline_color=np.array([200, 200, 200]))
        #     video_writer.write(diff_img[:,:,::-1])
        # result_log(f'Motion diff dump success')

    def debug_test_iou(self):
        bounds1 = Bounds.from_center(*self.screen_center_xy, 300, 200)
        draw_border(get_do(), bounds1, np.array([0, 0, 0, 255]), 1)
        img = self.no_overlay_grab_screen()
        mask = np.all(img == np.array([0, 255, 0]), axis=2)
        fragments, _, _, _ = detect_fragments(mask, mask)
        bounds2 = None
        last_size = 0
        for frag in fragments:
            if last_size > frag.point_count: continue
            bounds2 = frag.bounds
            last_size = frag.point_count
            
        if bounds2 is None: return
        draw_border(get_do(), bounds2, np.array([255, 0, 0, 255]), 1)
        iou1 = bounds1.get_iou(bounds2)
        iou2 = bounds2.get_iou(bounds1)
        result_log(f'IoU: {100*iou1:0.2f}/{100*iou2:0.2f} : {iou1 == iou2}')

###
### static void main string args
###

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fallout shelter automation')
    parser.add_argument('--force-pil-capture', action='store_true', help='Force screen capture using PIL (no native window capturing)')
    parser.add_argument('--mock-frames-path', type=str, help='Specify directory for mock frames (also enables mock screen capture)')
    args = parser.parse_args()

    app = FalloutShelterAutomationApp(force_pil_capture=args.force_pil_capture, mock_frames_path=args.mock_frames_path)
    app.run()
