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
import psutil

def visualize_fragment_grouping(fragments_mask, radius=40):
    create_debug_frame()
    fragments, _, _ = detect_fragments(fragments_mask, fragments_mask)

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
DIRECTION_MAP_XY = { 'left': np.array([-1, 0]), 'right': np.array([1, 0]), 'down': np.array([0, -1]), 'up': np.array([0, 1]) }

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

def get_panning_bounds(bounds, direction, container_bounds):
    if direction == 'left':
        return Bounds(bounds.x_min, bounds.y_min, container_bounds.x_max, bounds.y_max)
    elif direction == 'right':
        return Bounds(container_bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)
    elif direction == 'up':
        return Bounds(bounds.x_min, container_bounds.y_min, bounds.x_max, bounds.y_max)
    elif direction == 'down':
        return Bounds(bounds.x_min, bounds.y_min, bounds.x_max, container_bounds.y_max)

# TODO's list:
# 
# would be cool:
#   - make camera pan speed adaptive (traveled distance vs expected, adjust velocity)
#       + This is likely only possible if the room navigation is reworked (fullscreen analysis / pan frames analysis)
#       + Otherwise it is very hard to determine the reason of camera velocity mismatch as well as accurately keep track of the target room
#
#   - [might not be needed] modify loot detection: if a loot is consistently detected at a location but is failed to be collected, increase number of attempts, assuming loot detection is correct
#
#   - add a new chord starter for debug functions (?????)
#   - add auto capture check that verifies that native capture returns image that is not fully zeroed out
#
#  big things:
#   - detect both enemy and ally health/rad levels. detect enemy boss icon
#   - improve loot detection/collection: if no loot is detected, pan camera to look at a different angle
#   - look into tracking camera location at all times, building an internal map of the scene to gain better understanding of the game
#
# bugs(?):
#   - after dialogue / battle, look for `mission complete` pop up to avoid accidentally clicking on it (may be unnecessary if loot will not be searched in top half) 
#        - fun fact, you can also know the mission is complete by looking at topright button - there will be an exclamation mark if mission is complete
#   - note: `objective complete` popups are not clickable, no need to look for them i think
#
#   - fix the thing where program doesn't close if game is closed (windows native capture) 
#   

class GameRoom:
    def __init__(self, level, detection_type, bounds, limits, y, room_height):
        self.visited = False # this is set elsewhere
        self.discovered = detection_type == 'unvisited'
        self.level = level
        self.bounds = bounds
        self.unknown_directions = limits
        self.full_bounds = limits == []
        self.room_type = get_room_type(bounds, 'full' if self.full_bounds else '')
        self.level_y = int(y)
        self.room_target_height = room_height
        self.pos = None
        self.estimate_width()
        self.align_bounds()

    def estimate_width(self):
        ratio = self.bounds.width / self.bounds.height + 0.17
        self.width = round(3 * ratio / 1.75)
    
    def align_bounds(self):
        x0, x1 = self.bounds.x_min, self.bounds.x_max
        hrw = self.room_target_height // 2
        self.bounds = Bounds(x0, self.level_y - hrw, x1, self.level_y + hrw)
        self.pivot = (self.bounds.x_min + int(self.bounds.width / self.width / 2), self.bounds.y)
        self.pivots = [(self.pivot[0] + x * int(self.bounds.width / self.width), self.pivot[1]) for x in range(self.width)]

    def merge(self, other):
        low_iou = self.bounds.get_iou(other.bounds) < 0.1
        progress_log(f'Merging: IoU: {self.bounds.get_iou(other.bounds)}')
        if low_iou:
            progress_log(f'Warning: merging rooms with low IoU')
        
        self.discovered |= other.discovered

        if self.full_bounds: 
            return not low_iou

        self.bounds = Bounds.from_points(*self.bounds.get_corners(), *other.bounds.get_corners())
        self.align_bounds()
        self.unknown_directions = list(set(self.unknown_directions) & set(other.unknown_directions))

        self.full_bounds = self.unknown_directions == []
        self.room_type = get_room_type(bounds, 'full' if self.full_bounds else '')
        return True

class GameMap:
    def __init__(self, grid_map, rooms, room_height):
        self.room_grid = grid_map
        # sort rooms top -> bottom; left -> right (top-left -> bottom-right)
        self.rooms = sorted(rooms, key=lambda x: -x.level)
        self.rooms = sorted(self.rooms, key=lambda x: x.bounds.x)
        x_lists = [list(x.keys()) for x in grid_map.values()]
        self.grid_size = (len(grid_map), 1 + np.max([np.max(x) for x in x_lists]) - np.min([np.min(x) for x in x_lists]))
        self.room_count = len(rooms)
        self.room_height = round(room_height)

    def render_mini_map(self, scale=3):
        cell_width = 3 * scale
        cell_height = 6 * scale
        gap_size = 1 * scale
        border_thickness = round(0.51 * scale)
        padding = 2 * scale

        bg_color = np.array([57, 57, 57])
        starter_color = np.array([43, 204, 228])
        visited_color = np.array([162, 162, 162])
        structural_color = np.array([0, 0, 0])
        elevator_color = np.array([226, 204, 57])
        unvisited_color = np.array([67, 217, 42])
        error_color = np.array([255, 0, 255])

        canvas = np.full((*(np.array(self.grid_size) * (cell_height + gap_size, cell_width + gap_size) + padding * 2), 3), bg_color)

        coords = np.array([x.pos for x in self.rooms])
        x_min, y_min = np.min(coords, axis=0)

        for room in self.rooms:
            cell_pos = np.array([room.pos[0] - x_min, self.grid_size[0] - room.pos[1] + y_min - 1])
            pixel_pos = cell_pos * (cell_width + gap_size, cell_height + gap_size) + padding - gap_size
            width = (cell_width + gap_size) * room.width - gap_size
            height = cell_height
            color = error_color
            border = False

            if (not room.discovered or room.visited) and room.room_type == 'elevator':
                color = elevator_color
            elif room.visited and room.discovered:
                color = visited_color
            elif not room.visited and not room.discovered:
                color = starter_color
            elif room.discovered:
                color = unvisited_color
                border = True

            bounds = Bounds(pixel_pos[0], pixel_pos[1], pixel_pos[0] + width, pixel_pos[1] + height)
            if border:
                draw_border(canvas, bounds, color, thickness=border_thickness, replace=True)
            else:
                draw_rect(canvas, bounds, color, replace=True)

        return canvas

class GameMapComposer:
    def __init__(self, screen_bounds):
        self.raw_unvisited_rooms = [] # basically rooms detected with detect_rooms
        self.raw_visited_rooms = []   # rooms detected with structural detection
        self.screen_bounds = screen_bounds
        self.structural = None  # ? need
        self.map_bounds = None # bounds in absolute coords which contain all discovered objects

    def _had_same_room(self, raw_room_list, bounds):
        "check if similar room was already added to the list"
        min_iou = 0.9 # others will deal with actually visible deviations 
        for room_bounds, _ in raw_room_list:
            # progress_log(f'   -> IoU: {room_bounds.get_iou(bounds)}')
            if room_bounds.get_iou(bounds) >= min_iou:
                # progress_log(f'Warning: same room was already added: {bounds}')
                return True
        
        return False

    def _process_room_location(self, camera_pos, bounds):
        actual_bounds = bounds.offset(camera_pos)
        limited_directions = self.screen_bounds.touches_contained_bounds(bounds, return_directions=True)
        return (actual_bounds, limited_directions)

    def _update_map_bounds(self, new_bounds):
        self.map_bounds = Bounds.contain_all(self.map_bounds, new_bounds)

    def _add_unvisited_room(self, room_info: (Bounds, list[str]), room_list=None):
        if room_list is None:
            progress_log(f'Adding new unvisited room: {room_info[0]}')
            room_list = self.raw_unvisited_rooms
            self._update_map_bounds(room_info[0])
        
        if self._had_same_room(room_list, room_info[0]): return
        room_list.append(room_info)

    def _add_visited_room(self, room_info: (Bounds, list[str]), room_list=None):
        if room_list is None:
            progress_log(f'Adding new visited room: {room_info[0]}')
            room_list = self.raw_visited_rooms
            self._update_map_bounds(room_info[0])

        if self._had_same_room(room_list, room_info[0]): return
        room_list.append(room_info)

    def update_structural_map(self, structural, camera_pos):
        if self.structural is None:
            self.structural = structural.copy()
            self.structural.bounds = structural.bounds.offset(camera_pos)
            self.structural.simplify_source_mask()
            return
        
        self.structural.unite_with(structural, offset=camera_pos)

    def _process_raw_inputs(self, camera_pos, rooms, structural_rooms, vl=None, ul=None):
        if rooms is None: rooms = []
        if structural_rooms is None: structural_rooms = []

        for room_type, location, bounds in rooms:
            self._add_unvisited_room(self._process_room_location(camera_pos, bounds), room_list=ul)

        for bounds in structural_rooms:
            self._add_visited_room(self._process_room_location(camera_pos, bounds), room_list=vl)

    def add_rooms(self, structural, camera_pos=None, rooms=None, structural_rooms=None):
        "Add rooms detected at given camera x,y position to the map"

        if camera_pos is None: camera_pos = self.last_camera_pos
        self.update_structural_map(structural, camera_pos)
        self._process_raw_inputs(camera_pos, rooms, structural_rooms)

    def compose(self, camera_pos=None, rooms=None, structural_rooms=None):
        ###
        ### Step 0: select room lists
        raw_unvisited_rooms = []
        raw_visited_rooms = []
        if rooms is not None:
            if camera_pos is None: camera_pos = self.last_camera_pos
            self._process_raw_inputs(camera_pos, rooms, structural_rooms, raw_visited_rooms, raw_unvisited_rooms)
        else:
            raw_unvisited_rooms = self.raw_unvisited_rooms
            raw_visited_rooms = self.raw_visited_rooms

        ###
        ### Step 1: compute room height and find room to use as anchor
        self.room_height = None
        match_threshold = 0.12
        unvisited_height_factor = 0.09 # unvisited rooms have larger bounds
        max_level_gap = 1.5 # 150% of room height
        found_full_height = True
        anchor_room = None

        def get_full_height_rooms(room_list):
            return [room for room in room_list if [y for y in room[2] if is_vertical(y)] == []]
        
        def set_room_height(height):
            nonlocal match_threshold
            self.room_height = height
            match_threshold *= self.room_height

        def _match(x1, x2): return abs(x1 - x2) <= match_threshold

        def find_anchor_room(room_list):
            filtered_rooms = [len(limits) for t, bounds, limits in room_list if _match(bounds.height, self.room_height)]
            if filtered_rooms == []: return None
            return room_list[np.argmin(filtered_rooms)]

        raw_room_list = [('unvisited', x[0].offset_bounds(-int(x[0].height * unvisited_height_factor / 2)), x[1]) for x in raw_unvisited_rooms] + [('visited', *x) for x in raw_visited_rooms]
        progress_log(f'Compose: processing {len(raw_room_list)} rooms')
        for room in raw_room_list:
            progress_log(f'   > {room[0]} room at {room[1]}')

        full_height_unvisited = get_full_height_rooms([room for room in raw_room_list if room[0] == 'unvisited'])
        full_height_visited = get_full_height_rooms([room for room in raw_room_list if room[0] == 'visited'])

        # unvisited rooms have more reliable bounds as of now, try that first
        if full_height_unvisited != []:
            set_room_height(np.median([x.height for _, x, _ in full_height_unvisited]))
            anchor_room = find_anchor_room(full_height_unvisited)

        if full_height_visited != []:
            if self.room_height is None:
                set_room_height(np.median([x.height for _, x, _ in full_height_visited]))
            
            # look for anchor room even if we already have one, maybe there's a better one!
            anchor_room_candidate = find_anchor_room(full_height_visited)
            if anchor_room is None or (anchor_room_candidate is not None and len(anchor_room_candidate[2]) < len(anchor_room[2])):
                anchor_room = anchor_room_candidate

        # if no full-height rooms found, make something up
        if self.room_height is None:
            # bias the height to be large
            set_room_height(np.percentile([x.height for _, x, _ in raw_room_list], 75))
            anchor_room = find_anchor_room(raw_room_list)
            found_full_height = False

        # discard too tall rooms
        raw_room_list = [room for room in raw_room_list if room[1].height < self.room_height or _match(room[1].height, self.room_height)]
        progress_log(f'Compose: {len(raw_room_list)} match for height')
        
        ###
        ### Step 2: find and set vertical levels for rooms
        levels = { 0: anchor_room[1].y }
        level_rooms = { 0: [] }
        levels_gap = None
        # levels are done as in real life, e.g. the higher the level on the screen, the higher its number is

        def on_level(bounds, y):
            level_half_height = self.room_height / 2 + match_threshold
            y0, y1 = y - level_half_height, y + level_half_height
            return y0 <= bounds.y_min and bounds.y_max <= y1

        def select_level(bounds):
            diffs = [abs(bounds.y - y) for y in levels.values()]
            closest_level = list(levels.keys())[np.argmin(diffs)]

            if on_level(bounds, levels[closest_level]): return closest_level
            
            return None # no matching levels exist (yet)

        def make_new_level(bounds):
            nonlocal levels_gap
            
            y_delta = bounds.y - levels[0]
            delta_direction = -1 if y_delta > 0 else 1
            gap = abs(y_delta)
            if levels_gap is None:
                if not _match(bounds.height, self.room_height): return None
                if gap / self.room_height > max_level_gap: return None
                levels_gap = gap

                new_level = delta_direction # higher y -> lower on screen -> lower level
                levels[new_level] = bounds.y
                return new_level

            new_level = round(gap / levels_gap) * delta_direction
            levels[new_level] = levels[0] - new_level * levels_gap * delta_direction
            return new_level

        # start placing rooms on their levels
        unplaced_rooms = []
        room_list = raw_room_list
        for _ in range(2): # first iter: initial placement, second: placing unplaced ones
            for room in room_list:
                room_type, bounds, limits = room
                level = select_level(bounds)
                if level is not None:
                    progress_log(f'Compose: placed room on level {level}')
                    level_rooms[level].append(room)
                    continue

                new_level = make_new_level(bounds)
                if new_level is not None:
                    if not on_level(bounds, levels[new_level]):
                        progress_log(f'Compose: discarded room [unknown level]')
                        del levels[new_level] # discard room and new level
                        continue
                    
                    progress_log(f'Compose: placed room on [new] level {new_level}')
                    level_rooms[new_level] = [room]
                    continue

                unplaced_rooms.append(room)

            # stabilize levels
            for level in levels:
                rooms = level_rooms[level]
                rooms_normal_height = [room for room in rooms if _match(room[1].height, self.room_height)]
                if rooms_normal_height != []:
                    levels[level] = np.mean([room[1].y for room in rooms_normal_height])

            progress_log(f'Compose l-{_}: unplaced rooms: {len(unplaced_rooms)}')
            room_list = unplaced_rooms
            if room_list == []: break

            if levels_gap is None: # call it emergency heuristic 
                levels_gap = int(self.room_height * 1.18)

        ###
        ### Step 3: match, merge and classify rooms
        clean_level_rooms = { }
        overlap_iou_threshold = 0.01
        room_zero = None

        for level, rooms in level_rooms.items():
            clean_rooms = []
            for room_type, bounds, limits in rooms:
                room = GameRoom(level, room_type, bounds, limits, levels[level], self.room_height)

                # check if room overlaps with any existing clean room
                discard_room = False
                for clean_room in clean_rooms:
                    if room.bounds.get_iou(clean_room.bounds) >= overlap_iou_threshold:
                        progress_log(f'Compose: rooms overlap, attempting merge')
                        discard_room = clean_room.merge(room)
                        break

                if discard_room: continue
                clean_rooms.append(room)
                if room_zero is None and room.full_bounds or 'left' not in room.unknown_directions:
                    room_zero = room

            clean_level_rooms[level] = sorted(clean_rooms, key=lambda x: x.bounds.x)

        ###
        ### Step 4: put rooms on grid
        if room_zero is None:
            result_log('Critical: room zero was not set')
            return

        flat_room_list = [room for level in clean_level_rooms.values() for room in level]
        progress_log(f'Compose: making a grid, {len(flat_room_list)} rooms total')
        flat_room_list.remove(room_zero)
        grid_map = {}
        on_grid_rooms = []
        grid_origin = room_zero.pivot

        def put_on_grid(room, x, y):
            nonlocal grid_map
            progress_log(f'Compose: placing on grid {room.bounds}: {y}, {x}, width: {room.width}')
            on_grid_rooms.append(room)
            room.pos = (x, y)

            for pivot in room.pivots:
                draw_disk(get_do(), pivot, 8, np.array([255, 0, 0, 255]))

            if y not in grid_map:
                grid_map[y] = {c: room for c in range(x, x + room.width)}
                return

            for c in range(x, x + room.width):
                if c in grid_map[y]: raise Exception('your bad')
                grid_map[y][c] = room
        
        put_on_grid(room_zero, 0, 0)

        use_proximity_placing = True
        pivot_tolerance = int(self.room_height * 0.2)
        cell_width = int(self.room_height * 1.75 / 3)
        while len(flat_room_list) > 0:
            unplaced_rooms = []

            for room in flat_room_list:
                cell_y = room.level - room_zero.level
                best_match = np.argmin([abs(x.bounds.x - room.bounds.x) for x in on_grid_rooms])
                anchor = on_grid_rooms[best_match]
                progress_log(f'Selected anchor: {anchor.bounds} ({abs(anchor.bounds.x - room.bounds.x)})')
                i1, j1 = None, None # closest match
                closest = float('inf')
                i0, j0 = None, None # direct match

                for i, pivot in enumerate(room.pivots):
                    for j, placed_pivot in enumerate(anchor.pivots):
                        distance = abs(pivot[0] - placed_pivot[0])
                        if distance < closest:
                            i1, j1 = i, j
                            closest = distance
                        if distance < pivot_tolerance:
                            i0, j0 = i, j
                            break
                
                direct_match = j0 is not None
                indirect_match = (closest - cell_width < pivot_tolerance) and not direct_match
                no_match = not direct_match and not indirect_match
                if no_match and use_proximity_placing:
                    progress_log('Compose: Room not placed due to proximity setting')
                    unplaced_rooms.append(room)
                    continue
                
                i2, j2 = i0, j0
                offset = 0
                if indirect_match or no_match:
                    i2, j2 = i1, j1
                    delta = room.pivots[i2][0] - anchor.pivots[j2][0]
                    delta_sign = 1 if delta > 0 else -1
                    if indirect_match:
                        offset = delta_sign
                        progress_log('Compose: placing using indirect match')
                    else:
                        offset = delta_sign * round(abs(delta) / cell_width)
                        progress_log('Compose: placing using extrapolation')
                else:
                    progress_log('Compose: placing using direct match')


                cell_x = anchor.pos[0] + j2 - i2 + offset
                put_on_grid(room, cell_x, cell_y)

            use_proximity_placing = len(unplaced_rooms) != len(flat_room_list)
            if not use_proximity_placing:
                progress_log(f'Warning: proximity placement is off')
            flat_room_list = unplaced_rooms

        game_map = GameMap(grid_map, on_grid_rooms, self.room_height)
        if rooms is not None:
            self.current_map = game_map
        
        result_log(f'GameMap compose complete. Grid: {game_map.grid_size[0]} levels, width: {game_map.grid_size[1]} cells. Rooms: {game_map.room_count}')
        return game_map

    def get_current_discovery_direction(self):
        "which way and how far should camera move to (at some point) see unseen rooms / fragments"
        # primitive implementation, TODO improve
        #  - need `pathfinding` - camera shouldn't lose track of rooms, so it is essential to choose direction in which camera will still be able
        #       to track its location
        #  - need to discover around rooms: probably based on structural (or camera pos history) determine where there may be rooms
        #       that are really close to known ones but were not in camera's fov
        c_pos = self.last_camera_pos
        rooms = self.current_map.rooms
        # nothing special about height, could've been a hardcoded number instead
        min_camera_movement = self.current_map.room_height

        for room in rooms:
            if room.full_bounds: continue
            # direction we must move in
            delta_x = room.bounds.x - c_pos[0]
            delta_y = room.bounds.y - c_pos[1]

            if abs(delta_x) > min_camera_movement:
                return 'left' if delta_x < 0 else 'right'
            if abs(delta_y) > min_camera_movement:
                return 'down' if delta_y < 0 else 'up'

            return room.unknown_directions[0]

        return None

    def set_camera_position(self, camera_pos):
        self.last_camera_pos = camera_pos
    
    def _room_match_exact(self, anchor_room_info, room_info, direction):
        ### common setup
        anchor_bounds = anchor_room_info[1]
        bounds = room_info[1]
        shared_limits = set(anchor_room_info[2]).intersect(room_info[2])
        all_limits = set(anchor_room_info[2]).unite(room_info[2])
        different_limits = all_limits - shared_limits
        directions = ['left', 'down', 'right', 'up']

        ### direction specific setup

        v = is_vertical(direction)
        primary_index = directions.index(direction)
        secondary_index = directions.index(opposite_direction(direction))
        min_index = 1 if v else 0
        max_index = 3 if v else 2
        illegal_limit_diff = is_horizontal if v else is_vertical
        matching_indices = [0, 2] if v else [1, 3]

        ### common match code
        
        for index in matching_indices:
            if adjusted_bounds.to_rect()[index] != anchor_bounds.to_rect()[index]: return False, 0

        # only limits along pan direction can change
        for x in different_limits:
            if illegal_limit_diff(x): return False, 0
        
        if direction not in shared_limits: # choose anchor side
            camera_offset = anchor_bounds.to_rect()[primary_index] - bounds.to_rect()[primary_index]
        else:
            camera_offset = anchor_bounds.to_rect()[secondary_index] - bounds.to_rect()[secondary_index]

        if camera_offset < 0: return False, 0 # that would mean camera went down instead
        camera_offset = int(camera_offset) # just in case
        adjusted_bounds = bounds.offset(DIRECTION_MAP_XY[direction] * camera_offset)

        # basically height comparison: rooms should have non-contradictory heights to match
        if adjusted_bounds.to_rect()[max_index] > anchor_bounds.to_rect()[max_index] or \
            adjusted_bounds.to_rect()[min_index] < anchor_bounds.to_rect()[min_index]:
            return False, 0

        return True, camera_offset

        ###
        ###
        ### reference solution
        if False:
            if direction == 'up': # aka increasing Y coord (of camera, rooms' y is decreasing)
                if adjusted_bounds.x_min != anchor_bounds.x_min and adjusted_bounds.x_max != anchor_bounds.x_max:
                    return False, 0

                # only limits along pan direction can change
                for x in different_limits:
                    if is_horizontal(x): return False, 0

                if 'up' not in shared_limits: # choose anchor side
                    camera_offset = anchor_bounds.y_max - bounds.y_max
                else:
                    camera_offset = anchor_bounds.y_min - bounds.y_min

                if camera_offset < 0: return False, 0 # that would mean camera went down instead
                adjusted_bounds = bounds.offset(DIRECTION_MAP_XY['up'] * camera_offset)

                # basically height comparison: rooms should have non-contradictory heights to match
                if adjusted_bounds.y_max > anchor_bounds.y_max or adjusted_bounds.y_min < anchor_bounds.y_min:
                    return False, 0

                return True, camera_offset
        ###
        ###
        ###
        
    # direction is according to numerical direction, e.g. up means increasing Y, left means decreasing X
    def estimate_camera_movement(self, pan_direction, target_distance, rooms, structural_rooms, update_camera_pos=True):
        # we cannot start with room coordinate conversion since we don't know current camera position!

        ### Step 0: process new rooms (compute limits and tag with visited/unvisited)
        v_rooms, u_rooms = [], []
        # set camera pos to 0 so that bounds stay as is
        self._process_raw_inputs((0, 0), rooms, structural_rooms, v_rooms, u_rooms)
        new_room_list = [('unvisited', *x) for x in u_rooms] + [('visited', *x) for x in v_rooms]

        ### Step 1: Make a crop of all known rooms according to panning bounds
        last_camera_bounds = self.screen_bounds.offset(self.last_camera_pos)
        panning_bounds = get_panning_bounds(last_camera_bounds, pan_direction, self.map_bounds)

        raw_room_list = [('unvisited', *x) for x in raw_unvisited_rooms] + [('visited', *x) for x in raw_visited_rooms]
        cropped_rooms = []

        for room_type, bounds, limits in raw_room_list:
            crop = bounds.intersect(panning_bounds).collapse_negative()
            if crop.area <= 0: continue

            # this will make it way easier to compare local rooms' limits with known ones
            new_limits = panning_bounds.touches_contained_bounds(crop)

            # offset bounds to match as closely as possible with local coords (but it won't be exact!)
            cropped_rooms.append((room_type, crop.offset(-self.last_camera_pos), new_limits))

        ### Step 2: Match new rooms with known ones
        matches = {} # distance: [*room_matches]

        for room in new_room_list:
            for anchor in cropped_rooms:
                match, distance = self._room_match_exact(anchor, room, pan_direction)
                if not match: continue
                if distance not in matches:
                    matches[distance] = 0
                else:
                    matches[distance] += 1
        
        ### Step 3: find best match distance
        distances = list(matches.values())
        max_matches = np.max(distances)
        best_matches = [dist for dist, match_count in matches.items() if match_count == max_matches]
        estimated_distance = best_matches[np.argmin(np.abs(np.array(best_matches) - target_distance))]

        if update_camera_pos:
            self.last_camera_pos = DIRECTION_MAP_XY[pan_direction] * estimated_distance + self.last_camera_pos
            progress_log(f'Movement estimation: {estimated_distance} px. New position: {self.last_camera_pos}')

        return estimated_distance

###
### ======================== MAIN ========================
### 

class FalloutShelterAutomationApp:
    # constants (?)
    version = 'v0.8.0'
    app_update_interval_ms: int = 20
    "Controls how often does app make regular update iterations (e.g. updating displayed log, starting async tasks, etc.)"
    chord_start_sequence: str = '<ctrl>+f' # enter this key combination to start a chord
    capture_name: str = 'Fallout Shelter' # window name to capture images from
            # 'Mozilla Firefox' # 
    # mission script parameters (game related)
    camera_pan_deadzone_size: float = 0.04 # there are limits on how accurate you can pan camera using keys
    "Precision of camera panning - the center of the room should end up in a rectangle with dimensions camera_pan_deadzone_size * screen_shape"
    camera_default_pan_distance = 500 # pan_camera will use this distance if no arguments are supplied
    camera_pan_velocity = 3300 # distance divided by this is approximate pan time  
    crit_wait_count = 4 # how many crits to see for data collection before striking?
    dialogue_mode: str = 'manual' # random / manual
    "Controls which dialog handler is used when selecting dialog option"
    overlay_update_interval = 25 # how often should overlay image be updated (in update ticks)
    logs_font_size = 8 # font size for on-screen logs

    # state attributes
    tick_counter = 0 # Counts the number of times FalloutShelterAutomationApp.update() was called
    current_execution_target = None
    repeat_current_execution_target = False
    terminate_pending = False
    keyboard_chord_pending = False
    task_in_progress = False
    script_running = False
    last_key_pressed = None
    overlay_on = True # is FSA overlay currently displayed?
    show_log = True
    state_overlay: np.ndarray = None # Current overlay image that represents app state (normal / error) (currently as a red/yellow border)
    mission_paused = False # pauses mission script
    mock_mode = False # whether mock screen capture is used
    mask_escape_key = False # temporarily disables immediate shutdown when Esc is pressed
    normal_room_height = None # we cache (approximate) height of rooms on default zoom here

    ###
    ### Game-aware tech [WIP]
    ###

    def build_initial_game_map(self):
        ### make it so that camera barely sees top-left structural corner

        if not self.mock_mode and False:
            progress_log('Building map: fixing structural')
            self.full_zoom_out()
            did_down_pan = did_left_pan = did_up_pan = did_right_pan = False
            previous_structural = None

            while True:
                structural, directions = detect_structural(self.next_frame())
                panned = False

                # camera moves up
                if 'down' in directions:
                    self.pan_camera('down', duration=0.03, post_pan_sleep=False)
                    did_down_pan = True
                    panned = True
                # camera moves left
                if 'left' in directions:
                    self.pan_camera('left', duration=0.03, post_pan_sleep=False)
                    did_left_pan = True
                    panned = True

                if not did_down_pan:
                    if previous_structural is not None and did_up_pan and previous_structural.bounds == structural.bounds:
                        structural.compute(patch_mask=True)
                        previous_structural.compute(patch_mask=True)
                        if np.all(structural.patch_mask == previous_structural.patch_mask):
                            did_down_pan = True

                    self.pan_camera('up', duration=0.03, post_pan_sleep=False)
                    did_up_pan = True
                    panned = True

                if not did_left_pan:
                    if previous_structural is not None and did_right_pan and previous_structural.bounds == structural.bounds:
                        structural.compute(patch_mask=True)
                        previous_structural.compute(patch_mask=True)
                        if np.all(structural.patch_mask == previous_structural.patch_mask):
                            did_left_pan = True

                    self.pan_camera('right', duration=0.03, post_pan_sleep=False)
                    did_right_pan = True
                    panned = True

                previous_structural = structural
                if panned:
                    sleep(camera_post_pan_duration)
                    continue

                break
            
            ### find the first room to start building map
            progress_log('Building map: looking for a room')

            for _ in slow_loop(interval=0, max_iter_count=25):
                structural, directions = detect_structural(self.next_frame())
                rooms = detect_structural_rooms(structural)

                if rooms != []: break

                self.pan_camera('up')

            if rooms == []:
                result_log('Building map: failed to find a starter room')
                return False

        ### Now camera should be seeing a room. Start building the map
        progress_log('Building map: anchor set. Start building')
        self.game_map_composer = GameMapComposer(self.screen_bounds)
        self.game_map_composer.set_camera_position((0, 0))
        create_debug_frame()

        def draw_minimap(map):
            minimap = map.render_mini_map(scale=5)
            shape = minimap.shape[:2]
            offset = 5
            get_do()[offset:shape[0] + offset, offset:shape[1] + offset, :3] = minimap
            get_do()[offset:shape[0] + offset, offset:shape[1] + offset, 3] = 255
        
        frame = self.next_frame()
        structural, _ = detect_structural(frame)
        str_rooms = detect_structural_rooms(structural)
        rooms = detect_rooms(frame)

        if rooms == [] and str_rooms == []:
            result_log('ALERT: no rooms found, aborting')
            return

        while True:
            self.game_map_composer.add_rooms(structural, rooms=rooms, structural_rooms=str_rooms)
            draw_minimap(self.game_map_composer.compose())

            discovery_direction = self.game_map_composer.get_current_discovery_direction()
            if discovery_direction is None:
                result_log('Building map complete')
                break

            self.pan_camera(discovery_direction, duration=0.03)
            frame = self.next_frame()

            structural, _ = detect_structural(frame)
            str_rooms = detect_structural_rooms(structural)
            rooms = detect_rooms(frame)

            self.game_map_composer.estimate_camera_movement(*self.camera_pan_log[-1], rooms, str_rooms)

    ### 
    ### Game automation functions
    ###

    def direction_to_screen_center(self, loc: Bounds, blocked: str):
        """Which way should camera be panned for `loc` to move towards screen center? `blocked` specifies direction along which panning is not possible.
        Return value: (new_direction: str, distance_to_center_along_direction: int)"""
        # note that if loc is on the left of the screen, we should move right for it to be centered
        if not is_horizontal(blocked):
            if loc.x < self.camera_deadzone.x_min: return 'left', self.screen_center_xy[0] - loc.x
            if loc.x > self.camera_deadzone.x_max: return 'right', loc.x - self.screen_center_xy[0]

        if not is_vertical(blocked):
            if loc.y < self.camera_deadzone.y_min: return 'down', self.screen_center_xy[1] - loc.y
            if loc.y > self.camera_deadzone.y_max: return 'up', loc.y - self.screen_center_xy[1]

        return None, None

    def get_panning_bounds(self, bounds, direction):
        "Returns Bounds to be used for screen grab when panning in `direction` and keeping `bounds` in view after panning"
        return get_panning_bounds(bounds, direction, self.screen_bounds)

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

    # when clicking on a room, click animation may be as short as 0.25s (visible in original room bounds)
    # if you look outside of room bounds (~5-10% larger bounds), animation is visible for around 0.45-0.5s 
    def navigate_to_room(self, room_bounds: Bounds, click=True, zoom_in=True):
        progress_log(f'Navigating to the room: {room_bounds}')

        last_direction = None # pan direction on last iteration
        blocked_direction = None # direction along which panning is no longer possible

        ### Zoom in the camera on the room
        if zoom_in:
            room_anchors = room_bounds.get_corners(get_8=True)
            furthest_anchor_i = np.argmax([np.linalg.norm(x) for x in np.array(room_anchors) - self.screen_center_xy])
            zoom_point = self.filter_mouse_coords(*room_anchors[furthest_anchor_i])
            self.zoom_camera(*zoom_point)
            progress_log('zoomed in, rediscovering room...')
            rooms = detect_rooms(self.next_frame())
            if len(rooms) == 0:
                progress_log('Navigation: target lost')
                return False, None, None, None

            # find room closest to where we had cursor for zooming in
            vectors_norms = [[np.linalg.norm(np.array(corner) - zoom_point) for corner in z.get_corners()] for x, y, z in rooms]
            room_bounds = rooms[np.argmin([np.min(norms) for norms in vectors_norms])][2]

            if self.normal_room_height is None:
                self.normal_room_height = room_bounds.height

        ### Camera panning to center the room on the screen
        distance_multiplier = 1 # needed to avoid infinite panning loops
        while distance_multiplier > 0.05: # loop should end due to break, loop condition is a backup solution
            direction, distance = self.direction_to_screen_center(room_bounds, blocked_direction)
            if direction is None: break
            if are_opposite_directions(last_direction, direction): 
                distance_multiplier *= 0.35

            progress_log(f'Panning: {direction} for {distance}px ({int(100 * distance_multiplier)}%)')    
            self.pan_camera(direction, distance=distance * distance_multiplier)

            progress_log(f'Finding room again...')
            rescan_bounds = self.get_panning_bounds(room_bounds, direction)
            rooms = None
            sleep(0.1) # post-pan delay
            for _ in slow_loop(interval=0.1, max_duration=1):
                screen_crop = self.latest_frame[rescan_bounds.to_slice()]
                debug_log_image(screen_crop, 'navigation-rescan')
                rooms = detect_rooms(screen_crop)
                if len(rooms) > 0: break
            
            if rooms is None:
                progress_log('Panning failed: timeout')
                return False, None, None, None

            last_bounds, room_bounds = room_bounds, rooms[0][2].offset(rescan_bounds.low_pos)
            if last_bounds.x == room_bounds.x and last_bounds.y == room_bounds.y:
                progress_log(f'Panning blocked: `{direction}`')
                if blocked_direction is not None and blocked_direction != direction: break
                blocked_direction = direction
                distance_multiplier = 1

            last_direction = direction

        progress_log('Panning complete')
        if not click:
            return True, room_bounds, None, None

        ### Click on room and verify the click was registered
        debug_log_image(self.freeze_frame(), 'navigation-end-grab')
        rooms = detect_rooms(self.fixed_frame)
        filtered_rooms = [x for _, _, x in rooms if x.shape == room_bounds.shape]
        if len(filtered_rooms) == 0:
            progress_log('Navigation: target lost')
            return False, None, None, None

        room_bounds = min(filtered_rooms, key=lambda x: np.linalg.norm(x.pos - room_bounds.pos))

        max_click_attempts = 3
        for _ in range(max_click_attempts):
            pre_click_room = self.latest_frame[room_bounds.to_slice()]

            progress_log(f'Clicking: {room_bounds.x, room_bounds.y}')
            self.mouse_click(room_bounds.x, room_bounds.y)
            click_time = perf_counter()

            # detect click animation, 1 second timeout
            for frame in self.iterate_latest_frames(min_frame_delta=0.04, max_duration=1):
                next_room_frame = frame[room_bounds.to_slice()]
                click_diff, diff_mask = compute_img_diff_ratio(next_room_frame, pre_click_room, get_mask=True)
                if click_diff < 0.01: continue

                if is_room_click_diff(diff_mask):
                    debug_log_image(pre_click_room, 'preclick', increment_counter=False)
                    debug_log_image(next_room_frame, 'postclick', increment_counter=False)
                    debug_log_image(diff_mask, 'clickdiff')
                    return True, room_bounds, pre_click_room, rooms

            progress_log(f'Click failed, repeat:')

        progress_log('Gave up on clicking...')
        return False, None, None, None

    def make_critical_strike(self, crit_bounds):
        grab_size = min(crit_bounds.width, crit_bounds.height) // 3
        grab_bounds = Bounds.from_center(crit_bounds.x, crit_bounds.y, grab_size, grab_size)
        validation_bounds = grab_bounds.get_scaled_from_center(scale=10) # original bounds are too small, may miss the changes
        grab_bbox = grab_bounds.get_bbox()
        validation_bbox = validation_bounds.get_bbox()

        def get_cue_pixel_count(pixels):
            return np.sum(match_color_exact(pixels, critical_cue_color)) + np.sum(match_color_exact(pixels, critical_progress_color))

        progress_log(f'Collecting timings...')
        
        crit_frames_debug = []
        hit_timings = [] # [(crit_start, crit_end), ...]
        last_i = -2 # does not have to be -2, any negative (except -1) will do the same
        restore_overlay = False
        overlay_disabled = False
        capture_start = perf_counter()
        for i, (frame, timestamp) in enumerate(self.iterate_latest_frames(max_duration=crit_max_duration - 1.5, yield_timestamp=True)):
            crop = crop_image(frame, grab_bbox)
            crit_pixels = np.sum(match_color_exact(crop, critical_hit_color))
            if crit_pixels >= min_critical_pixels:
                if last_i + 1 == i: # this frame is continuation of the previous hit, update end timestamp
                    hit_timings[-1] = (hit_timings[-1][0], timestamp)
                else:
                    hit_timings.append((timestamp, timestamp))
                last_i = i
                crit_frames_debug.append((frame, timestamp))

                if len(hit_timings) >= self.crit_wait_count:
                    break
            else: # make sure we are even doing a crit right now
                cue_pixels = get_cue_pixel_count(crop)
                if cue_pixels < min_critical_pixels:
                    debug_log_image(frame, 'crit-abort')
                    progress_log('Aborting crit, no matching pixels')
                    break
            
            if len(hit_timings) + 1 == self.crit_wait_count and not overlay_disabled:
                restore_overlay = self.hide_overlay()
                overlay_disabled = True

        if len(hit_timings) < 2: # because need at least 2 timings to get time delta
            progress_log('Crit failed: not enough data')
            self.show_overlay(restore_overlay)
            return

        diffs = [y[0] - x[1] for x, y in zip(hit_timings[:-1], hit_timings[1:])]
        avg_diff = np.median(diffs) # median should "filter out" outliers, unlike mean
        std_diff = np.std(diffs)
        progress_log(f'Crit diff info gathered: {[f"{x:0.2f}" for x in diffs]}, avg: {avg_diff:0.2f}s, std: {std_diff:0.4f}')
        if std_diff > 0.02: result_log(f'Warning: high timing deviation')

        # calculate next crit time
        next_crit = hit_timings[-1][0]
        crit_scored = False
        best_score = False

        # loop in case a click doesn't go through the first time :)
        while not crit_scored and perf_counter() - capture_start < crit_max_duration:
            for _ in range(15): # limit the iteration number
                if next_crit < perf_counter(): 
                    next_crit += avg_diff

            # next_crit -= 0.03 # bias
            while perf_counter() < next_crit: pass

            self.mouse_input.press(mouse.Button.left)
            sleep(0.08)
            self.mouse_input.release(mouse.Button.left)

            # determine crit type and verify the click went through
            validation_frame = crop_image(self.latest_frame, validation_bbox)
            ref_miss_pixels = get_cue_pixel_count(crop)
            ref_hit_pixels = np.sum(match_color_exact(validation_frame, critical_hit_color))
            if ref_miss_pixels < min_critical_pixels and ref_hit_pixels < min_critical_pixels:
                result_log(f'Error: no crit pixels detected to verify ({ref_miss_pixels}/{ref_hit_pixels})')
                break

            best_score = ref_hit_pixels >= min_critical_pixels
            crit_scored = True

            for _ in slow_loop(interval=0.03, max_duration=0.35):
                validation_frame = crop_image(self.latest_frame, validation_bbox)
                miss_pixels = get_cue_pixel_count(crop)
                hit_pixels = np.sum(match_color_exact(validation_frame, critical_hit_color))

                if miss_pixels != ref_miss_pixels or hit_pixels != ref_hit_pixels: 
                    crit_scored = False
                    progress_log('Failed to verify crit, retrying')
                    break

        if crit_scored:
            result_log(f'Crit scored! is x5: {best_score}')
        else:
            result_log('Failed to score a crit...')

        self.show_overlay(restore_overlay)
        for frame, timestamp in crit_frames_debug:
            debug_log_image(frame, f'crit-frame-{timestamp-crit_frames_debug[0][1]:0.2f}')

        sleep(max(0, 2 - perf_counter() + next_crit)) # wait for crit message to disappear

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
            # idk why it was there, let's try without it
            # sleep(0.3)
            self.make_critical_strike(bounds)
    
        return len(meds) > 0, len(crits) > 0

    def full_zoom_out(self):
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

    def zoom_camera(self, x, y, zoom_duration=0.75, zoom_out=False):
        "Zooms in while focusing on specified point. After zoom-in a 3-slot room should still fit on screen + a bit of extra space around"
        self.mouse_input.position = self.filter_mouse_coords(x, y)
        zoom_key = 'q' if zoom_out else 'e'
        self.keyboard_input.press(zoom_key)
        sleep(zoom_duration)
        self.keyboard_input.release(zoom_key)
        # experimenting without this sleep
        # sleep(0.1)
        if not zoom_out:
            self.zoomed_out = False

    def look_for_room_using_structural(self):
        "Returns true if at least one unvisited room has been found and is on screen right now"
        self.full_zoom_out()
        reached_left, reached_right = False, False
        direction = 'down' # camera actually goes up
    
        def scan_primary(directions):
            nonlocal reached_left, reached_right
            if direction in directions: 
                self.pan_camera(direction)
                reached_left = False
                reached_right = False
                return True
            
            return False
    
        def scan_secondary():
            if not reached_left:
                self.pan_camera('left')
                return True
            if not reached_right:
                self.pan_camera('right')
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

        self.hide_ui_panels() # always good to remove that
        # if we didn't see a new room in a while, start structural scan
        if self.room_scan_retry_start is not None:
            if perf_counter() - self.room_scan_retry_start >= structural_scan_begin_timeout and not self.last_iteration_room_detected:
                progress_log('General iteration timeout, engaging structure-based scanning...')
                if not self.look_for_room_using_structural():
                    progress_log('Failed to detect new rooms, aborting execution')
                    self.script_complete = True # terminate mission script
                    return

        self.last_iteration_room_detected = False
        if self.need_zoom_out and not self.zoomed_out:
            self.full_zoom_out()
        debug_log_image(self.freeze_frame(), 'iteration-capture')
        self.battle_iteration() # meds / level-ups should be clicked

        rooms = detect_rooms(self.fixed_frame) # scan for unvisited rooms
        if len(rooms) == 0:
            progress_log('No rooms detected, retrying...')
            self.need_zoom_out = True
            if self.room_scan_retry_start is None:
                self.room_scan_retry_start = perf_counter()
            return

        self.room_scan_retry_start = None
        self.last_iteration_room_detected = True
        self.need_zoom_out = False

        self.target_room = rooms[0] # room we will navigate to next
        progress_log(f'Found room: {self.target_room[2]}, type: {self.target_room[0]}')
        self.script_state = self.room_navigation_state

    def room_navigation_state(self):
        "Pans camera over to target room, clicks, and waits until room is reached"
        _, _, target_bounds = self.target_room
        navigate_successful, self.current_room_bounds, pre_click_img, self.last_detected_rooms = self.navigate_to_room(target_bounds, click=True, zoom_in=self.zoomed_out)
        if not navigate_successful:
            self.script_state = self.room_scan_state
            return

        progress_log(f'Room navigation successful, waiting for walk complete...')
        # this is used to later detect if any new rooms appear after we reach the room
        # self.last_detected_rooms = detect_rooms(self.next_frame())
        self.current_room_type = get_room_type(self.current_room_bounds, 'full')

        # wait until the room is reached
        for _ in slow_loop(interval=walk_wait_min_interval):
            start_time = perf_counter()
            current_room = self.latest_frame[self.current_room_bounds.to_slice()]
            diff = compute_img_diff_ratio(pre_click_img, current_room)
            progress_log(f'Waiting for walk completion: current diff {100*diff:0.2f}%')

            if diff >= room_reached_min_diff: 
                debug_log_image(current_room, f'room-reached-{diff*100:0.2f}-diff')
                break # room reached!

        progress_log(f'Walk complete! Analyzing situation...') # room reached, wrap up and switch state
        self.post_walk_screen_img = self.latest_frame
        debug_log_image(self.post_walk_screen_img, f'post-walk-structural')

        self.script_state = self.room_analysis_state

    def structural_detection_task(self, delay=0):
        sleep(delay)
        frame = self.latest_frame # do not use freeze frame from async tasks
        structural, _ = detect_structural(frame)
        structural_rooms = detect_structural_rooms(structural)
        progress_log(f'Structural: detected {len(structural_rooms)} rooms')

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
        progress_log(f'Structural detection done')

    # some notes for reference
    #  - if the room is empty, the adjacent rooms are revealed pretty much immediately as the first dweller enters it
    #       + except when it's not happening. sometimes it waits for the last dweller to enter before revealing
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
        self.enemies_detected = False
        self.dialogue_detected = False
        self.loot_bound_detected = True
        self.loot_collected = False
        self.loot_collection_cancelled = False
        self.new_room_detected = False
        room_analysis_timeout = 7

        # first thing we do - check if there is a new room appeared. If so, collect loot in current room
        def room_detection_task():
            state_index = self.state_index
            for _ in slow_loop(interval=0.5, max_duration=room_analysis_timeout):
                if state_index != self.state_index: return # room analysis ended, return
                new_rooms = detect_rooms(self.latest_frame)
                if has_new_room(new_rooms, self.last_detected_rooms):
                    progress_log('Room analysis: new room')
                    self.new_room_detected = True
                    return

        room_detection_thread = threading.Thread(target=room_detection_task)
        room_detection_thread.start()

        # the enemies can appear at any point, just look for them in background
        self.run_enemy_detection = True
        def enemy_detection_task():
            for _ in slow_loop(interval=0.25): # scan at most four times a second
                self.enemies_detected, self.current_enemies = detect_enemies(self.latest_frame)
                if not self.run_enemy_detection or self.enemies_detected: 
                    if self.enemies_detected: progress_log('Interrupting: enemies detected')
                    break

        enemy_detection_thread = threading.Thread(target=enemy_detection_task)
        enemy_detection_thread.start()
        self.hide_ui_panels() # why not

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
        loot_collection_thread = None
        loot_collection_delay = 2

        # ~~ EveNT lOoP ~~
        progress_log(f'Starting event loop...')
        loop_start = perf_counter()
        for _ in slow_loop(interval=0.1):
            if self.enemies_detected:
                self.run_dialogue_detection = False
                self.script_state = self.battle_state
                self.loot_collection_cancelled = True
                return
            if self.dialogue_detected: # leave enemy detection on
                self.script_state = self.dialogue_state
                # bounds change when entering the dialogue
                self.loot_bound_detected = False
                self.loot_collection_cancelled = True
                progress_log(f'Detecting structural...')
                # delay needed since room zoom in before dialogue needs time to complete
                structural_detection_thread = threading.Thread(target=lambda: self.structural_detection_task(delay=1))
                structural_detection_thread.start()
                return
            if loot_collection_thread is None and perf_counter() - loop_start > loot_collection_delay:
                loot_collection_thread = threading.Thread(target=self.loot_collection_task)
                loot_collection_thread.start()

            if self.loot_collected and self.new_room_detected:
                progress_log(f'Loot collected and new room detected, proceeding')
                break # loot collection state is skipped if self.loot_collected is True

            if perf_counter() - loop_start > room_analysis_timeout:
                progress_log(f'Event loop timeout, proceed to looting')
                break # wait up to 7 seconds for something, otherwise proceed to looting
        
        # stop threads and switch state
        self.run_enemy_detection = False
        self.run_dialogue_detection = False
        self.script_state = self.loot_collection_state

    def dialogue_state(self):
        result_log('Dialog: waiting...')
        dialogue_complete = False
        dialogue_finish_time = None
        for _ in slow_loop(interval=0.2):
            dialogue_choice, buttons = detect_dialogue_buttons(self.freeze_frame(), self.screen_shape_xy)
            if dialogue_choice:
                progress_log('Dialog choice detected!')
                progress_log(f'Starting handler: {self.dialogue_mode}')
                debug_log_image(self.fixed_frame, 'dialogue-choice-capture')
                self.dialogue_handlers[self.dialogue_mode](buttons)
                continue
            if self.enemies_detected:
                self.script_state = self.battle_state
                return
            if detect_ui(self.fixed_frame):
                if dialogue_finish_time is None:
                    dialogue_finish_time = perf_counter()
                    progress_log('Dialogue finished')
                if perf_counter() - dialogue_finish_time >= 3: # wait for potential enemies to appear
                    progress_log('Timeout, exiting dialogue')
                    self.script_state = self.loot_collection_state
                    self.run_enemy_detection = False # stop enemy detection now
                    return
            else: # mouse click to skip dialogue faster!
                self.mouse_click(*self.screen_center_xy)

    def battle_state(self):
        have_enemies = True

        def ui_closer_task():
            state_index = self.state_index
            for _ in slow_loop(interval=1):
                if self.state_index != state_index: return
                self.hide_ui_panels()
        
        ui_closer_thread = threading.Thread(target=ui_closer_task)
        ui_closer_thread.start()

        last_enemy_detection = perf_counter()
        # this is around how long we should wait for enemy death animations anyway
        enemy_wait_timeout = 3

        while perf_counter() - last_enemy_detection < enemy_wait_timeout:
            debug_log_image(self.freeze_frame(), 'battle-iteration-capture')
            had_meds, had_crits = self.battle_iteration() # take care of meds / crits
            if had_crits: continue # critical hit marker can obscure enemies and they go undetected

            have_enemies, enemies = detect_enemies(self.fixed_frame)
            progress_log(f'BIT: {len(enemies)} enemies detected')

            if have_enemies: last_enemy_detection = perf_counter()
        
        self.script_state = self.loot_collection_state

    def hide_ui_panels(self, return_immediately=False):
        "Returns True if there were UI panels to be closed"
        if not detect_character_panel(self.latest_frame) and not detect_objective_panel(self.latest_frame):
            return False

        self.mask_escape_key = True
        self.keyboard_input.press(keyboard.Key.esc)
        sleep(0.03)
        self.keyboard_input.release(keyboard.Key.esc)
        self.mask_escape_key = False
        if not return_immediately: sleep(0.2) # character panel hide animation (is probably longer)
        return True

    def do_collect_loot(self, loot_coords):
        for loot in loot_coords: 
            self.mouse_click(*loot)

    def do_detect_loot(self, delay=0, debug_prefix=''):
        "runs loot detection using frames from frames buffer with optional delay before capture"
        sleep(delay)
        frames = [x[0] for x in self.copy_frames_buffer(min_frame_delta=0.5)[-2:]]
        for i, frame in enumerate(frames):
            debug_log_image(frame, f'{debug_prefix}loot-detection-frame-{i}')
        return detect_loot(self.current_room_bounds, frames=frames)

    def filter_loot_coords(self, loot_coords):
        "Modifies coordinates to avoid clicking the same spots we already clicked"
        filter_radius = 9

        if len(self.loot_click_locations) == 0: 
            self.loot_click_locations += loot_coords
            return loot_coords

        filtered_coords = []
        for point in loot_coords:
            matches = np.linalg.norm(np.array(self.loot_click_locations) - point, axis=1) <= filter_radius
            if np.any(matches):
                ox, oy = point
                search_radius = 5
                progress_log(f'Warning: modifying loot point {point}')

                while True:
                    search_space = np.ones((search_radius * 2 + 1, search_radius * 2 + 1), dtype=bool)

                    # limit search space to a circle shape
                    yy, xx = np.ogrid[-search_radius:search_radius + 1, -search_radius:search_radius + 1]
                    mask = xx ** 2 + yy ** 2 > search_radius ** 2
                    search_space[mask] = False

                    # mask out all the clicked spots
                    for clicked_point in self.loot_click_locations:
                        x, y = clicked_point
                        dx, dy = ox - x, oy - y
                        yy, xx = np.ogrid[dy - search_radius:dy + search_radius + 1, dx - search_radius:dx + search_radius + 1]
                        mask = xx ** 2 + yy ** 2 <= filter_radius ** 2
                        search_space[mask] = False

                    ys, xs = search_space.nonzero()
                    if len(xs) == 0:
                        search_radius = int(search_radius * 1.35)
                        continue

                    index = randrange(len(xs))
                    point = (xs[index] + ox - search_radius, ys[index] + oy - search_radius)
                    break
            
            filtered_coords.append(point)
            self.loot_click_locations.append(point)

        return filtered_coords

    def structural_pan(self, direction, distance, room_bounds):
        pan_step = 200
        
        progress_log(f'Starting panning, location: {room_bounds}')
        while distance > 0:
            current_distance = min(pan_step, distance)
            distance -= pan_step 

            self.pan_camera(direction, distance=current_distance)
            rescan_bounds = self.get_panning_bounds(room_bounds, direction).get_scaled_from_center(scale=1.2)

            structural, _ = detect_structural(self.next_frame()[rescan_bounds.to_slice()])
            rooms = [x.offset(rescan_bounds.low_pos) for x in detect_structural_rooms(structural)]
            if rooms == []: return None
            room_bounds = rooms[np.argmin([np.linalg.norm(np.array(room_bounds.pos) - x.pos) for x in rooms])]
            progress_log(f'New room location: {room_bounds}')

            # delta = np.linalg.norm(np.array(room_bounds.pos) - orig_pos)

    def pan_camera_for_looting(self):
        if self.loot_camera_position >= 2: return False
        progress_log(f'Starting camera pan for looting: {self.loot_camera_position}')
        bounds = self.current_room_bounds

        # fix camera zoom
        if self.dialogue_detected and self.normal_room_height is not None:
            # actual / base : high -> low
            room_height_threshold = 1.05 # this or lower
            progress_log(f'Starting zoom out, threshold: {room_height_threshold}')
            
            while bounds.height / self.normal_room_height <= room_height_threshold:
                self.zoom_camera(*bounds.pos, zoom_duration=0.05, zoom_out=True)
                structural, _ = detect_structural(self.next_frame())
                rooms = detect_structural_rooms(structural)
                rooms = [x for x in rooms if x.contains_point(bounds.pos)]
                if rooms == []:
                    result_log(f'Error: lost target room')
                    return False

                bounds = rooms[0]
            progress_log(f'Zoom in complete')

        # pan left
        if self.loot_camera_position == 0:
            room_ratio = self.current_room_bounds.width / self.current_room_bounds.height
            obscure_fraction = (room_ratio - 0.6) ** 2 / 100
            self.obscure_amount = int(obscure_fraction * self.current_room_bounds.width)

            pan_distance = self.current_room_bounds.x_min + self.obscure_amount
            self.current_room_bounds = self.structural_pan('left', pan_distance, bounds)
        # pan right
        else:
            pan_distance = self.screen_shape_xy[0] - self.current_room_bounds.x_max + self.obscure_amount
            self.current_room_bounds = self.structural_pan('right', pan_distance, bounds)

        if self.current_room_bounds:
            result_log(f'Error: lost room during panning')
            return False
        draw_border(get_do(), self.current_room_bounds, np.array([255, 150, 0, 255]), thickness=5)

        self.loot_camera_position += 1
        return True

    def loot_collection_task(self):
        progress_log(f'Starting loot collection')
        # enemies leave more loot, increase attempt count
        scan_attempts = 3 if self.enemies_detected else 2

        self.loot_click_locations = []
        self.loot_camera_position = 0

        if not self.loot_bound_detected:
            progress_log(f'Waiting for loot bound detection...')
            while not self.loot_bound_detected: sleep(0.1)

        # TODO: figure out whether we want to delay loot collection if hide_ui_panels returns true (for panels to not be in frames)
        self.hide_ui_panels()
        draw_border(get_do(), self.current_room_bounds, np.array([255, 0, 0, 255]), thickness=5)
        for attempt in range(15): # hard cap attempt count at 15
            progress_log(f'Scan attempt: #{attempt + 1}')

            loot_coords = self.do_detect_loot()
            scan_attempts -= 1

            if len(loot_coords) == 0:
                if scan_attempts < 1:
                    if self.pan_camera_for_looting():
                        scan_attempts += 1
                        continue
                    break
                sleep(0.25) # small delay to get new frames
                continue
            
            loot_coords = self.filter_loot_coords(loot_coords)
            if self.loot_collection_cancelled: return
            self.do_collect_loot(loot_coords)
            loot_collection_stop_time = perf_counter()

            # try to detect loot pickup animation. Timeout: 3.5s since after that delay all animations finish
            frame_iterator = self.iterate_latest_frames(min_frame_delta=0.2, yield_timestamp=True, max_runtime=3.5)
            last_frame = next(frame_iterator)[0]
            loot_collected = False
            for frame, timestamp in frame_iterator:
                if timestamp <= loot_collection_stop_time: continue # too old frame
                if loot_collected: continue # wait for max_runtime to run out
                # If no animation in first 1.5s, no need to wait any longer
                if timestamp - loot_collection_stop_time > 1.5:
                    progress_log(f'<Loot not collected>')
                    break

                magnitude = compute_diff_magnitude(frame, last_frame)
                # progress_log(f'Looking for diff spike: {magnitude} : {timestamp - loot_collection_stop_time} s')
                # seen magnitude as low as 0.72
                if magnitude > 0.95:
                    progress_log(f'<Loot collection confirmed: {magnitude:0.1f}>')
                    debug_log_image(last_frame, f'loot-collected-last-frame', increment_counter=False)
                    debug_log_image(frame, f'loot-collected-new-frame')
                    scan_attempts += 1 # always good to retry until no loot is left
                    loot_collected = True
                    self.loot_click_locations = []
                last_frame = frame
            
            if self.loot_collection_cancelled: return
            self.hide_ui_panels() # doing the last to not influence diff detection
            if scan_attempts < 1:
                if self.pan_camera_for_looting():
                    scan_attempts += 1
                    continue
                
                break
        
        self.loot_collected = True
    
    def loot_collection_state(self):
        if not self.loot_collected:
            self.loot_collection_cancelled = False
            self.loot_collection_task()
        self.script_state = self.room_scan_state

    def mission_script(self):
        progress_log('>>> Starting mission script')

        # state initialization
        self.script_running = True                      # is game automation script running now
        self.script_complete = False                    # is script complete and should be terminated? (usually inverse of script_running)
        self.room_scan_retry_start = None               # start with normal room detection
        self.last_iteration_room_detected = False       # has an unvisited room been detected on previous iteration?
        self.script_state = self.room_scan_state        # current state function of the mission script
        self.state_index = 0                            # increments each time we switch the state
        self.mission_paused = False

        progress_log('>>> Starting capture thread')
        self.start_capture_thread()

        # ensure the first iteration performs zoom out
        self.zoomed_out = False
        self.need_zoom_out = True

        last_state = None
        progress_log('>>> Mission loop starts')
        while not self.script_complete:
            if self.mission_paused:
                progress_log('>>> Script is paused now')
                while self.mission_paused: sleep(0.1)
                progress_log('>>> Resuming the mission script')
            
            last_state = self.script_state
            self.script_state()
            if self.script_state is not last_state: self.state_index += 1

        self.state_overlay = self.state_overlay_mission_complete
        self.update_overlay()
        progress_log('>>> Mission script complete')

    def toggle_mission_pause(self):
        self.mission_paused = not self.mission_paused

    ###
    ### Capture worker and frame utils
    ###

    def start_capture_thread(self):
        if self.capture_thread is not None: return
        self.screen_frames_lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self.capture_worker)
        self.capture_thread.start()
        while self.latest_frame is None: pass # get capture thread started

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
        if len(buffer_copy) == 0: return buffer_copy

        i = -2 # index -1 should never be deleted, start with -2
        while -i < len(buffer_copy):
            while -i <= len(buffer_copy) and buffer_copy[i + 1][1] - buffer_copy[i][1] < min_frame_delta:
                del buffer_copy[i]
            i -= 1

        if max_total_duration < 0:
            return buffer_copy

        end_time = buffer_copy[-1]
        for i in range(len(buffer_copy)):
            if end_time[1] - buffer_copy[i][1] <= max_total_duration:
                return buffer_copy[i:]
        
        raise Exception('I think this should not be reached? idk')

    def iterate_latest_frames(self, min_frame_delta=1e-6, max_frames_count=None, max_duration=None, wait_for_next=False, yield_timestamp=False, include_oldest=False, max_runtime=None):
        """Generator for frames: returns every captured frame starting from now with specified frame delta, allowing for long processing for each frame, 
        limited by buffer duration. Thus, it allows to process every frame as long as processing takes sub 3 seconds (with current buffer settings)

        Arguments:
            min_frame_delta: float - minimum time in seconds between two consecutive frames yielded. By default yields every frame (as long as FPS is under 1000000)
            max_frames_count: int - maximum number of frames yielded by this generator
            max_duration: float - maximum delta time between the first and the last yielded frames. When this is exceeded, generator stops yielding frames
            wait_for_next: bool - if True, generator will wait for the next frame to be captured before yielding the first frame. Default: False
            yield_timestamp: bool - if True, yields tuples of (frame, timestamp: float), otherwise yields just frame. Default: False
            include_oldest: bool - if True, generator will also yield frames captured prior to generator start (from the frames buffer). Default: False
            max_runtime: float - maximum time the generator should run for. Do note that the factual runtime might end up significantly higher, depending on processing time
        """
        # this is to avoid yielding the same frame multiple times
        if min_frame_delta <= 0: raise Exception('min_frame_delta should be positive')
        if max_frames_count is None: max_frames_count = float('inf')
        if max_duration is None: max_duration = float('inf')
        if max_runtime is None: max_runtime = float('inf')
        if max_frames_count < 1 or max_duration <= 0: return
        if wait_for_next: self.next_frame()

        # get latest frame data
        start_time = perf_counter()
        start_frame, start_frame_time, _ = self.copy_frames_buffer()[0 if include_oldest else -1]
        last_frame_time = start_frame_time
        frames_buffer = []
        frames_yielded = 0  
        if not include_oldest: # if need oldest frames, need to buffer them before yielding the first frame
            yield (start_frame, start_frame_time) if yield_timestamp else start_frame
            frames_yielded += 1  
        else:
            frames_buffer.append((start_frame, start_frame_time))

        while frames_yielded < max_frames_count:
            # buffer all available frames
            current_frames = self.copy_frames_buffer()
            for frame, time, _ in current_frames:
                if time < last_frame_time + min_frame_delta: continue
                frames_buffer.append((frame, time))
                last_frame_time = time
            
            del current_frames
            if len(frames_buffer) == 0:
                sleep(max(0, min_frame_delta - perf_counter() + last_frame_time))
                continue
            
            # yield next frame from buffer
            next_frame, next_time = frames_buffer[0]
            if next_time - start_frame_time > max_duration or perf_counter() - start_time > max_runtime: return
            yield (next_frame, next_time) if yield_timestamp else next_frame
            if next_time - start_frame_time + min_frame_delta > max_duration or perf_counter() - start_time > max_runtime: return
            del frames_buffer[0]
            frames_yielded += 1

    ### 
    ### App initialization
    ###

    def __init__(self, force_pil_capture=False, mock_frames_path=None, disable_image_logging=False, fixed_mock_frame=False):
        self.dialogue_handlers = {
            'random': self.dialogue_random_handler,
            'manual': self.dialogue_manual_handler
        }
        
        # Entries format:  <character>: (function, name/title, repeat_execution, [no_new_thread])
        self.execution_target_chord_map = {
             # replace with whatever you need at the time
            # '`': (lambda x: debug_detect_generic_loot(grab_screen_func=x.no_overlay_grab_screen, mock_mode=self.mock_mode), 'temp debug function', False),
            '`': (FalloutShelterAutomationApp.build_initial_game_map, 'temp debug function', False),
            'm': (lambda x: detect_med_buttons(x.no_overlay_grab_screen()), 'meds detection', True),
            'c': (lambda x: detect_critical_button(x.no_overlay_grab_screen()), 'critical cue detection', True),
            'r': (lambda x: detect_rooms(x.no_overlay_grab_screen()), 'rooms detection', True),
            'b': (FalloutShelterAutomationApp.debug_start_battle_detect, 'battle cues detection', True),
            'p': (FalloutShelterAutomationApp.debug_start_visualizing_diff, 'diff detection', True),
            'f': (lambda x: detect_structural(x.no_overlay_grab_screen()), 'structural detection', True),
            'e': (lambda x: detect_enemies(x.no_overlay_grab_screen()), 'enemies detection', True),
            'h': (FalloutShelterAutomationApp.debug_start_detect_structural_rooms, 'structural room detection', False), # return
            'n': (lambda x: detect_dialogue_buttons(x.no_overlay_grab_screen(), x.screen_shape_xy), 'dialogue button detection', True),
            '/': (lambda x: detect_loot(x.screen_bounds, x.no_overlay_grab_screen), 'loot detection', False), # no repeat
            't': (FalloutShelterAutomationApp.debug_start_visualizing_motion_diff, 'motion diff detection', True),
            '.': (FalloutShelterAutomationApp.debug_toggle_capture_thread, 'thread toggle', False),
            '+': (FalloutShelterAutomationApp.debug_dump_motion_diff_data, 'motion data dump', False),
            '-': (FalloutShelterAutomationApp.debug_test_iou, 'iou test', True),
            '=': (FalloutShelterAutomationApp.toggle_mission_pause, 'Pause/resume mission script', False, True),
        }
        "Maps characters on keyboard to functions to start upon chord completion"
        self.execution_target_map = { # '<char>': (function, message, needs_debug_frame?)
            ']': (FalloutShelterAutomationApp.next_mock_frame_directory, 'Next mock frame directory', False),
            '[': (FalloutShelterAutomationApp.prev_mock_frame_directory, 'Previous mock frame directory', False),
            '0': (FalloutShelterAutomationApp.debug_reset_mock_frames, 'Reset mock frame index', False)
        }

        self.keyboard_input = keyboard.Controller()
        self.mouse_input = mouse.Controller()
        self.recent_click_coords = []
        self.recent_click_timestamps = []
        self.loot_click_locations = []
        self.latest_frame = None
        self.camera_pan_log = []

        self.max_screen_frames = 250                     # kind of arbitrary (~1500 MB for 1920x1080 frames)
        self.screen_frame_max_age = 3                    # (seconds) (probably can be reduced, 3 seconds for debug reasons)
        self.capture_thread = None
        self.force_pil_capture = force_pil_capture
        self.mock_frames_path = mock_frames_path
        self.disable_image_logging = disable_image_logging
        self.fixed_mock_frame = fixed_mock_frame

        if fixed_mock_frame and mock_frames_path is None:
            print('Error: cannot use fixed mock frame if no frame path is specified')
            quit()

    def run(self):
        print('Welcome to FSA! Initializing...')
        
        native_capture = not self.force_pil_capture
        if self.mock_frames_path is not None:
            mock_mode = 'fixed' if self.fixed_mock_frame else 'frames'
            self.grab_screen, self.native_grab_screen = init_screen_capture(mode='mock', mock_directory=self.mock_frames_path, mock_mode=mock_mode)
            set_max_debug_frames(10)
            self.mock_mode = True
            set_debug_log_images_enabled(enabled=False) # do not log frames
            self.log_mock_frames_path()
            self.set_screen_shape(np.array(get_mock_frames()[0].shape[:2]))
        else:
            self.grab_screen, self.native_grab_screen = init_screen_capture(mode='real', window_title=self.capture_name, use_native=native_capture)
            self.set_screen_shape(np.array(self.grab_screen(None).shape[:2]))
            if self.disable_image_logging:
                set_debug_log_images_enabled(enabled=False)

        if self.native_grab_screen:
            self.no_overlay_grab_screen = self.grab_screen
            result_log('Native screen capture initialized')
        elif not native_capture:
            result_log('Forced PIL screen capture')
        elif self.mock_mode:
            self.no_overlay_grab_screen = self.grab_screen
            result_log('Mock screen capture initialized')

        if not self.mock_mode and not self.disable_image_logging: init_output_directory()

        # Initialize UI
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-transparentcolor','#f0f0f0')
        self.root.attributes("-topmost", True)   

        self.panel = tk.Label(self.root)
        self.log_label = tk.Label(self.root, text='Error text...', background='#000000', foreground='#44dd11', justify='left', font=('Segoe UI', self.logs_font_size))
        self.log_label.pack(side=tk.LEFT)

        self.status_label = tk.Label(self.root, text='<no data>', background='#000000', foreground='#d0d0d0', justify='right', font=('Segoe UI', self.logs_font_size))
        self.status_label.pack(side=tk.BOTTOM, anchor=tk.SE)

        if os.path.exists('./resources/test_overlay.png'):
            with Image.open('./resources/test_overlay.png') as img:
                self.update_overlay(np.array(img.resize(self.screen_shape_xy)))
        else:
            self.update_overlay(np.zeros(self.screen_shape, dtype=np.uint8))

        # Start background workers
        self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_listener_thread.start()

        self.status_label_thread = threading.Thread(target=self.status_label_worker)
        self.status_label_thread.start()

        self.root.after(self.app_update_interval_ms, self.update)
        
        # Start
        log(f'\n' \
            f'Fallout Shelter Automation: {self.version}\n' \
            f'Start mission script: Ctrl + F; Enter (Esc to terminate)\n' \
            f'Toggle log display: Ctrl + F; L\n' \
            f'Shutdown: Ctrl + F; Esc \n')

        self.root.mainloop()

    def set_screen_shape(self, screen_shape_yx):
        self.screen_shape = screen_shape_yx
        self.screen_shape_xy = self.screen_shape[::-1]
        override_debug_frame_shape(self.screen_shape)
        progress_log(f'Using screen shape: {self.screen_shape_xy}')

        self.screen_center_xy = self.screen_shape_xy // 2
        self.screen_bounds = Bounds.from_rect(0, 0, *self.screen_shape_xy)
        self.state_overlay_ok = np.ones((*self.screen_shape, 4), dtype=np.uint8) * np.array([255, 0, 0, 255])
        self.state_overlay_ok[1:-1,1:-1] = 0
        self.state_overlay_error = np.ones((*self.screen_shape, 4), dtype=np.uint8) * np.array([180, 255, 0, 255])
        self.state_overlay_error[2:-2,2:-2] = 0
        self.state_overlay_mission_complete = np.ones((*self.screen_shape, 4), dtype=np.uint8) * np.array([60, 210, 10, 255])
        self.state_overlay_mission_complete[3:-3,3:-3] = 0
        draw_circle(self.state_overlay_mission_complete, self.screen_center_xy, self.screen_shape_xy[1] // 8, np.array([60, 210, 10, 255]), 15)
        self.state_overlay = self.state_overlay_ok

        sx, sy = self.screen_center_xy
        sw, sh = (self.screen_shape_xy * self.camera_pan_deadzone_size / 2).astype(int)
        self.camera_deadzone = Bounds(sx - sw, sy - sh, sx + sw, sy + sh)

        if self.mock_mode:
            create_debug_frame()
            get_do()[:, :, :3] = get_mock_frames()[0]
            get_do()[:, :, 3] = 255
            create_debug_frame()

        create_debug_frame()

    ####
    #### Technical util functions
    ####

    def show_overlay(self, do_show=True):
        "Enables FSA overlay and on-screen log (if logs are not disabled specifically). If `do_show` is False, no work is done"
        if not do_show: return
        self.panel.place(relheight=1, relwidth=1)
        if self.show_log: self.log_label.pack(side='left')
        self.overlay_on = True

    def hide_overlay(self): 
        "Disables FSA overlay and on-screen logs regardless of current state. Returns bool: whether overlay was enabled before this call"
        # should we also hide status label? surely not
        self.panel.place_forget()
        self.log_label.pack_forget()
        was_on = self.overlay_on
        self.overlay_on = False
        return was_on

    def no_overlay_grab_screen(self, bbox=None):
        restore_overlay = self.hide_overlay()
        sleep(0.06) # last tried: 0.05, sometimes overlay still gets captured
        screen = self.grab_screen(bbox)
        self.show_overlay(restore_overlay)
        return screen

    def no_overlay_grab_screen_native(self, bbox=None):
        return self.grab_screen(bbox)

    def update_overlay(self, image=0, autoshow=True):
        image_data = np.clip(image + self.state_overlay, 0, 255).astype(np.uint8)

        with io.BytesIO() as output:
            Image.fromarray(image_data).save(output, format='PNG')
            new_overlay = tk.PhotoImage(data=output.getvalue(), format='png')
            self.panel.configure(image=new_overlay)
            self.panel.image = new_overlay

        self.show_overlay(autoshow)
    
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

    def mouse_click(self, x, y, protect_borders=True, auto_delay=True):
        """Perform a left mouse click at the specified coordinates. 
        
        Arguments:
            x: int - x coordinate, in pixels (left to right) of the click
            y: int - y coordinate, in pixels (top to bottom) of the click
            protect_borders: bool - when True, clicks that are too close to screen edge are aborted to avoid camera panning (due to cursor position)
            auto_delay: bool - when True, click is delayed if necessary to avoid activating a game double-click event
        """
        # camera panning protection
        if protect_borders and self.filter_mouse_coords(x, y, bool_mode=True):
            result_log(f'Warning: click at {x}, {y} aborted')
            return

        # clear old clicks (lists for double click protection)
        while len(self.recent_click_timestamps) > 0 and perf_counter() - self.recent_click_timestamps[0] >= safe_mouse_click_delay:
            del self.recent_click_timestamps[0]
            del self.recent_click_coords[0]
        
        # double click protection
        while auto_delay and len(self.recent_click_timestamps) > 0: # only using a loop because it supports break
            matches = np.linalg.norm(np.array(self.recent_click_coords) - (x, y), axis=1) <= double_click_activation_radius
            if not np.any(matches): break

            last_match_index = len(self.recent_click_coords) - np.argmax(matches[::-1]) - 1
            delay = max(0, safe_mouse_click_delay - (perf_counter() - self.recent_click_timestamps[last_match_index]))
            progress_log(f'Warning: auto click delay for {int(1000 * delay)} ms')
            sleep(delay)
            break

        restore_overlay = self.hide_overlay() # so we don't click overlay instead :)
        self.mouse_input.position = (x, y)
        sleep(0.07) # mostly to wait for overlay to disappear. Lower values are not recommended
        self.mouse_input.press(mouse.Button.left)
        sleep(mouse_click_duration) # some delay is required for the game to register a click apparently
        self.mouse_input.release(mouse.Button.left)
        self.show_overlay(restore_overlay)

        self.recent_click_coords.append((x, y))
        self.recent_click_timestamps.append(perf_counter())

    def pan_camera(self, direction, duration=None, distance=None, post_pan_sleep=True):
        if duration is None and distance is None:
            distance = self.camera_default_pan_distance
        
        if distance is not None:
            if duration is not None: raise Exception('duration and distance cannot both be set')
            duration = distance / self.camera_pan_velocity
        
        if duration < 0:
            result_log(f'Warning: cannot pan for {duration:0.1f} s, aborting')
            return

        self.keyboard_input.press(CAMERA_PAN_KEYS[direction])
        sleep(duration)
        self.keyboard_input.release(CAMERA_PAN_KEYS[direction])
        if post_pan_sleep: sleep(camera_post_pan_duration)

        self.camera_pan_log.append((direction, duration * self.camera_pan_velocity))

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
        if self.tick_counter % self.overlay_update_interval == 0: self.display_debug()

        self.log_label.config(text=get_current_log())
        if self.current_execution_target is not None and not self.task_in_progress:
            self.state_overlay = self.state_overlay_ok
            self.update_overlay()

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

        if key.char in self.execution_target_chord_map:
            if len(self.execution_target_chord_map[key.char]) == 4 and self.execution_target_chord_map[key.char][3]:
                func, message, repeat, _ = self.execution_target_chord_map[key.char]
                if message is not None: result_log(f'Executing {message}...')
                create_debug_frame()
                func(self)
                self.display_debug()
                return

            func, message, repeat = self.execution_target_chord_map[key.char]
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

    def handle_keyboard_press(self, key):
        if self.script_running and key == keyboard.Key.esc and not self.mask_escape_key: 
            self.terminate() # esc during mission script should abort execution (kill-switch)

        if not hasattr(key, 'char'): return
        if key.char in self.execution_target_map:
            func, message, visual_debug = self.execution_target_map[key.char]
            if message is not None: result_log(f'Executing {message}...')
            if visual_debug: create_debug_frame()
            func(self)
            if visual_debug: self.display_debug()

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
            self.last_key_pressed = key

            # exception == keyboard listener dies, so have to catch that 
            try:
                if self.keyboard_chord_pending:
                    self.handle_keyboard_chord(key) 
                else:
                    self.handle_keyboard_press(key)
            except:
                progress_log('Keyboard listener thread encountered exception:')
                traceback.print_exc()
            
            if self.terminate_pending: return False # should be AFTER chord handler
            for_canonical(chord_start_hotkey.press)(key)

        def keyboard_on_release(key):
            for_canonical(chord_start_hotkey.release)(key)
    
        with keyboard.Listener(on_press=keyboard_on_press, on_release=keyboard_on_release) as l:
            l.join()

    # CPU usage this function reports on my machine looks like it's randomly generated, but whatever, it's fun to look at i guess
    def status_label_worker(self):
        "Worker that updates status label"
        this_process = psutil.Process(os.getpid())

        # interval is smaller to terminate quickly
        for i in slow_loop(interval=0.1):
            if self.terminate_pending: return
            if i % 10 != 0: continue
            
            status = '[Error]'
            try:
                system_cpu = psutil.cpu_percent()
                process_cpu = this_process.cpu_percent() / psutil.cpu_count()
                timing_string = f'CPU: {system_cpu:0.1f}% ({process_cpu:0.1f}%) [{threading.active_count()}]'
                capture_string = ' capture: '
                if self.capture_thread is None:
                    capture_string += 'disabled'
                else:
                    frames = self.copy_frames_buffer(max_total_duration=2)
                    duration = frames[-1][1] - frames[0][1]
                    fps = len(frames) / duration if duration > 0 else float('NaN')
                    capture_string += f'{fps:0.1f} FPS'

                status = timing_string + capture_string
            except:
                print('Status worker error:')
                traceback.print_exc()
            
            self.status_label.config(text=status)

    ####
    #### Debug functions
    ####

    def log_mock_frames_path(self):
        "Log a message saying the current mock frames path"
        progress_log(f'Mock frames from: {Path(get_current_mock_path()).name}')

    def next_mock_frame_directory(self):
        if not self.mock_mode:
            result_log('Warning: Not in mock mode')
            return
            
        load_next_mock_path()
        self.set_screen_shape(np.array(get_mock_frames()[0].shape[:2]))
        self.log_mock_frames_path()
    
    def prev_mock_frame_directory(self):
        if not self.mock_mode:
            result_log('Warning: Not in mock mode')
            return

        load_next_mock_path(previous=True)
        self.set_screen_shape(np.array(get_mock_frames()[0].shape[:2]))
        self.log_mock_frames_path()

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

        buffer_len = len(self.screen_frames)
        full_buffer = self.copy_frames_buffer()
        sparse_buffer = self.copy_frames_buffer(min_frame_delta=0.5)
        result_log(f'Got buffer lengths: {buffer_len} vs. {len(full_buffer)}')

        min_time = full_buffer[0][1]
        for i, frame_data in enumerate(full_buffer):
            frame, abs_time, delta_time = frame_data
            is_in_sparse = len([x for x in sparse_buffer if x[0] is frame]) > 0
            debug_log_image(frame, f'frame-dump-{i}-time-{abs_time-min_time:0.2f}{"-SELECTED" if is_in_sparse else ""}')
        
        result_log(f'Frames dumped')

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
        fragments, _, _, _ = detect_fragments(mask, mask, point_count=True)
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

    def debug_reset_mock_frames(self):
        reset_mock_frame_index()
        result_log(f'Mock frame index has been reset')

    def debug_test_double_click(self):
        for _ in range(50):
            x, y = randrange(820, 980), randrange(820, 980)
            print(f'Clicking: {x}, {y}')
            self.mouse_click(x, y)

    def debug_test_loot_click_dispersion(self):
        update_interval = self.overlay_update_interval
        self.overlay_update_interval = 3
        for _ in range(300):
            point = self.filter_loot_coords([(450, 300)])[0]
            print(f'New coords: {point}')
            draw_disk(get_do(), point, 9, get_debug_color(randrange(100000000)))
            sleep(0.05)
        self.overlay_update_interval = update_interval

    def debug_test_instant_pan(self):
        self.pan_camera('left', duration=0)
        sleep(1)
        self.pan_camera('right', duration=0)
        sleep(1)
        self.pan_camera('up', duration=0)
        sleep(1)
        self.pan_camera('down', duration=0)

###
### static void main string args
###

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fallout shelter automation')
    parser.add_argument('--force-pil-capture', action='store_true', help='Force screen capture using PIL (no native window capturing)')
    parser.add_argument('--mock-frames-path', type=str, help='Specify directory for mock frames (also enables mock screen capture). Either provide path to folder with frames, or to folder of folders with frames')
    parser.add_argument('--disable-image-logging', action='store_true', default=False, help='Disables logging images as files')
    parser.add_argument('--fixed-mock-frame', action='store_true', help='When using mock mode, allows to use a single frame repeatedly')
    args = parser.parse_args()

    app = FalloutShelterAutomationApp(force_pil_capture=args.force_pil_capture, mock_frames_path=args.mock_frames_path, disable_image_logging=args.disable_image_logging, fixed_mock_frame=args.fixed_mock_frame)
    app.run()
