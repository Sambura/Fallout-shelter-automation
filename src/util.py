from time import perf_counter, sleep
import numpy as np

H_DIRECTIONS: list[str] = ['left', 'right']
V_DIRECTIONS: list[str] = ['up', 'down']

def are_opposite_directions(direction: str, other_direction: str) -> bool:
    if set([direction, other_direction]) == set(H_DIRECTIONS): return True
    if set([direction, other_direction]) == set(V_DIRECTIONS): return True
    return False

def is_vertical(direction: str) -> bool: return direction in V_DIRECTIONS
def is_horizontal(direction: str) -> bool: return direction in H_DIRECTIONS

def opposite_direction(direction: str) -> str:
    direction = set([direction])
    directions = set(H_DIRECTIONS) - direction
    if len(directions) == 2:
        directions = set(V_DIRECTIONS) - direction

    return list(directions)[0]

def slow_loop(interval, max_iter_count=None, max_duration=None):
    "for x in slow_loop(...); interval in seconds. Infinite iterations by default"
    if max_iter_count is None: max_iter_count = float('inf')
    if max_duration is None: max_duration = float('inf')
    if max_iter_count < 1 or max_duration <= 0: return
    start_time = last_yield_time = perf_counter()
    i = 0
    while True:
        current_time = perf_counter()
        yield i
        last_yield_time = perf_counter()
        sleep(max(0, interval - (last_yield_time - current_time)))
        i += 1
        if i >= max_iter_count: return
        if perf_counter() - start_time >= max_duration: return

def rotate_vector_2d(vector, angle_deg):
    angle = np.radians(angle_deg)
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])

    return matrix @ vector
