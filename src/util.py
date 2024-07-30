debug_show_progress_visuals = True
debug_show_progress_log = True
debug_show_result_visuals = True
debug_show_result_log = True

def progress_log(str):
    if debug_show_progress_log: print(f'[PDEBUG] {str}')

def result_log(str):
    if debug_show_result_log: print(f'[RDEBUG] {str}')

def are_opposite_directions(direction, other_direction):
    if set([direction, other_direction]) == set(['left', 'right']): return True
    if set([direction, other_direction]) == set(['up', 'down']): return True
    return False

def is_vertical(direction: str) -> bool: return direction == 'up' or direction == 'down'
def is_horizontal(direction: str) -> bool: return direction == 'left' or direction == 'right'
