debug_show_progress_visuals = True
debug_show_progress_log = True
debug_show_result_visuals = True
debug_show_result_log = True
log_limit = 10000000

current_log = []

def log(str):
    global current_log
    current_log.append(str)
    truncate_log()
    print(str)

def progress_log(str):
    if debug_show_progress_log:
        log(f'[PDEBUG] {str}')

def result_log(str):
    if debug_show_result_log: 
        log(f'[RDEBUG] {str}')

def truncate_log():
    global current_log
    if len(current_log) > log_limit:
        current_log = current_log[log_limit // 2:]

def get_current_log(lines=60):
    return '\n'.join(current_log[-lines:])

def are_opposite_directions(direction, other_direction):
    if set([direction, other_direction]) == set(['left', 'right']): return True
    if set([direction, other_direction]) == set(['up', 'down']): return True
    return False

def is_vertical(direction: str) -> bool: return direction == 'up' or direction == 'down'
def is_horizontal(direction: str) -> bool: return direction == 'left' or direction == 'right'
