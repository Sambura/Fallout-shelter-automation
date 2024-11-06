from PIL import Image
import threading
import datetime
import os

debug_show_progress_visuals = True
debug_log_images = True
debug_show_progress_log = True
debug_show_result_visuals = True
debug_show_result_log = True
log_limit = 10000000

current_log = []
output_folder = None
output_index = 0

def init_output_directory(path=None):
    global output_folder, output_index
    if path is None:
        launch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f'output/{launch_time}/'

    os.makedirs(path)
    result_log(f"Output folder created: {path}")
    output_folder = path
    output_index = 0

def log_image(img, postfix, increment_counter=True):
    global output_index
    filename = f'{output_folder}/{output_index}-{postfix}.png'
    img_copy = img.copy()
    if increment_counter: output_index += 1
    save_thread = threading.Thread(target=lambda: Image.fromarray(img_copy).save(filename))
    save_thread.start()

# maybe add some option to disable debug logs specifically
def debug_log_image(img, postfix, increment_counter=True):
    if not debug_log_images: return
    log_image(img, postfix, increment_counter)

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

H_DIRECTIONS = ['left', 'right']
V_DIRECTIONS = ['up', 'down']

def are_opposite_directions(direction, other_direction):
    if set([direction, other_direction]) == set(H_DIRECTIONS): return True
    if set([direction, other_direction]) == set(V_DIRECTIONS): return True
    return False

def is_vertical(direction: str) -> bool: return direction in V_DIRECTIONS
def is_horizontal(direction: str) -> bool: return direction in H_DIRECTIONS
