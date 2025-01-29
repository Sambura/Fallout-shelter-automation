from PIL import Image
import threading
import datetime
import os

debug_log_images = True
debug_show_progress_log = True
debug_show_result_log = True

VISUAL_DEBUG_NONE = 0
VISUAL_DEBUG_RESULT = 1
VISUAL_DEBUG_PROGRESS = 2

visual_debug_level = VISUAL_DEBUG_RESULT
extra_visual_debug = True

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

def make_output_filename(filename):
    return f'{output_folder}/{filename}'

def log_image(img, postfix, increment_counter=True):
    global output_index
    filename = make_output_filename(f'{output_index}-{postfix}.png')
    img_copy = img.copy()
    if increment_counter: output_index += 1
    save_thread = threading.Thread(target=lambda: Image.fromarray(img_copy).save(filename))
    save_thread.start()

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

def set_debug_log_images_enabled(enabled=True, verbose=True):
    global debug_log_images
    debug_log_images = enabled
    if verbose:
        progress_log(f'{"Enabled" if enabled else "Disabled"} debug image logging')
