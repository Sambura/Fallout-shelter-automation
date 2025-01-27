import sys
sys.path.append("..")

from .test_case import TestCase
from src.vision import *
from src.vision import _is_fragment_rectangular
from src.fallout_shelter_vision import detect_generic_loot, compute_loot_masks, detect_corpse_loot

import re
import os
import numpy as np
from pathlib import Path
from PIL import Image

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

def debug_show_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

class LootTestCase(TestCase):
    def __init__(self, test_path):
        def check_and_load_image(path):
            if not os.path.exists(path) or not os.path.isfile(path):
                self._error = f'Couldn\'t find file {path}'
                return None
            
            return np.array(Image.open(path).convert('RGB')).astype(np.uint8)
            
        test_name = Path(test_path).name
        test_suite_name = Path(test_path).parent
        test_index = int(test_name.split('-')[0])
        super().__init__(test_index, test_name, test_suite_name)
        code_name = test_name.split('-')[1]

        self.frame1 = check_and_load_image(os.path.join(test_path, '01-frame.png'))
        self.frame2 = check_and_load_image(os.path.join(test_path, '02-frame.png'))
        self.check_frame = check_and_load_image(os.path.join(test_path, 'autotest-frame.png'))
        self.false_positive_test = 'N' in code_name
        self.optional_test = 'O' in code_name
        self.should_have_generic = 'G' in code_name
        self.should_have_corpse = 'C' in code_name

        if self.frame1 is None or self.frame2 is None or (self.check_frame is None and not self.false_positive_test):
            self.valid = False
            return
        
        self.valid = True
        self.initialized = False
    
    def init_test(self):
        if self.initialized: return self.valid
        self.initialized = True
        self.generic_loot_fragments = []
        self.corpse_loot_fragments = []

        if self.false_positive_test: return True

        red, green, blue = self.check_frame[:,:,0], self.check_frame[:,:,1], self.check_frame[:,:,2]

        red_fragments = detect_fragments(red, red == 255)[0]
        green_fragments = detect_fragments(green, green == 255)[0]
        blue_fragments = detect_fragments(blue, blue == 255)[0]
        fragments = red_fragments + green_fragments + blue_fragments

        if len(fragments) % 2 == 1:
            self._error = 'Odd number of fragments, should be even'
            return False

        while len(fragments) > 0:
            base_fragment = fragments[0]
            base_fragment.compute(masked_patch=True)
            fragments.remove(base_fragment)
            holes = detect_fragments(base_fragment.masked_patch, base_fragment.masked_patch == 0, patch_mask=True)[0]
            inner_hole = None
            for hole in holes:
                hole.bounds = hole.bounds.offset(base_fragment.bounds.low_pos)
            
                if not base_fragment.bounds.contains_bounds(hole.bounds, strict=True): continue
                if inner_hole is not None:
                    self._error = f'Fragment {base_fragment.bounds} has multiple inner holes'
                    return False

                inner_hole = hole

            if inner_hole is None:
                continue # probably an inner fragment, just remove from list
            
            for fragment in fragments[1:]:
                if inner_hole.bounds.contains_bounds(fragment.bounds):
                    fragments.remove(fragment)
                    break
            
            score = inner_hole.point_count / inner_hole.bounds.area
            inner_hole.points += base_fragment.bounds.low_pos[::-1]

            base_fragment.unite_with(inner_hole)
            base_fragment.compute(patch_mask=True)
            if score >= 0.8:
                self.generic_loot_fragments.append(base_fragment)
            else:
                self.corpse_loot_fragments.append(base_fragment)
        
        if len(self.generic_loot_fragments) == 0 and self.should_have_generic:
            self._error = 'Test should contain generic loot, but none were detected'
            return False
        
        if len(self.corpse_loot_fragments) == 0 and self.should_have_corpse:
            self._error = 'Test should contain corpse loot, but none were detected'
            return False
        
        return True

    def run_test(self):
        self.valid = self.init_test()
        if not self.valid: return False
        
        self.test_executed = True
        def match_results(target_fragments, results, label):
            mask = np.zeros_like(results)

            for fragment in target_fragments:
                crop = results[fragment.bounds.to_slice()]
                if not np.any(crop & fragment.patch_mask):
                    self.failure_reason += f'Failed to detect one or more {label} loots. '
                    return False
                mask[fragment.points[:, 0], fragment.points[:, 1]] = True
            
            if np.any(results & mask != results):
                self.failure_reason += f'False positive on {label} loot. '
                return False

            return True

        sources = compute_loot_masks(self.frame1, self.frame2)
        generic_results = detect_generic_loot(self.frame1, self.frame2, *sources)['result']
        corpse_results = detect_corpse_loot(self.frame1, self.frame2, *sources)['result']

        self.failure_reason = ''
        pass_1 = match_results(self.generic_loot_fragments, generic_results, 'generic')
        pass_2 = match_results(self.corpse_loot_fragments, corpse_results, 'corpse')

        self.test_pass = pass_1 and pass_2
        return self.test_pass

def loot_collection_make_tests(path):
    print(f'Looking for loot tests in {path}')
    children = [os.path.join(path, x) for x in os.listdir(path)]
    test_paths = [x for x in children if os.path.isdir(x)]

    tests = []
    for test_path in test_paths:
        if not re.match(r'\d+-(G?C?|N?)O?', Path(test_path).name):
            print(f'{YELLOW}Warning: ignoring directory {test_path}{RESET}')
            continue

        test = LootTestCase(test_path)
        if not test.valid:
            error = test.get_error()
            print(f'{YELLOW}Error: {error}. Skipping test {test.test_name}{RESET}')
            continue

        tests.append(test)
        
    return tests
