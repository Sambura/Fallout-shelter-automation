from test_src.loot_collection import *
from test_src.test_case import TestCase

import argparse
import traceback
from time import perf_counter

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

def main(loot_collection_path=None, test_indices=[]):
    print('Starting FSA tests...')
    tests = []

    if loot_collection_path is not None:
        loot_tests = loot_collection_make_tests(loot_collection_path)
        tests += loot_tests

    if len(test_indices) == 0:
        print('Note: Running all tests')
        test_indices = [x.test_index for x in tests]
    else:
        print('Note: Running subset of tests')

    if len(test_indices) == 0:
        print(f'No tests to run, exiting')
        quit()
    
    print()
    print(f'Tests loaded. About to run {len(test_indices)} tests')
    
    test_passes = 0
    test_fails = 0
    optional_fails = 0
    test_start = perf_counter()
    for index in test_indices:
        print(f'[TEST {index}]: ', end='')
        test = [x for x in tests if x.test_index == index]
        if len(test) == 0:
            test_fails += 1
            print(f'{RED}NOT FOUND{RESET}')
            continue
        
        test = test[0]
        error_color = YELLOW if test.optional_test else RED
        try:
            success = test.run_test()
        except:
            if test.optional_test:
                optional_fails += 1
            else:
                test_fails += 1
            print(f'{error_color}ERROR{RESET} ({test.test_suite_name}/{test.test_name})')
            traceback.print_exc()
            continue

        if success:
            test_passes += 1
            print(f'{GREEN}PASS{RESET}', end='')
        else:
            if test.optional_test:
                optional_fails += 1
            else:
                test_fails += 1
            print(f'{error_color}FAIL{RESET}', end='')
        
        print(f' ({test.test_suite_name}/{test.test_name})')

    test_end = perf_counter()
    if test_fails + optional_fails > 0:
        print()
        print('Failure reasons:')

        for test in tests:
            if not test.test_executed or test.test_pass: continue
            print(f'Test {test.test_name} failure: {test.failure_reason}')

    print()
    print(f'Finished testing in {test_end - test_start:0.2f} seconds. Breakdown:')
    print(f'{GREEN if test_passes > 0 else RESET}Passed: {test_passes} / {len(test_indices)}{RESET}')
    print(f'{YELLOW if optional_fails > 0 else RESET}Optional failed: {optional_fails} / {len(test_indices)}{RESET}')
    print(f'{RED if test_fails > 0 else RESET}Failed: {test_fails} / {len(test_indices)}{RESET}')
    print()
    overall_pass = test_fails == 0
    print(f'{(GREEN if optional_fails == 0 else YELLOW) if overall_pass else RED}Overall: {"PASS" if overall_pass else "FAIL"}{RESET}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSA tests')
    parser.add_argument('--loot-collection-path', type=str, default='loot_collection', help='Directory containing loot collection tests')
    parser.add_argument('--indices', type=str, default='', help='Comma separated list of tests to run (only if one test type is selected)')
    args = parser.parse_args()
    
    test_indices = [int(x) for x in args.indices.split(',') if x]
    main(args.loot_collection_path, test_indices)
