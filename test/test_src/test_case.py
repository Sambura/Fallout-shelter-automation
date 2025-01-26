class TestCase:
    def __init__(self, test_index, test_name, test_suite_name):
        self.test_index = test_index
        self.test_name = test_name
        self.valid = False
        self._error = f'{self.test_name}: base class instance exception'
        self.test_executed = False
        self.test_suite_name = test_suite_name
        self.optional_test = False
    
    def run_test(self):
        raise NotImplementedError('A base class run_test was called which is illegal')

    def get_error(self):
        return self._error
