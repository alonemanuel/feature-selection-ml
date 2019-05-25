import time

class Garcon:
    def __init__(self):
        self.start_time = time.time()

    def show_time(self):
        elapsed_time = time.time() - self.start_time
        self.log(f'Execution took {elapsed_time} seconds.')

    def log(self, *args):
            print('Log:', end=' ')
            for arg in args:
                print(arg, end=' ')
            print()

    def __del__(self):
        self.show_time()