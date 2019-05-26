import time
import inspect
import numpy as np
import matplotlib.pyplot as plt

IMDIR = '../Images/'


class Garcon:
    def __init__(self):
        self.start_time = time.time()
        self.fig = None

    def show_time(self):
        elapsed_time = time.time() - self.start_time
        self.log('Execution took {0:.2f} seconds.'.format(elapsed_time))

    def log(self, *args):
        print('Log:', end=' ')
        for arg in args:
            print(arg, end=' ')
        print()

    def enter_func(self):
        curr_frame = inspect.currentframe()
        call_frame = inspect.getouterframes(curr_frame, 2)
        self.log(f'In {call_frame[1][3]}')

    def init_plt(self, title=''):
        self.fig = plt.figure()
        if title:
            plt.title(title)

    def init_subplt(self, title):
        plt.subplot()
        plt.title(title)

    def save_plt(self, fn=''):
        if not fn:
            curr_frame = inspect.currentframe()
            call_frame = inspect.getouterframes(curr_frame, 2)
            fn = call_frame[1][3]
        plt.tight_layout()
        plt.savefig(IMDIR + fn)

    def __del__(self):
        self.show_time()
