import time
import datetime
import numpy as np
from sklearn import preprocessing as prep


class timer(object):
    def __init__(self, name='default'):
        """
        timer object to record running time of functions, not for micro-benchmarking
        usage is:
            $ timer = utils.timer('name').tic()
            $ timer.toc('process A').tic()


        :param name: label for the timer
        """
        self._start_time = None
        self._name = name
        self.tic()


    def tic(self):
        self._start_time = time.time()
        return self


    def toc(self, message):
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        print("[{0:s}] {1:s} elapsed [{2:s}]".format(self._name, message, timer.format(elapsed)))
        return self

    def reset(self):
        self._start_time = None
        return self


    @staticmethod
    def format(elapsed_time_in_sec):
        delta = datetime.timedelta(seconds=elapsed_time_in_sec)
        d = datetime.datetime(1, 1, 1) + delta
        s = ''
        if (d.day - 1) > 0:
            s = s + '{:d} days'.format(d.day - 1)
        if d.hour > 0:
            s = s + '{:d} hr'.format(d.hour)
        if d.minute > 0:
            s = s + '{:d} min'.format(d.minute)
        s = s + '{:d} s'.format(d.second)
        return s


def standardize(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 5 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    #x_nzrow = exist none zero row in x

    x_nzrow = x.any(axis=1)
    # StandardScaler : z-score function
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_2(x, cap=1.0):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 1 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :param cap: absolute value of min/max cap
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > cap] = cap
    x_scaled[x_scaled < -cap] = -cap
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled

