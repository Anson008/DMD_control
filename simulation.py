import numpy as np
import matplotlib.pyplot as plt
from binary_images import BinaryImage
from periods import PeriodGenerator
import math


class SimulationEngine:
    def __init__(self, t_step, pattern_shape=(2, 2)):
        self._t_step = t_step
        self._data = None
        self._freq_to_index = dict()
        self._pattern_shape = pattern_shape

    def freq_map(self):
        period = PeriodGenerator
        num_pixels = self._pattern_shape[0] * self._pattern_shape[1]
        p_num = period.prime_numbers()[:num_pixels]
        for i in range(num_pixels):
            freq = 1 / (self._t_step * p_num[i] * 2)

            # x = index % width, y = index // width
            y, x = divmod(i, self._pattern_shape[1])
            self._freq_to_index[freq] = (x, y)

    def get_freq_map(self):
        return self._freq_to_index

    def load_data(self, data_path='./raw_data/test.npy'):
        self._data = np.load(data_path)

    def get_data_shape(self):
        return self._data.shape

    def do_fft(self):
        avg = np.mean(self._data, axis=(1, 2))
        ft_abs = np.abs(np.fft.fft(avg))
        n = avg.size
        freq = np.fft.fftfreq(n, d=self._t_step)
        return ft_abs, freq

    def construct_image(self):
        pass


if __name__ == "__main__":
    simEng = SimulationEngine(0.1)
    simEng.load_data()
    frame, width, height = simEng.get_data_shape()
    print("({:d}, {:d}, {:d})".format(frame, width, height))

    ft, freq = simEng.do_fft()
    plt.plot(freq[1:math.ceil(frame / 2)], ft[1:math.ceil(frame / 2)])
    plt.show()

    simEng.freq_map()
    f_map = simEng.get_freq_map()
    print(f_map)


