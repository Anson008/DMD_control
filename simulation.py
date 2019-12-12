import numpy as np
import matplotlib.pyplot as plt
from binary_images import BinaryImage
from periods import PeriodGenerator
import math


class SimulationEngine:
    def __init__(self, t_step, pattern_shape=(2, 2)):
        """
        :_t_step: float, time interval between two successive image in real experiment.
        :_data: numpy array, raw image sequence.
        :_freq_to_index: dictionary, a map between frequency ba{freq: (x, y)}.
        :_pattern_shape: native shape of the image
        """
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

            # x = index % width (column index), y = index // width (row index)
            y, x = divmod(i, self._pattern_shape[1])
            self._freq_to_index[freq] = (y, x)

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
        return ft_abs[1:math.ceil(frame / 2)], freq[1:math.ceil(frame / 2)]

    @staticmethod
    def wrap_ft_results(freq, ft):
        return dict(zip(freq, ft))

    @staticmethod
    def construct_image(freq_index, freq_to_ampl, height, width):
        res = np.zeros((height, width))
        for freq1, ampl in freq_to_ampl.items():
            if freq1 in freq_index:
                res[freq_index[freq1][0], freq_index[freq1][1]] = ampl
                print("freq: {:f}, ampl: {:f}, "
                      "index: ({:d}, {:d})".format(freq1, ampl, freq_index[freq1][0], freq_index[freq1][1]))
        return res


if __name__ == "__main__":
    simEng = SimulationEngine(0.1)
    simEng.load_data()
    frame, width, height = simEng.get_data_shape()
    print("({:d}, {:d}, {:d})".format(frame, width, height))

    ft, freq = simEng.do_fft()
    #plt.plot(freq, ft)
    #plt.show()

    i_test = 5 / 3
    simEng.freq_map()
    f_map = simEng.get_freq_map()
    print(f_map)
    print(i_test in freq)
    print(i_test in f_map)

    freq_to_ampl = simEng.wrap_ft_results(freq, ft)
    #print(freq_to_ampl[i_test])
    img_constructed = simEng.construct_image(f_map, freq_to_ampl, 2, 2)
    #print(img_constructed[0, 1])



