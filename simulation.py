import numpy as np
import matplotlib.pyplot as plt
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
        """
        Map each frequency with its position in the pattern.
        """
        period = PeriodGenerator
        num_pixels = self._pattern_shape[0] * self._pattern_shape[1]
        p_num = period.prime_numbers()[:num_pixels]
        for i in range(num_pixels):
            # freq = np.around(1 / (self._t_step * p_num[i] * 2), 4)
            freq = 1 / (self._t_step * p_num[i] * 2)

            # x = index % width (column index), y = index // width (row index)
            y, x = divmod(i, self._pattern_shape[1])
            self._freq_to_index[freq] = (y, x)

    def get_freq_map(self):
        return self._freq_to_index

    def load_data(self, data_path='./raw_data/test.npy'):
        """
        :data_path: staring, path of image sequence data to load.
        """
        self._data = np.load(data_path)

    def get_data_shape(self):
        return self._data.shape

    def get_data(self):
        return self._data

    @staticmethod
    def cut_off(data, threshold=127):
        data[data <= threshold] = 0
        return data

    def fft(self):
        avg = np.mean(self._data, axis=(1, 2))
        ft_abs = np.abs(np.fft.fft(avg))
        n = avg.size
        freq = np.fft.fftfreq(n, d=self._t_step)
        return ft_abs[1:math.ceil(n / 2)], freq[1:math.ceil(n / 2)]

    @staticmethod
    def wrap_ft_results(freq, ft):
        return dict(zip(freq, ft))

    def construct_image(self, freq_index, freq_to_ampl, height, width, scale=5, epsilson=0.001):
        """
        :freq_index: dictionary, frequency-to-index map obtained from 'freq_map' method.
        :freq_to_ampl: dictionary, frequency-to-amplitude map obtained from 'wrap_ft_results' method.
        :height: int, native height of the image
        :width: int, native width of the image
        :scale: int, scale factor to enlarge the image.
        :return: numpy array, scaled image data.
        """
        res = np.zeros((height * scale, width * scale))
        for freq1 in freq_index:
            for freq2, ampl in freq_to_ampl.items():
                if abs(freq1 - freq2) <= epsilson:
                    for i in range(scale):
                        for j in range(scale):
                            res[freq_index[freq1][0] * scale + i, freq_index[freq1][1] * scale + j] = ampl
                    break
        return res


if __name__ == "__main__":
    t_step = 1 / 60
    simEng = SimulationEngine(t_step, pattern_shape=(8, 8))
    simEng.load_data(data_path='./raw_data/image_T_8X8_60fps.npy')
    frame, width, height = simEng.get_data_shape()
    print("({:d}, {:d}, {:d})".format(frame, width, height))

    ft, freq = simEng.do_fft()
    freq_shape = freq.shape
    print("freq points:", freq_shape)
    #plt.plot(freq[:300], ft[:300])
    #plt.show()

    simEng.freq_map()
    f_map = simEng.get_freq_map()
    #print(f_map)


    freq_to_ampl = simEng.wrap_ft_results(freq, ft)
    #print(freq_to_ampl[i_test])
    img_constructed = simEng.construct_image(f_map, freq_to_ampl, 8, 8, 10, epsilson=2*(1 / frame))
    #print(img_constructed[0, 1])
    print(img_constructed.shape)
    plt.imshow(img_constructed, origin='lower')
    plt.show()

