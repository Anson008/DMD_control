import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math


class PatternSequence:
    """
    Generate gray scale pattern sequence.
    """

    def __init__(self, frames, width, height, scale):
        """
        :param frames: int. The number of frames in the pattern sequence
        :param width: int. Pattern width
        :param height: int. Pattern height
        :param scale: int. The factor by which a single pixel is enlarged. For example, enlarge a native 4*4 pattern by
                    scale 10 will produce a pattern of actual size 40*40, i.e, each pixel is enlarged by 10 times but
                    the native structure of the pattern is still represented by a 4*4 grid.
        """
        self._frames = frames
        self._width = width
        self._height = height
        self._scale = scale

    def get_shape(self):
        """
        Return the shape of pattern.

        :return: tuple. The shape of pattern sequence in (frames, height, width).
        """
        return self._frames, self._height, self._width, self._scale

    @staticmethod
    def save_to_npy(arr, directory='./raw_data', filename='test'):
        """
        Save the numpy array of pattern sequence to .npy file.

        :param directory: str. The directory to save file.
        :param filename: str. The file name.
        :param arr: numpy array. Pattern sequence to save
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = directory + '/' + filename + '.npy'
        np.save(file_path, arr)


class PatternSequenceGenerator(PatternSequence):
    def generate_binary_patterns(self, periods, mode='uniform'):
        """
        Generate binary pattern sequence. Along the time axis, the pixel values forms a square wave (binary signal).

        If in 'nonuniform' mode, the period of the wave at each pixel is the ith prime number, where i is the index
        of the pixel. For example, a 2*2 pattern has 4 pixels in total. The period of the first pixel (1, 1) is the
        first prime number 2. The period of the last pixel (2, 2) is 7 which is the 4th prime number.

        If in 'uniform' mode, the period of the wave over all pixels is the same.

        :param periods: list of int. The periods of signal at position (x, y).
        :param mode: string. Specify on which mode the generator works. Options are 'uniform' and 'nonuniform'.
        :return: numpy array. Pattern sequence generated.
        """
        pattern = np.zeros((self._frames, self._height, self._width), dtype=np.uint8)
        if mode == 'nonuniform':
            for z in range(self._frames):
                for y in range(self._height):
                    for x in range(self._width):
                        i_period = x + self._width * y
                        if (z // periods[i_period]) % 2 == 0:
                            pattern[z][y][x] = 255
                        else:
                            pattern[z][y][x] = 0
            # Computes the Kronecker product, a composite array made of blocks of the second array scaled by the first.
            pattern = np.kron(pattern, np.ones((self._scale, self._scale), dtype=np.uint8))
        elif mode == 'uniform':
            for z in range(self._frames):
                if (z // periods[0]) % 2 == 0:
                    pattern[z, :, :] = 255
                else:
                    pattern[z, :, :] = 0
            pattern = np.kron(pattern, np.ones((self._scale, self._scale), dtype=np.uint8))
        else:
            raise ValueError('Mode must be "nonuniform" or "uniform"')
        return pattern

    def generate_sinusoidal_patterns(self, base_freq=0.1, freq_step=0.1, mode='uniform'):
        """
        Generate gray-scale pattern sequence. Along the time axis, the pixel values forms a sinusoidal wave.
        The frequency of the wave at each pixel is the multiple of a base frequency and the index of the pixel.
        For example, a 2*2 pattern has 4 pixels in total. The frequency of the first pixel (1, 1) is 1*base_freq.
        The frequency of the last pixel (2, 2) is 4*base_freq.

        :param base_freq: float. Base frequency of the signal, i.e., the frequency of the pixel at upper-left corner.
        :param freq_step: float. The increment by which frequency is increased.
        :param mode: string. Specify on which mode the generator works. Options are 'uniform' and 'nonuniform'.
        :return: numpy array. Pattern sequence generated.
        """
        pattern = np.zeros((self._frames, self._height, self._width), dtype=np.uint8)
        if mode == 'uniform':
            max_freq = self._height * self._width * base_freq
            sample_rate = 20 * max_freq
            for y in range(self._height):
                for x in range(self._width):
                    freq = base_freq + (x + self._width * y) * freq_step
                    pattern[:, y, x] = 127.5 * np.sin(2 * np.pi * freq *
                                                      np.linspace(0, (self._frames / sample_rate),
                                                                  num=self._frames, endpoint=True)) + 127.5
            return np.kron(pattern, np.ones((self._scale, self._scale), dtype=np.uint8))
        elif mode == 'nonuniform':
            sample_rate = 20 * base_freq
            for y in range(self._height):
                for x in range(self._width):
                    pattern[:, y, x] = 127.5 * np.sin(2 * np.pi * base_freq *
                                                      np.linspace(0, (self._frames / sample_rate),
                                                                  num=self._frames, endpoint=True)) + 127.5
            return np.kron(pattern, np.ones((self._scale, self._scale), dtype=np.uint8))

    @staticmethod
    def gray2binary(pattern):
        """
        Convert elements along the first axis of a uint8 pattern array into a binary-valued output array.
        """
        return np.unpackbits(pattern, axis=0)

    @staticmethod
    def binary2gray(pattern):
        """
        Convert elements along the first axis of a binary-valued pattern array into a uint8 output array.
        """
        return np.packbits(pattern, axis=0)

    @staticmethod
    def look_along_time_axis(pattern, xy_range):
        """
        Print out a selected x-y region of pattern sequence along time axis.
        :param pattern: numpy array
        :param xy_range: list of tuple. Specify the (x, y) range of pattern we want to print out along z axis
                        (time axis). [(x1, x2), (y1, y2)]
        """
        print("-"*10, "Preview of signal along z axis", "-"*10)
        for y in range(xy_range[1][0], xy_range[1][1]):
            for x in range(xy_range[0][0], xy_range[0][1]):
                print(pattern[:, y, x])
        print("-"*10, "End of printing", "-"*10, "\n")

    @staticmethod
    def check_freq(pattern, xy=(0, 0), time_step=0.1):
        """
        Check the frequency of signal along z axis at pixel (x, y).

        :param pattern: numpy array
        :param xy: tuple of int. The x-y position of the pixel being checked.
        :param time_step: float. The time interval between two successive patterns.
        """
        spectrum = np.fft.fft(pattern[:, xy[1], xy[0]])
        n = pattern.shape[0]
        freq = np.fft.fftfreq(n, d=time_step)
        plt.plot(freq, spectrum.real)
        plt.show()

    @staticmethod
    def get_time_series(pattern, x, y):
        """
        Get the time series along z axis at position (y, x).

        :param pattern: numpy array
        :param x: int. Index of pixel along x axis (width).
        :param y: int. Index of pixel along y axis (height).
        """
        return pattern[:, y, x]

    def preview(self, pattern):
        """
        Preview every pattern in the pattern sequence. Enter 'q' to exit the preview.
        """
        for z in range(self._frames):
            cv2.imshow('Preview of patterns', pattern[z, :, :])
            if cv2.waitKey(0) == ord('q'):
                break
        cv2.destroyAllWindows()

    def pad(self, pattern, width=1920, height=1080):
        """
        Pad the edge of patterns so that the size of image is width * height.

        :param pattern: numpy array
        :param width: int, width of the pattern after padded.
        :param height: int, height of the pattern after padded.
        """
        img_w = self._width * self._scale
        img_h = self._height * self._scale
        if width <= img_w:
            bf_w = 0
            af_w = 0
        else:
            bf_w = int(math.floor((width - img_w) / 2))
            af_w = int(width - img_w - bf_w)
        if height <= img_h:
            bf_h = 0
            af_h = 0
        else:
            bf_h = int(math.floor((height - img_h) / 2))
            af_h = int(height - img_h - bf_h)
        if pattern.ndim == 3:
            return np.pad(pattern, ((0, 0), (bf_h, af_h), (bf_w, af_w)), 'constant')
        if pattern.ndim == 2:
            return np.pad(pattern, ((bf_h, af_h), (bf_w, af_w)), 'constant')

    def undo_pad(self, pattern):
        """
        Undo pad.
        """
        if pattern.shape[1] == 1080 and pattern.shape[2] == 1920:
            w = 1920
            h = 1080
            bf_w = int(math.floor((w - self._width * self._scale) / 2))
            bf_h = int(math.floor((h - self._height * self._scale) / 2))
            af_w = int(w - self._width * self._scale - bf_w)
            af_h = int(h - self._height * self._scale - bf_h)
            return pattern[:, bf_h:-af_h, bf_w:-af_w]
        else:
            print("Dimension of input array is not 1920 by 1080.")

    def make_calibration_pattern(self):
        """
        Generate a pattern of "+" in case a known pattern is needed to test the program or do simulations.
        """
        w, h = self._width * self._scale, self._height * self._scale
        img = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                c1i = (int(0.2*h) <= i <= int(0.4*h)) or (int(0.6*h) <= i <= int(0.8*h))
                c1j = (int(0.4*w) <= j <= int(0.6*w))
                img[i, j] = 255 if c1i or c1j else 0
        return img

    @staticmethod
    def save_single_pattern(img, directory='./calibration', filename='calib_01.png'):
        """
        Save a single pattern to a user specified directory.

        :param img: numpy array. The pattern to be saved as a picture.
        :param directory: str. Destination directory.
        :param filename: str. File name with extension.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, filename)
        cv2.imwrite(file_path, img, [cv2.IMWRITE_PNG_BILEVEL, 1])

    @staticmethod
    def save_to_images(pattern, directory='./img_sequence', prefix='test', fmt='.png', bit_level=8):
        """
        Save pattern sequence as images to a user specified directory, with auto naming from "0" to "(No. of patterns) - 1".

        :param pattern: numpy array
        :param directory: str. Destination directory.
        :param prefix: str. The prefix of file names. Index will be appended automatically when generating full names.
        :param fmt: str. Format of the pattern files. Default is '.png'
        :param bit_level: int. Specify the bit level of images to save. Default is 8.
                        Choose from 8 (grayscale) or 1 (binary).
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        total = pattern.shape[0]
        length = len(str(total))
        if bit_level == 1:
            for z in range(total):
                filename = prefix + '_' + "{:d}".format(z).zfill(length) + fmt
                file_path = os.path.join(directory, filename)
                cv2.imwrite(file_path, pattern[z, :, :], [cv2.IMWRITE_PNG_BILEVEL, 1])
        elif bit_level == 8:
            for z in range(total):
                filename = prefix + '_' + "{:d}".format(z).zfill(length) + fmt
                file_path = os.path.join(directory, filename)
                cv2.imwrite(file_path, pattern[z, :, :])

    @staticmethod
    def save_to_video(pattern, fps, directory='./videos', filename='video1.avi',
                      pattern_type="grayscale", threshold=127):
        """
        Save pattern sequence as a video to user specified directory.

        :param pattern: numpy array
        :param fps: int. Frame rate of the video.
        :param directory: str. Destination directory.
        :param filename: str. File name of the video. Default extension is '.avi'
        :param pattern_type: str. The type of patterns with two options "binary" or "grayscale". Default is "grayscale".
        :param threshold: int. Set the threshold to convert pixel values in 0~255 to binary values 0 and 1. For example,
                        if threshold = 127, any value greater than 127 would be set to 1, and values less than or equal
                        to 127 would be set to 0.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        fs, w, h = pattern.shape
        file_path = os.path.join(directory, filename)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(file_path, fourcc, fps, (w, h), False)
        if pattern_type == "binary":
            for z in range(fs):
                # Convert gray scale to binary.
                thresh, frame_b = cv2.threshold(pattern[z, :, :], threshold, 255, cv2.THRESH_BINARY)
                video.write(frame_b)
            video.release()
        elif pattern_type == "grayscale":
            for z in range(fs):
                video.write(pattern[z, :, :])
            video.release()
        else:
            print("Not a valid type! Enter exactly 'binary' or 'grayscale'.")


if __name__ == "__main__":
    pass
