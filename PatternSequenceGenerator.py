import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt
import math
from PeriodGenerator import PeriodGenerator


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
        pattern = np.zeros((self._frames, self._height * self._scale, self._width * self._scale), dtype=np.uint8)
        if mode == 'nonuniform':
            for z in range(self._frames):
                for y in range(self._height * self._scale):
                    for x in range(self._width * self._scale):
                        # first define the mapping between the index of period value and position (y, x)
                        i_period = (x // self._scale) + self._width * (y // self._scale)
                        if (z // periods[i_period]) % 2 == 0:
                            pattern[z][y][x] = 255
                        else:
                            pattern[z][y][x] = 0
        elif mode == 'uniform':
            for z in range(self._frames):
                if (z // periods[0]) % 2 == 0:
                    pattern[z, :, :] = 255
                else:
                    pattern[z, :, :] = 0
        else:
            raise ValueError('Mode must be "nonuniform" or "uniform"')
        return pattern

    def generate_sinusoidal_patterns(self, base_freq=0.1, freq_step=0.1):
        """
        Generate gray-scale pattern sequence. Along the time axis, the pixel values forms a sinusoidal wave.
        The frequency of the wave at each pixel is the multiple of a base frequency and the index of the pixel.
        For example, a 2*2 pattern has 4 pixels in total. The frequency of the first pixel (1, 1) is 1*base_freq.
        The frequency of the last pixel (2, 2) is 4*base_freq.

        :param base_freq: float. Base frequency of the signal, i.e., the frequency of the pixel at upper-left corner.
        :param freq_step: float. The increment by which frequency is increased.
        """
        max_freq = self._height * self._width * base_freq
        sample_rate = 20 * max_freq
        pattern = np.zeros((self._frames, self._height * self._scale, self._width * self._scale), dtype=np.uint8)
        for y in range(self._height * self._scale):
            for x in range(self._width * self._scale):
                freq = base_freq + ((x // self._scale) + self._width * (y // self._scale)) * freq_step
                pattern[:, y, x] = 128 * np.sin(2 * np.pi * freq *
                                                np.linspace(0, (self._frames / sample_rate),
                                                            num=self._frames, endpoint=True)) + 127
        return pattern

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

    def pad(self, pattern):
        """
        Pad the edge of patterns so that the image is 1920 * 1080.
        """
        W = 1920
        H = 1080
        bf_w = int(math.floor((W - self._width * self._scale) / 2))
        bf_h = int(math.floor((H - self._height * self._scale) / 2))
        af_w = int(W - self._width * self._scale - bf_w)
        af_h = int(H - self._height * self._scale - bf_h)
        if pattern.ndim == 3:
            return np.pad(pattern, ((0, 0), (bf_h, af_h), (bf_w, af_w)), 'constant')
        if pattern.ndim == 2:
            return np.pad(pattern, ((bf_h, af_h), (bf_w, af_w)), 'constant')

    def undo_pad(self, pattern):
        """
        Undo pad.
        """
        if pattern.shape[1] == 1080 and pattern.shape[2] == 1920:
            W = 1920
            H = 1080
            bf_w = int(math.floor((W - self._width * self._scale) / 2))
            bf_h = int(math.floor((H - self._height * self._scale) / 2))
            af_w = int(W - self._width * self._scale - bf_w)
            af_h = int(H - self._height * self._scale - bf_h)
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
                c1i = (int(0.4*h) <= i <= int(0.6*h))
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
    # Load the leading 100 prime numbers
    periods = PeriodGenerator()
    prime_num = periods.prime_numbers()

    # Set pattern pad, create an instant of PatternSequenceGenerator class.
    patt = PatternSequenceGenerator(200, 2, 2, 100)

    # Get pattern shape.
    frames, height, width, scale = patt.get_shape()

    # Generate binary pattern sequence.
    start = time.time()
    pattern_arr = patt.generate_binary_patterns(periods=prime_num, mode='nonuniform')
    print("\nIt took {:.2f} s to generate binary pattern sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual patterns should be ({:d}, {:d}, {:d}), without taking account into pad."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))

    # Generate sinusoidal pattern sequence.
    """start = time.time()
    pattern_arr = patt.generate_sinusoidal_patterns(base_freq=0.1)
    print("\nIt took {:.2f} s to generate pattern sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual patterns should be ({:d}, {:d}, {:d}), without taking account into pad."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))"""
    
    # Get time series along z axis at position (y, x)
    """ts00 = patt.get_time_series(arr=patterns, x=0, y=0)
    plt.plot(ts00)
    plt.show()"""
    
    # Check frequency of time series at position (x, y)
    """for i in range(height):
        for j in range(width):
            patt.check_freq(xy=(j, i), time_step=0.1)"""

    # Convert uint8 elements to binary-valued elements
    #patt.gray2binary()
    #print(f"Shape after converting to binary:", patt.get_shape())

    # Pad patterns.
    patt.pad(pattern_arr)

    # Undo pad to patterns.
    # patt.undo_pad()

    # Preview pattern sequence frame by frame.
    # patt.preview()

    # Get pixel values.
    # pattern = patt.get_pattern()
    # print("\nShape of generated patterns:", pattern.shape)

    # Save pixel values to .npy file.
    # patt.save_to_npy(filename='test_Sin_200kFrames_8X8')

    # Make calibration patterns.
    # calib_img = patt.make_calibration_pattern()
    # patt.save_single_pattern(calib_img, filename='calib_8by8')

    file_name = 'b_20Frames_2X2_scale100_pad1'
    # file_name = 'sinB_20Frames_2X2_scale100_baseFreq1E-1_pad1'
    # Save pattern sequence to images
    patt.save_to_images(pattern_arr, directory="./" + file_name, prefix=file_name, bit_level=1)

    # Save pattern sequence to video
    # patt.save_to_video(fps=20, filename=file_name + '.avi')






