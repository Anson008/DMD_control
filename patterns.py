import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt
import math
from periods import PeriodGenerator


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

        Attributes
        :_pattern: numpy array. Store the pixel values of pattern.
        """
        self._frames = frames
        self._width = width
        self._height = height
        self._scale = scale
        self._pattern = None

    def get_shape(self):
        """
        Return the shape of pattern.

        :return: tuple. The shape of pattern sequence in (frames, height, width).
        """
        return self._frames, self._height, self._width, self._scale

    def get_pattern(self):
        """
        Return the pattern sequence.

        :return: numpy array. The pattern sequence.
        """
        return self._pattern

    def save_to_npy(self, directory='./raw_data', filename='test'):
        """
        Save the numpy array of pattern sequence to .npy file.

        :param directory: str. The directory to save file.
        :param filename: str. The file name.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = directory + '/' + filename + '.npy'
        np.save(file_path, self._pattern)


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
        """
        self._pattern = np.zeros((self._frames, self._height * self._scale, self._width * self._scale), dtype=np.uint8)
        if mode == 'nonuniform':
            for z in range(self._frames):
                for y in range(self._height * self._scale):
                    for x in range(self._width * self._scale):
                        # first define the mapping between the index of period value and position (y, x)
                        i_period = (x // self._scale) + self._width * (y // self._scale)
                        if (z // periods[i_period]) % 2 == 0:
                            self._pattern[z][y][x] = 255
                        else:
                            self._pattern[z][y][x] = 0
        elif mode == 'uniform':
            for z in range(self._frames):
                if (z // periods[0]) % 2 == 0:
                    self._pattern[z, :, :] = 255
                else:
                    self._pattern[z, :, :] = 0
        else:
            raise ValueError('Mode must be "nonuniform" or "uniform"')

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
        self._pattern = np.zeros((self._frames, self._height * self._scale, self._width * self._scale), dtype=np.uint8)
        for y in range(self._height * self._scale):
            for x in range(self._width * self._scale):
                freq = base_freq + ((x // self._scale) + self._width * (y // self._scale)) * freq_step
                self._pattern[:, y, x] = 128 * np.sin(2 * np.pi * freq *
                                                      np.linspace(0, (self._frames / sample_rate),
                                                                  num=self._frames, endpoint=True)) + 127

    def gray2binary(self):
        """
        Convert elements along the first axis of a uint8 pattern array into a binary-valued output array.
        """
        self._pattern = np.unpackbits(self._pattern, axis=0)

    def binary2gray(self):
        """
        Convert elements along the first axis of a binary-valued pattern array into a uint8 output array.
        :param arr: numpy array. The array to be converted.
        """
        self._pattern = np.packbits(self._pattern, axis=0)

    def look_along_time_axis(self, xy_range):
        """
        Print out a selected x-y region of pattern sequence along time axis.

        :param xy_range: list of tuple. Specify the (x, y) range of pattern we want to print out along z axis
                        (time axis). [(x1, x2), (y1, y2)]
        """
        print("-"*10, "Preview of signal along z axis", "-"*10)
        for y in range(xy_range[1][0], xy_range[1][1]):
            for x in range(xy_range[0][0], xy_range[0][1]):
                print(self._pattern[:, y, x])
        print("-"*10, "End of printing", "-"*10, "\n")

    def check_freq(self, xy=(0, 0), time_step=0.1):
        """
        Check the frequency of signal along z axis at pixel (x, y).

        :param xy: tuple of int. The x-y position of the pixel being checked.
        :param time_step: float. The time interval between two successive patterns.
        """
        spectrum = np.fft.fft(self._pattern[:, xy[1], xy[0]])
        n = self._pattern.shape[0]
        freq = np.fft.fftfreq(n, d=time_step)
        plt.plot(freq, spectrum.real)
        plt.show()

    def get_time_series(self, x, y):
        """
        Get the time series along z axis at position (y, x).

        :param x: int. Index of pixel along x axis (width).
        :param y: int. Index of pixel along y axis (height).
        """
        return self._pattern[:, y, x]

    def preview(self):
        """
        Preview every pattern in the pattern sequence. Enter 'q' to exit the preview.
        """
        for z in range(self._frames):
            cv2.imshow('Preview of patterns', self._pattern[z, :, :])
            if cv2.waitKey(0) == ord('q'):
                break
        cv2.destroyAllWindows()

    def pad(self):
        """
        Pad the edge of patterns so that the image is 1920 * 1080.
        """
        W = 1920
        H = 1080
        bf_w = int(math.floor((W - self._width * self._scale) / 2))
        bf_h = int(math.floor((H - self._height * self._scale) / 2))
        af_w = int(W - self._width * self._scale - bf_w)
        af_h = int(H - self._height * self._scale - bf_h)
        self._pattern = np.pad(self._pattern, ((0, 0), (bf_h, af_h), (bf_w, af_w)), 'constant')

    def undo_pad(self):
        """
        Undo pad.
        :param arr: numpy array. Array of which the pad to be removed.
        """
        if self._pattern.shape[1] == 1080 and self._pattern.shape[2] == 1920:
            W = 1920
            H = 1080
            bf_w = int(math.floor((W - self._width * self._scale) / 2))
            bf_h = int(math.floor((H - self._height * self._scale) / 2))
            af_w = int(W - self._width * self._scale - bf_w)
            af_h = int(H - self._height * self._scale - bf_h)
            self._pattern = self._pattern[:, bf_h:-af_h, bf_w:-af_w]
        else:
            print("Dimension of input array is not 1920 by 1080.")

    def make_calibration_pattern(self, pad):
        """
        Generate a pattern of "丰" in case a known pattern is needed to test the program or do simulations.
        """
        w, h = self._width * self._scale, self._height * self._scale
        img = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                c1i = (int(0.1*h) <= i <= int(0.3*h) or
                       int(0.4*h) <= i <= int(0.6*h) or
                       int(0.7*h) <= i <= int(0.9*h))
                c1j = (int(0.4*w) <= j <= int(0.6*w))
                img[i, j] = 255 if c1i or c1j else 0
        img = np.pad(img, ((pad[0][0], pad[0][1]), (pad[1][0], pad[1][1])),
                     'constant', constant_values=((0, 0),))
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
        cv2.imwrite(file_path, img)

    def save_to_images(self, directory='./img_sequence', prefix='test', fmt='.png', bit_level=8):
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
        if bit_level == 1:
            for z in range(self._pattern.shape[0]):
                filename = prefix + '_' + "{:d}".format(z) + fmt
                file_path = os.path.join(directory, filename)
                cv2.imwrite(file_path, self._pattern[z, :, :], [cv2.IMWRITE_PNG_BILEVEL, 1])
        elif bit_level == 8:
            for z in range(self._pattern.shape[0]):
                filename = prefix + '_' + "{:d}".format(z) + fmt
                file_path = os.path.join(directory, filename)
                cv2.imwrite(file_path, self._pattern[z, :, :])

    def save_to_video(self, fps, directory='./videos', filename='video1.avi',
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
        fs, w, h = self._pattern.shape
        file_path = os.path.join(directory, filename)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(file_path, fourcc, fps, (w, h), False)
        if pattern_type == "binary":
            for z in range(fs):
                # Convert gray scale to binary.
                thresh, frame_b = cv2.threshold(self._pattern[z, :, :], threshold, 255, cv2.THRESH_BINARY)
                video.write(frame_b)
            video.release()
        elif pattern_type == "grayscale":
            for z in range(fs):
                video.write(self._pattern[z, :, :])
            video.release()
        else:
            print("Not a valid type! Enter exactly 'binary' or 'grayscale'.")


if __name__ == "__main__":
    # Load the leading 100 prime numbers
    # periods = PeriodGenerator()
    # prime_num = periods.prime_numbers()

    # Set pattern pad, create an instant of PatternSequenceGenerator class.
    patt = PatternSequenceGenerator(20, 2, 2, 100)

    # Get pattern shape.
    frames, height, width, scale = patt.get_shape()

    # Generate binary pattern sequence.
    """start = time.time()
    patt.generate_binary_patterns(periods=prime_num)
    print("\nIt took {:.2f} s to generate binary pattern sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual patterns should be ({:d}, {:d}, {:d}), without taking account into pad."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))"""

    # Generate sinusoidal pattern sequence.
    start = time.time()
    patt.generate_sinusoidal_patterns(base_freq=0.1)
    print("\nIt took {:.2f} s to generate pattern sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual patterns should be ({:d}, {:d}, {:d}), without taking account into pad."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))
    
    # Get time series along z axis at position (y, x)
    """ts00 = patt.get_time_series(arr=patterns, x=0, y=0)
    plt.plot(ts00)
    plt.show()"""
    
    # Check frequency of time series at position (x, y)
    """for i in range(height):
        for j in range(width):
            patt.check_freq(xy=(j, i), time_step=0.1)"""

    # Convert uint8 elements to binary-valued elements
    patt.gray2binary()
    print(f"Shape after converting to binary:", patt.get_shape())

    # Pad patterns.
    patt.pad()

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

    # file_name = 'b_20Frames_2X2_scale100_pad1'
    file_name = 'sinB_20Frames_2X2_scale100_baseFreq1E-1_pad1'
    # Save pattern sequence to images
    patt.save_to_images(directory="./" + file_name, prefix=file_name, bit_level=1)

    # Save pattern sequence to video
    # patt.save_to_video(fps=20, filename=file_name + '.avi')






