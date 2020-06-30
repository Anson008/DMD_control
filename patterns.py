import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt
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
    def generate_binary_patterns(self, periods):
        """
        Generate binary pattern sequence. Along the time axis, the pixel values forms a square wave (binary signal).
        The period of the wave at each pixel is the ith prime number, where i is the index of the pixel. For example,
        a 2*2 pattern has 4 pixels in total. The period of the first pixel (1, 1) is the first prime number 2. The period
        of the last pixel (2, 2) is 7 which is the 4th prime number.

        :param periods: list of int. The periods of signal at position (x, y)
        """
        self._pattern = np.zeros((self._frames, self._height * self._scale, self._width * self._scale), dtype=np.uint8)
        for z in range(self._frames):
            for y in range(self._height * self._scale):
                for x in range(self._width * self._scale):
                    # first define the mapping between the index of period value and position (y, x)
                    i_period = (x // self._scale) + self._width * (y // self._scale)
                    if (z // periods) % 2 == 0:
                        self._pattern[z][y][x] = 255
                    else:
                        self._pattern[z][y][x] = 0

    def generate_sinusoidal_patterns(self, time=100, base_freq=0.1):
        """
        Generate gray-scale pattern sequence. Along the time axis, the pixel values forms a sinusoidal wave.
        The frequency of the wave at each pixel is the multiple of a base frequency and the index of the pixel.
        For example, a 2*2 pattern has 4 pixels in total. The frequency of the first pixel (1, 1) is 1*base_freq.
        The frequency of the last pixel (2, 2) is 4*base_freq.

        :param time: int. Total length of the signal (pseudo time).
        :param base_freq: float. Base frequency of the signal, i.e., the frequency of the pixel at upper-left corner.
        """
        self._pattern = np.zeros((self._frames, self._height * self._scale, self._width * self._scale), dtype=np.uint8)
        for y in range(self._height * self._scale):
            for x in range(self._width * self._scale):
                freq = ((x // self._scale) + self._width * (y // self._scale) + 1) * base_freq
                self._pattern[:, y, x] = 128 * np.sin(2 * np.pi * freq *
                                                      np.linspace(0, time, num=self._frames, endpoint=True)) + 127

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

    def preview(self):
        """
        Preview every pattern in the pattern sequence. Enter 'q' to exit the preview.
        """
        for z in range(self._frames):
            cv2.imshow('Preview of patterns', self._pattern[z, :, :])
            if cv2.waitKey(0) == ord('q'):
                break
        cv2.destroyAllWindows()

    def pad(self, pad=((20, 20), (20, 20))):
        """
        Pad the edge of patterns.
        """
        self._pattern = np.pad(self._pattern,
                               ((0, 0), (pad[0][0], pad[0][1]),
                                (pad[1][0], pad[1][1])), 'constant')

    def undo_pad(self, pad):
        """
        Undo pad.
        """
        self._pattern = self._pattern[:, pad[0][0]:-pad[0][1], pad[1][0]:-pad[1][1]]

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

    def save_to_images(self, directory='./img_sequence', prefix='test', fmt='.png'):
        """
        Save pattern sequence as images to a user specified directory, with auto naming from "0" to "(No. of patterns) - 1".

        :param directory: str. Destination directory.
        :param prefix: str. The prefix of file names. Index will be appended automatically when generating full names.
        :param fmt: str. Format of the pattern files. Default is '.png'
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        for z in range(self._frames):
            filename = prefix + '_' + "{:d}".format(z).zfill(len(str(z))) + fmt
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
    fr = 10000  # Total number of frames
    patt = PatternSequenceGenerator(fr, 2, 2, 100)

    # Get pattern shape.
    frames, height, width, scale = patt.get_shape()

    # Generate sinusoidal pattern sequence.
    start = time.time()
    sample_rate = 1000
    patt.generate_sinusoidal_patterns(time=int(fr / sample_rate), base_freq=10)
    print("\nIt took {:.2f} s to generate pattern sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual patterns should be ({:d}, {:d}, {:d}), without taking account into pad."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))

    # Preview pattern pixel values along time axis.
    # patt.print_through_time([(0, 2), (0, 2)])

    # Check frequency of time series at position (x, y)
    # patt.check_freq(xy=(4, 4), time_step=length/fr)

    # Pad patterns.
    # pad1 = ((200, 200), (200, 200))
    # patt.pad(pad1)

    # Undo pad to patterns if necessary.
    # patt.undo_pad(pad_width=pad)

    # Preview pattern sequence frame by frame if necessary.
    # patt.preview()

    # Get pixel values if necessary.
    # pattern = patt.get_pattern()
    # print("\nShape of generated patterns:", pattern.shape)

    # Save pixel values to .npy file if necessary.
    # patt.save_to_npy(filename='test_Sin_200kFrames_8X8')

    # Make calibration patterns.
    # calib_img = patt.make_calibration_pattern()
    # patt.save_single_pattern(calib_img, filename='calib_8by8')

    file_name = 'sin_10kFrames_2X2_scale100_sr1000_bf10_pad0'
    # Save pattern sequence to images
    # patt.save_to_images(directory="./" + file_name, prefix=file_name)

    # Save pattern sequence to video
    patt.save_to_video(fps=20, filename=file_name + '.avi')




