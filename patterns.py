import numpy as np
import time
import cv2
import os
from periods import PeriodGenerator


class PatternSequence:
    """
    Generate gray scale pattern sequence.
    """

    def __init__(self, frames, width, height, scale, padding, periods):
        """
        :arg frames: int, number of frames in the pattern sequence
        :arg width: int, pattern width
        :arg height: int, pattern height
        :arg periods: list of int, the periods of signal at position (x, y)
        :arg padding: tuple, ((before1, after1), (before2, after2)) specifies the number of values padded to the edges
                    of each axis. It is the width of a frame added to the pattern.
        :arg scale: int, the factor by which a single pixel is enlarged. For example, enlarge a native 4*4 pattern by
                    scale 10 will produce a pattern of actual size 40*40, i.e, each pixel is enlarged by 10 times but
                    the native structure of the pattern is still represented by a 4*4 grid.

        Attributes
        :_pattern: numpy array, store the pixel values of pattern.
        """
        self._frames = frames
        self._width = width
        self._height = height
        self._periods = periods
        self._padding = padding
        self._scale = scale
        self._pattern = None

    def get_shape(self):
        """
        Return the shape of pattern.

        :return: tuple, the shape of pattern sequence in (frames, height, width).
        """
        return self._frames, self._height, self._width, self._scale

    def get_pattern(self):
        """
        Return the pattern sequence.

        :return: numpy array, the pattern sequence
        """
        return self._pattern

    def save_to_npy(self, directory='./raw_data', filename='test'):
        """
        Save the numpy array of pattern sequence to .npy file.

        :arg directory: str, the directory to save file.
        :arg filename: str, the file name.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = directory + '/' + filename + '.npy'
        np.save(file_path, self._pattern)


class PatternSequenceGenerator(PatternSequence):
    def generate_prime_patterns(self):
        """
        Generate binary pattern sequence. Along the time axis, the pixel values forms a square wave (binary signal).
        The period of the wave at each pixel is the ith prime number, where i is the index of the pixel. For example,
        a 2*2 pattern has 4 pixels in total. The period of the first pixel (1, 1) is the first prime number 2. The period
        of the last pixel (2, 2) is 7 which is the 4th prime number.
        """
        self._pattern = np.zeros((self._frames, self._height * self._scale, self._width * self._scale), dtype=np.uint8)
        for z in range(self._frames):
            for y in range(self._height * self._scale):
                for x in range(self._width * self._scale):
                    # first define the mapping between the index of period value and position (y, x)
                    i_period = (x // self._scale) + self._width * (y // self._scale)
                    if (z // self._periods[i_period]) % 2 == 0:
                        self._pattern[z][y][x] = 255
                    else:
                        self._pattern[z][y][x] = 0

    def generate_sinusoidal_patterns(self, time=2000, base_freq=0.1):
        """
        Generate gray-scale pattern sequence. Along the time axis, the pixel values forms a sinusoidal wave.
        The frequency of the wave at each pixel is the multiple of a base frequency and the index of the pixel.
        For example, a 2*2 pattern has 4 pixels in total. The frequency of the first pixel (1, 1) is 1*base_freq.
        The frequency of the last pixel (2, 2) is 4*base_freq.

        :arg time: int, total length of the signal (pseudo time).
        :arg base_freq: float, base frequency of the signal, i.e., the frequency of the pixel at upper-left corner.
        """
        self._pattern = np.zeros((self._frames, self._height * self._scale, self._width * self._scale), dtype=np.uint8)
        for y in range(self._height * self._scale):
            for x in range(self._width * self._scale):
                freq = ((x // self._scale) + self._width * (y // self._scale) + 1) * base_freq
                self._pattern[:, y, x] = 128 * np.sin(2 * np.pi * freq *
                                                      np.linspace(0, time, num=self._frames, endpoint=False)) + 127

    def look_along_time_axis(self, xy_range):
        """
        Print out a selected x-y region of pattern sequence along time axis.

        :xy_range: list of tuple, specify the (x, y) range of pattern we want to print out along z axis (time axis).
                   [(x1, x2), (y1, y2)]
        """
        print("-"*10, "Preview of signal along z axis", "-"*10)
        for y in range(xy_range[1][0], xy_range[1][1]):
            for x in range(xy_range[0][0], xy_range[0][1]):
                print(self._pattern[:, y, x])
        print("-"*10, "End of printing", "-"*10, "\n")

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
        Pad the edge of patterns.
        """
        self._pattern = np.pad(self._pattern,
                               ((0, 0), (self._padding[0][0], self._padding[0][1]),
                                (self._padding[1][0], self._padding[1][1])), 'constant')

    def undo_pad(self):
        """
        Undo pad.
        """
        self._pattern = self._pattern[:, self._padding[0][0]:-self._padding[0][1], self._padding[1][0]:-self._padding[1][1]]

    def make_calibration_pattern(self):
        w, h = self._width * self._scale, self._height * self._scale
        img = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                c1i = (int(0.1*h) <= i <= int(0.3*h) or
                       int(0.4*h) <= i <= int(0.6*h) or
                       int(0.7*h) <= i <= int(0.9*h))
                c1j = (int(0.4*w) <= j <= int(0.6*w))
                img[i, j] = 255 if c1i or c1j else 0
        img = np.pad(img, ((self._padding[0][0], self._padding[0][1]), (self._padding[1][0], self._padding[1][1])),
                     'constant', constant_values=((0, 0),))
        return img

    @staticmethod
    def save_pattern(img, directory='./calibration', filename='calib_01', fmt='.png'):
        """
        Save a single pattern to a user specified directory.
        :para directory: string, destination directory.
        :para filename: string, file name without extension.
        :para fmt: string, format of the pattern files. Default is '.png'.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = directory + "/" + filename + fmt
        cv2.imwrite(file_path, img)

    def make_pattern_sequence(self, directory='./img_sequence', filename='test', fmt='.png'):
        """
        Save pattern sequence to a user specified directory, with auto naming from "0" to "# of patterns - 1".
        :para directory: string, destination directory.
        :para fmt: string, format of the pattern files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        for z in range(self._frames):
            file_path = directory + "/" + filename + '_' + "{:d}".format(z).zfill(len(str(z))) + fmt
            cv2.imwrite(file_path, self._pattern[z, :, :])

    def make_binary_video(self, fps, directory='./videos', filename='video1', fmt='.avi', threshold=127):
        """
        Save pattern sequence as a video to user specified directory.
        :para fps: int, frame rate of the video.
        :para directory: string, destination directory.
        :para filename: string, file name of the video, without extension.
        :para fmt: string, format of the pattern files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        fs, w, h = self._pattern.shape
        file_path = directory + '/' + filename + fmt
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(file_path, fourcc, fps, (w, h), False)
        for z in range(fs):
            # Convert gray scale to binary.
            thresh, frame_b = cv2.threshold(self._pattern[z, :, :], threshold, 255, cv2.THRESH_BINARY)
            video.write(frame_b)
        video.release()

    def make_grayscale_video(self, fps, directory='./videos', filename='video1', fmt='.avi'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        fs, w, h = self._pattern.shape
        file_path = directory + '/' + filename + fmt
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(file_path, fourcc, fps, (w, h), False)
        for z in range(fs):
            video.write(self._pattern[z, :, :])
        video.release()


if __name__ == "__main__":
    # Load the leading 100 prime numbers
    periods = PeriodGenerator()
    prime_num = periods.prime_numbers()

    # Set pattern padding, create an object of Binarypattern class.
    padding = ((20, 20), (20, 20))
    binary_img = Binarypattern(200000, 8, 8, 5, padding, prime_num)

    # Get pattern shape.
    frames, height, width, scale = binary_img.get_img_shape()

    # Generate pattern sequence without padding.
    # start = time.time()
    # binary_img.generate_patterns()
    """print("\nIt took {:.2f} s to generate pattern sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual patterns should be ({:d}, {:d}, {:d}), without taking account into padding."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))"""

    # Generate sinusoidal pattern sequence
    start = time.time()
    binary_img.generate_sinusoidal_patterns(time=2000)
    print("\nIt took {:.2f} s to generate pattern sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual patterns should be ({:d}, {:d}, {:d}), without taking account into padding."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))

    # Preview pattern pixel values along time axis.
    # binary_img.print_through_time([(0, 2), (0, 2)])

    # Add padding to patterns.
    binary_img.add_padding()

    # Undo padding to patterns if necessary.
    # binary_img.undo_padding(pad_width=padding)

    # Preview pattern sequence frame by frame if necessary.
    # binary_img.preview()

    # Get pixel values if necessary.
    # pattern = binary_img.get_pattern()
    # print("\nShape of generated patterns:", pattern.shape)

    # Save pixel values to .npy file if necessary.
    binary_img.save_to_npy(filename='testSin_200kFrames_8X8')

    # Make calibration patterns.
    # calib_img = binary_img.make_calibration_pattern()
    # binary_img.save_pattern(calib_img, filename='calib_8by8')

    # binary_img.make_pattern_sequence()
    # binary_img.make_binary_video(fps=20, filename='d')

    # Make gray scale video
    # binary_img.make_grayscale_video(fps=20, filename='sinusoidal_200kFrames_20fps_8X8_natNumFreq')




