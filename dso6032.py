# INSTRUCTIONS:
# Edit in the VISA address of the oscilloscope
# Edit in the file save locations ## IMPORTANT NOTE:  This script WILL overwrite previously saved files!
# Manually (or write more code) acquire data on the oscilloscope.  Ensure that it finished (Run/Stop button is red).


import pyvisa
import sys
import time
import numpy as np


class DSO6034A(object):
    def __init__(self, points, visa_address, filename, file_directory, visa_dir="C:\\Windows\\System32\\visa32.dll", timeout=10000):
        self.USER_REQUESTED_POINTS = int(points)
        self.VISA_ADDRESS = visa_address
        self.GLOBAL_TIMEOUT = timeout
        self.FILE_NAME = filename
        self.FILE_DIRECTORY = file_directory
        self.VISA_DIR = visa_dir
        self.number_channels_on = 0
        self.chs_on = []
        self.scope = None
        self.analog_vert_pres = np.zeros((3, 4))
        # analog_vert_pres = [[Y_INCrement_Ch1, Y_INCrement_Ch2, Y_INCrement_Ch3, Y_INCrement_Ch4],
        #                    [Y_ORIGin_Ch1, Y_ORIGin_Ch2, Y_ORIGin_Ch3, Y_ORIGin_Ch4],
        #                    [Y_REFerence_Ch1, Y_REFerence_Ch2, Y_REFerence_Ch3, Y_REFerence_Ch4]]

    def connect(self):
        """
        Define and open the scope by the VISA address. This uses PyVISA
        :return: scope object
        """
        rm = pyvisa.ResourceManager(self.VISA_DIR)
        try:
            self.scope = rm.open_resource(self.VISA_ADDRESS)
        except Exception:
            print("Unable to connect to oscilloscope at {:s}. Aborting script.\n".format(self.VISA_ADDRESS))
            sys.exit()
        self.scope.timeout = self.GLOBAL_TIMEOUT  # Set Global Timeout
        self.scope.clear()  # Clear the instrument bus

    def find_channels(self):
        """
        Determine Which channels are on AND have acquired data.
        Scope should have already acquired data and be in a stopped state (Run/Stop button is red).
        :return: tuple, (number_channels_on, [chs_on]). [chs_on] is a list of index of channels that are on.
        """
        idn = str(self.scope.query("*IDN?"))
        idn = idn.split(",")  # IDN parts are separated by commas, so parse on the commas
        model = idn[1]
        if list(model[1]) == "9":
            number_analog_chs = 2
        else:
            number_analog_chs = int(model[len(model) - 2])
        if number_analog_chs == 2:
            chs_list = [0, 0]
        else:
            chs_list = [0, 0, 0, 0]

        # After the CHS_LIST array is filled it could, for example look like: if chs 1,3 and 4 were on,
        # CHS_LIST = [1,0,1,1].

        # analog_vert_pres = np.zeros((3, 4))

        # analog_vert_pres = [[Y_INCrement_Ch1, Y_INCrement_Ch2, Y_INCrement_Ch3, Y_INCrement_Ch4],
        #                    [Y_ORIGin_Ch1, Y_ORIGin_Ch2, Y_ORIGin_Ch3, Y_ORIGin_Ch4],
        #                    [Y_REFerence_Ch1, Y_REFerence_Ch2, Y_REFerence_Ch3, Y_REFerence_Ch4]]

        ch_units = ["BLANK", "BLANK", "BLANK", "BLANK"]
        self.scope.write(":WAVeform:POINts:MODE MAX")
        ch_index = 1  # Channel index
        for ch in chs_list:
            on_off = int(self.scope.query(":CHANnel{:s}:DISPlay?".format(ch)))
            if on_off == 1:
                channel_acquired = int(self.scope.query(":WAVeform:SOURce CHANnel{:s};POINts?".format(ch)))
            else:
                channel_acquired = 0
            if on_off == 0 or channel_acquired == 0:
                self.scope.write(":CHANnel{:s}:DISPlay OFF".format(ch))
                chs_list[ch-1] = 0
            else:
                chs_list[ch-1] = 1
                self.number_channels_on += 1
                pre_amble = self.scope.query(":WAVeform:PREamble?").split(",")
                self.analog_vert_pres[0, ch-1] = float(pre_amble[7])
                self.analog_vert_pres[1, ch-1] = float(pre_amble[8])
                self.analog_vert_pres[2, ch-1] = float(pre_amble[9])
                ch_units[ch-1] = str(self.scope.query(":CHANnel{:s}:UNITs?".format(ch)).strip('\n'))
            ch_index += 1

        if self.number_channels_on == 0:
            self.scope.clear()
            self.scope.close()
            sys.exit("No data has been acquired. Properly closing scope and aborting script.")

        # chs_on = []
        ch_index = 1
        for ch in chs_list:
            if ch == 1:
                self.chs_on.append(int(ch_index))
            ch_index += 1

    def acquisition(self):
        """
        Setup data export - For repetitive acquisitions, this only needs to be done once unless settings are changed
        :param scope: scope object returned by 'connect' method.
        :return:
        """
        self.scope.write(":WAVeform:FORMat WORD")
        self.scope.write(":WAVeform:BYTeorder LSBFirst")
        self.scope.write(":WAVeform:UNSigned 0")

        acq_type = str(self.scope.query(":ACQuire:TYPE?")).strip("\n")
        # This can also be done when pulling pre-ambles (pre[1]) or may be known ahead of time,
        # but since the script is supposed to find everything, it is done now.

        if acq_type == "AVER" or acq_type == "HRES":
            points_mode = "NORMal"
            # If the :WAVeform:POINts:MODE is RAW, and the Acquisition Type is Average, the number of points
            # available is 0. If :WAVeform:POINts:MODE is MAX, it may or may not return 0 points.
            # If the :WAVeform:POINts:MODE is RAW, and the Acquisition Type is High Resolution, then the effect
            # is (mostly) the same as if the Acq. Type was Normal (no box-car averaging).
            # Note: if you use :SINGle to acquire the waveform in AVERage Acq. Type, no average is performed,
            # and RAW works. See sample script "InfiniiVision_2_Simple_Synchronization_Methods.py"
        else:
            points_mode = "RAW"

        # Note:
        # :WAVeform:POINts:MODE RAW corresponds to saving the ASCII XY or Binary data formats
        # to a USB stick on the scope
        # :WAVeform:POINts:MODE NORMal corresponds to saving the CSV or H5 data formats to a USB stick on the scope

        # Find max points for scope as is, ask for desired points, find how many points will actually be returned.
        # KEY POINT: the data must be on screen to be retrieved.  If there is data off-screen, :WAVeform:POINts? will
        # not "see it."
        # Addendum 1 shows how to properly get all data on screen, but this is never needed for Average and High
        # Resolution Acquisition Types, since they basically don't use off-screen data; what you see is what you get.

        self.scope.write(":WAVeform:SOURce CHANnel{:s}".format(self.chs_on[0]))
        self.scope.write(":WAVeform:POINts MAX")
        self.scope.write(":WAVeform:POINts:MODE {:s}".format(points_mode))
        max_currently_available_points = int(self.scope.query(":WAVeform:POINts?"))

        #  The scope will return a -222,"Data out of range" error if fewer than 100 points are requested,
        #  even though it may actually return fewer than 100 points.
        if self.USER_REQUESTED_POINTS < 100:
            self.USER_REQUESTED_POINTS = 100
        if self.USER_REQUESTED_POINTS > max_currently_available_points or acq_type == "PEAK":
            self.USER_REQUESTED_POINTS = max_currently_available_points

        # If one wants some other number of points...
        # Tell it how many points you want
        self.scope.write(":WAVeform:POINts " + str(self.USER_REQUESTED_POINTS))
        pts_to_retrieve = int(self.scope.query(":WAVeform:POINts?"))
        pre_amble = self.scope.query(":WAVeform:PREamble?").split(',')
        x_increment = float(pre_amble[4])
        x_origin = float(pre_amble[5])
        x_reference = float(pre_amble[6])

        data_time = ((np.linspace(0, pts_to_retrieve - 1, pts_to_retrieve) - x_reference) * x_increment) + x_origin
        if acq_type == "PEAK":
            data_time = np.repeat(data_time, 2)
        if acq_type == "PEAK":
            data_wave = np.zeros([2 * pts_to_retrieve, self.number_channels_on])
        else:
            data_wave = np.zeros([pts_to_retrieve, self.number_channels_on])

        # Get the waveform format
        wave_format = str(self.scope.query(":WAVeform:FORMat?"))
        if wave_format == "BYTE":
            format_multiplier = 1
        else:  # wave_format == "WORD"
            format_multiplier = 2

        if acq_type == "PEAK":
            points_multiplier = 2
        else:
            points_multiplier = 1

        total_bytes_to_xfer = points_multiplier * pts_to_retrieve * format_multiplier + 11
        # Why + 11?  The IEEE488.2 waveform header for definite length binary blocks (what this
        # will use) consists of 10 bytes.  The default termination character, \n, takes up another byte.
        # If you are using mutliplr termination characters, adjust accordingly.
        # Note that Python 2.7 uses ASCII, where all characters are 1 byte.
        # Python 3.5 uses Unicode, which does not have a set number of bytes per character.

        # Set chunk size for pyvisa.
        # More info: https://pyvisa.readthedocs.io/en/latest/introduction/resources.html#chunk-length
        if total_bytes_to_xfer >= 400000:
            self.scope.chunk_size = total_bytes_to_xfer

        # Any given user may want to tweak this for best throughput, if desired.  The 400,000 was
        # chosen after testing various chunk sizes over various transfer sizes, over USB, and
        # determined to be the best, or at least simplest, cutoff.  When the transfers are smaller,
        # the intrinsic "latencies" seem to dominate, and the default chunk size works fine.

        # How does the default chuck size work?
        # It just pulls the data repeatedly and sequentially (in series) until the termination character is found...

        # Do I need to adjust the timeout for a larger chunk sizes, where it will pull up to an
        # entire 8,000,000 sample record in a single IO transaction?
        #   If you use a 10s timeout (10,000 ms in PyVisa), that will be good enough for USB and LAN.
        #   If you are using GPIB, which is slower than LAN or USB, quite possibly, yes.
        #   If you don't want to deal with this, don't set the chunk size, and use a 10 second
        #   timeout, and everything will be fine in Python.
        # When you use the default chunk size, there are repeated IO transactions to pull the total
        # waveform.  It is each individual IO transaction that needs to complete within the timeout.

        time_start = time.clock()  # Only to show how long it takes to transfer and scale the data.
        i = 0  # index of Wav_data, recall that python indices start at 0, so ch1 is index 0
        for ch in self.chs_on:
            data_wave[:, i] = np.array(self.scope.query_binary_valuese(':WAVeform:SOURce CHANnel' + str(ch) + ';DATA?',
                                                                       datatype="h", is_big_endian=False))

            # Scaled_waveform_Data[*] = [(Unscaled_Waveform_Data[*] - Y_reference) * Y_increment] + Y_origin
            data_wave[:, i] = ((data_wave[:, i] - self.analog_vert_pres[2, ch-1])*self.analog_vert_pres[0, ch-1]+self.analog_vert_pres[1, ch-1])

        # Reset the chunk size back to default if needed.
        # If you don't do this, and now wanted to do something else... such as ask for a measurement
        # result, and leave the chunk size set to something large, it can really slow down the script,
        # so set it back to default, which works well.
        if total_bytes_to_xfer >= 400000:
            self.scope.chunk_size = 20480
        print("\n\nIt took {:.2f} seconds to transfer and scale {:d} channel(s).".format(time.clock() - time_start, self.number_channels_on))
        print("Each channel had {:d} points.\n".format(pts_to_retrieve))
        self.scope.clear()
        self.scope.close()



