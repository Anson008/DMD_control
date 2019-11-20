# INSTRUCTIONS:
# Edit in the VISA address of the oscilloscope
# Edit in the file save locations ## IMPORTANT NOTE:  This script WILL overwrite previously saved files!
# Manually (or write more code) acquire data on the oscilloscope.  Ensure that it finished (Run/Stop button is red).


import visa
import sys
import time
import numpy as np


class DSO6034A(object):
    def __init__(self, points, visa_address, name, directory, visa_dir="C:\\Windows\\System32\\visa32.dll", timeout=10000):
        self.USER_REQUESTED_POINTS = int(points)
        self.VISA_ADDRESS = visa_address
        self.GLOBAL_TIMEOUT = timeout
        self.FILE_NAME = name
        self.DIRECTORY = directory
        self.VISA_DIR = visa_dir

    def connect(self):
        """
        Define and open the scope by the VISA address. This uses PyVISA
        :return: scope object
        """
        rm = visa.ResourceManager(self.VISA_DIR)
        try:
            scope = rm.open_resource(self.VISA_ADDRESS)
        except Exception:
            print("Unable to connect to oscilloscope at {:s}. Aborting script.\n".format(self.VISA_ADDRESS))
            sys.exit()
        scope.timeout = self.GLOBAL_TIMEOUT  # Set Global Timeout
        scope.clear()  # Clear the instrument bus
        return scope

    def channels(self, scope):
        idn = str(scope.query("*IDN?"))
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

        number_channels_on = 0
        """
        After the CHS_LIST array is filled it could, for example look like: if chs 1,3 and 4 were on, 
        CHS_LIST = [1,0,1,1].
        """

        analog_vert_pres = np.zeros((3, 4))
        """
        analog_vert_pres = [[Y_INCrement_Ch1, Y_INCrement_Ch2, Y_INCrement_Ch3, Y_INCrement_Ch4], 
                            [Y_ORIGin_Ch1, Y_ORIGin_Ch2, Y_ORIGin_Ch3, Y_ORIGin_Ch4], 
                            [Y_REFerence_Ch1, Y_REFerence_Ch2, Y_REFerence_Ch3, Y_REFerence_Ch4]]
        """

        ch_units = ["BLANK", "BLANK", "BLANK", "BLANK"]
        scope.write(":WAVeform:POINts:MODE MAX")
        ch_index = 1  # Channel index
        for ch in chs_list:
            on_off = int(scope.query(":CHANnel{:s}:DISPlay?".format(ch)))
            if on_off == 1:
                channel_acquired = int(scope.query(":WAVeform:SOURce CHANnel{:s};POINts?".format(ch)))
            else:
                channel_acquired = 0
            if on_off == 0 or channel_acquired == 0:
                scope.write(":CHANnel{:s}:DISPlay OFF".format(ch))
                chs_list[ch-1] = 0
            else:
                chs_list[ch-1] = 1
                number_channels_on += 1
                pre_amble = scope.query(":WAVeform:PREamble?").split(",")
                analog_vert_pres[0, ch-1] = float(pre_amble[7])
                analog_vert_pres[1, ch-1] = float(pre_amble[8])
                analog_vert_pres[2, ch-1] = float(pre_amble[9])
                ch_units[ch-1] = str(scope.query(":CHANnel{:s}:UNITs?".format(ch)).strip('\n'))
            ch_index += 1

        if number_channels_on == 0:
            scope.clear()
            scope.close()
            sys.exit("No data has been acquired. Properly closing scope and aborting script.")

        chs_on = []
        ch_index = 1
        for ch in chs_list:
            if ch == 1:
                chs_on.append(int(ch))
            ch_index += 1
        return chs_on

    def acquisite(self, scope, channels):
        """
        Setup data export - For repetitive acquisitions, this only needs to be done once unless settings are changed
        :param scope: scope object returned by 'connect' method.
        :return:
        """
        scope.write(":WAVeform:FORMat WORD")
        scope.write(":WAVeform:BYTeorder LSBFirst")
        scope.write(":WAVeform:UNSigned 0")

        acq_type = str(scope.query(":ACQuire:TYPE?")).strip("\n")
        """
        This can also be done when pulling pre-ambles (pre[1]) or may be known ahead of time, 
        but since the script is supposed to find everything, it is done now.
        """

        if acq_type == "AVER" or acq_type == "HRES":
            points_mode = "NORMal"
            """
            If the :WAVeform:POINts:MODE is RAW, and the Acquisition Type is Average, the number of points 
            available is 0. If :WAVeform:POINts:MODE is MAX, it may or may not return 0 points.
            If the :WAVeform:POINts:MODE is RAW, and the Acquisition Type is High Resolution, then the effect 
            is (mostly) the same as if the Acq. Type was Normal (no box-car averaging).
            Note: if you use :SINGle to acquire the waveform in AVERage Acq. Type, no average is performed, 
            and RAW works. See sample script "InfiniiVision_2_Simple_Synchronization_Methods.py"
            """
        else:
            points_mode = "RAW"

        """
        Note:
        :WAVeform:POINts:MODE RAW corresponds to saving the ASCII XY or Binary data formats to a USB stick on the scope
        :WAVeform:POINts:MODE NORMal corresponds to saving the CSV or H5 data formats to a USB stick on the scope
        
        Find max points for scope as is, ask for desired points, find how many points will actually be returned.
        KEY POINT: the data must be on screen to be retrieved.  If there is data off-screen, :WAVeform:POINts? will 
        not "see it."
        Addendum 1 shows how to properly get all data on screen, but this is never needed for Average and High 
        Resolution Acquisition Types, since they basically don't use off-screen data; what you see is what you get.
        """
        scope.write(":WAVeform:SOURce CHANnel{:s}".format(channels[0]))
        scope.write(":WAVeform:POINts MAX")
        scope.write(":WAVeform:POINts:MODE {:s}".format(points_mode))
        max_currently_available_points = int(scope.query(":WAVeform:POINts?"))

        #  The scope will return a -222,"Data out of range" error if fewer than 100 points are requested,
        #  even though it may actually return fewer than 100 points.
        if self.USER_REQUESTED_POINTS < 100:
            self.USER_REQUESTED_POINTS = 100
        if self.USER_REQUESTED_POINTS > max_currently_available_points or acq_type == "PEAK":
            self.USER_REQUESTED_POINTS = max_currently_available_points

        # If one wants some other number of points...
        # Tell it how many points you want
        scope.write(":WAVeform:POINts " + str(self.USER_REQUESTED_POINTS))
        number_of_points_to_actually_retrieve = int(scope.query(":WAVeform:POINts?"))





