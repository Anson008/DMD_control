# INSTRUCTIONS:
# Edit in the VISA address of the oscilloscope
# Edit in the file save locations ## IMPORTANT NOTE:  This script WILL overwrite previously saved files!
# Manually (or write more code) acquire data on the oscilloscope.  Ensure that it finished (Run/Stop button is red).


import visa
import sys
import time


class DSO6034A(object):
    def __init__(self, points, visa_address, name, directory, visa_dir="C:\\Windows\\System32\\visa32.dll", timeout=10000):
        self.USER_REQUESTED_POINTS = points
        self.VISA_ADDRESS = visa_address
        self.GLOBAL_TIMEOUT = timeout
        self.FILE_NAME = name
        self.DIRECTORY = directory
        self.VISA_DIR = visa_dir

    def connect(self):
        """
        Define and open the scape by the VISA address. This uses PyVISA
        """
        rm = visa.ResourceManager(self.VISA_DIR)
        try:
            dso6032a = rm.open_resource(self.VISA_ADDRESS)
        except Exception:
            print("Unable to connect to oscilloscope at {:s}. Aborting script.\n".format(self.VISA_ADDRESS))
            sys.exit()
        dso6032a.timeout = self.GLOBAL_TIMEOUT  # Set Global Timeout
        dso6032a.clear()  # Clear the instrument bus
        return dso6032a

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
