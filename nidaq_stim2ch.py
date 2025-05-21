import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import nidaqmx
import time



niDevice = 'Dev1'
debug = False



class NIDAQ_Stim2ch():
    def __init__( self):
        self.niDevice = niDevice
        self.fs = 100e3 # Hz
        self.active = False

        self.time_start      = None
        self.time_stop       = None
        self.time            = None
        self.output_ch1      = None
        self.output_ch2      = None
        self.recording_ch1   = None
        self.recording_ch2   = None
        self.recording_ch3   = None
        self.stimID          = None


    def run(self, output_ch1, output_ch2, stimID=None, waitForCamera=False, channelOut=None, channelIn=None):
        self.time = np.arange(output_ch1.shape[-1]) / self.fs
        self.time_waitUser = 30
        self.output_ch1      = output_ch1
        self.output_ch2      = output_ch2
        self.stimID          = stimID

        if debug:
            self.recording_ch1 = np.sin(self.time*4*3.14)
            self.recording_ch2 = np.cos(self.time*4*3.14)
            self.recording_ch3 = np.sin(self.time*4*3.14)
            self.time_start = datetime.now()
            self.time_stop = datetime.now()
            self.plot_stim()
            return
        
        # Initialize Pins
        self.active = True
        nidaq_pins = NIDAQ_Pins()
        if (not nidaq_pins.test_connection()):
            print("Could not connect to NIDAQ.")
            return

        # Define Output Data
        daqOutData  = np.concatenate(( self.output_ch1, self.output_ch2)).reshape(2,-1)
        nsamples    = daqOutData.shape[-1]
        if channelOut is None:
            channelOut  = self.niDevice+"/ao0:1"
        if channelIn is None:
            channelIn   = self.niDevice+"/ai0,"+self.niDevice+"/ai1,"+self.niDevice+"/ai16"
        channelTrig = self.niDevice+"/ai0" # Has to be self.niDevice+"/ai0" or "apfi0"

        # Configure Analog Pins
        with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
            write_task.ao_channels.add_ao_voltage_chan(channelOut, min_val=-5, max_val=5)
            read_task.ai_channels.add_ai_voltage_chan(channelIn, min_val=-5, max_val=5)
            for task in (read_task, write_task):
                task.timing.cfg_samp_clk_timing(rate=self.fs, source='OnboardClock', samps_per_chan=nsamples)
                if waitForCamera:
                    task.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS
                    task.triggers.start_trigger.delay = self.trigger_delay

            write_task.triggers.start_trigger.cfg_dig_edge_start_trig(read_task.triggers.start_trigger.term)
            if waitForCamera:
                read_task.triggers.start_trigger.cfg_anlg_edge_start_trig( trigger_source=channelTrig, trigger_slope=nidaqmx.constants.Slope.FALLING, trigger_level=1)

            write_task.write(daqOutData, auto_start=False)

            self.time_start = datetime.now()
            write_task.start()
            [self.recording_ch1, self.recording_ch2, self.recording_ch3] = read_task.read(nsamples, timeout=self.time[-1] + self.time_waitUser )
            #TODO: is there a way to get timestamps from read?
            self.time_stop = datetime.now()

        # Set Digital Pins
        self.active = False

        return

    
    def get_matVars(self):
        matVars = {}

        matVars['fs']           = self.fs
        matVars['data_output']  = np.concatenate(( self.output_ch1, self.output_ch2)).reshape(2,-1)
        matVars['data_record']  = np.concatenate(( self.recording_ch1, self.recording_ch2, self.recording_ch3)).reshape(3,-1)
        matVars['time']         = self.time.reshape(1,-1)
        matVars['time_start']   = self.time_start.timestamp()
        matVars['time_stop']    = self.time_stop.timestamp()
        matVars['stimID']       = self.stimID if (self.stimID is not None) else -1

        return matVars


    def plot_stim(self):
        #plot figures
        plt.figure()
        ax0 = plt.subplot(2,1,1)
        plt.plot(self.time, self.output_ch1, 'b')
        plt.ylabel("Channel 1 (V)")

        ax1 = plt.subplot(2,1,2,sharex=ax0)
        plt.plot(self.time, self.output_ch2, 'g')
        plt.ylabel("Channel 2 (V)")
        
        plt.show()


class NIDAQ_Pins:
    channelDigital = niDevice+"/port0/line0:6"

    def test_connection(self):
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.channelDigital)
                return 1
        except nidaqmx.errors.DaqError:
            print("Could not connect to NIDAQ")
            return 0
        
    def nidaq_write(self, my_data):
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.channelDigital)
                data = np.array(my_data, np.uint32)
                task.write(data,auto_start=True)
        except nidaqmx.errors.DaqError:
            print("Could not connect to NIDAQ")
'''
class NIDAQ_Pins:
    eo = 0
    channel = niDevice+"/port0/line0:6"

    def test_connection(self):
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.channel)
                return 1
        except nidaqmx.errors.DaqError:
            print("Could not connect to NIDAQ")
            return 0

    def disable_switches(self):
        self.eo = 0
        self.nidaq_write([self.eo])

    def enable_switches(self):
        self.eo = 1
        self.nidaq_write([self.eo])
        
    def enable_stim( self, stimCH0, stimCH1, stimCH2):
        data = (stimCH2 << 0) | (stimCH1 << 4) | (stimCH0 << 1) | (self.eo)
        self.nidaq_write([data])
    
    def disable_stim( self):
        self.nidaq_write([self.eo])

    # time in ms
    def write_srclk_pusle(self, data, pulse_length):
        data1 = (data << 3) | self.eo
        data2 = (data << 3) | 0b10 | self.eo
        self.nidaq_write([data1, data2])
        time.sleep(pulse_length*0.001)
        data3 = (data << 3) | self.eo
        self.nidaq_write([data3])

    # time in ms
    def write_rclk_pusle(self, pulse_length):
        data1 = 0b100 | self.eo
        self.nidaq_write([data1])
        time.sleep(pulse_length*0.001)
        data2 = self.eo
        self.nidaq_write([data2])
    
    def nidaq_write(self, my_data):
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.channel)
                data = np.array(my_data, np.uint32)
                task.write(data,auto_start=True)
        except nidaqmx.errors.DaqError:
            print("Could not connect to NIDAQ")

    def write_pins(self, data, rclk, srclk):
        data = (data << 3) | (rclk << 2) | (srclk << 1) | self.eo
        self.nidaq_write([data])
'''