from bottleneck import move_mean, move_median, move_sum
from scipy.signal import butter, filtfilt, find_peaks
from nidaqmx import stream_readers, constants
from PyQt6 import QtCore, QtGui, QtWidgets
from scipy.fft import fft, fftfreq
import helperFunctions_UI
import numpy as np
import threading
import datetime
import nidaqmx
import wave
import time
import sys
import os



class mainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):


        self.num_channels     = 3
        self.defaultChannels  = ['ai0','ai1', 'ai16']
        self.niDevice         = 'Dev1'
        self.plot_refreshRate = 20 # Hz
        self.fs               = 50e3 # Hz
        self.ai_channel_order = ['ai0','ai1','ai2','ai3','ai4','ai5','ai6','ai7','ai16','ai17','ai18','ai19','ai20','ai21','ai22','ai23']


        if len(sys.argv) > 1:
            self.num_channels = int(sys.argv[1])
        if len(sys.argv) > 2:
            self.niDevice     = sys.argv[2]

        super(mainWidget, self).__init__(parent)
        self.setWindowTitle("NiDAQ Stream")
        self.pathSave_default = os.path.join('.','data_raw')
        self.displayPaused = False
        self.dataToDisplay = []
        self.recording = False
        self.displayIndex = 0
        self.stopStream = False

        if not(os.path.exists(self.pathSave_default)) and not(os.path.isdir(self.pathSave_default)):
            os.mkdir(self.pathSave_default)
            
        # Create the graphs
        self.graphLayout = QtWidgets.QGridLayout()
        self.graphicsViews = []
        self.plotItems = []
        self.plots = []
        # Create the processing options
        self.process_gb      = []
        self.channel_dd      = []
        self.filterL_l       = []
        self.filterL_t       = []
        self.filterH_l       = []
        self.filterH_t       = []
        self.filterOrder_s   = []
        self.psd_cb          = []
        self.abs_cb          = []
        self.zNormalize_cb   = []
        self.detectPeaks_cb  = []
        self.detectPeaksC_l  = []
        self.detectPeaksR_l  = []
        self.peakRate_cb     = []
        self.peakRate_l      = []
        self.peakHeight      = []
        self.peakThreshold   = []
        self.peakDistance    = []
        self.peakProminence  = []
        self.peakWidth       = []
        self.movingMean_cb   = []
        self.movingMean_t    = []
        self.movingMedian_cb = []
        self.movingMedian_t  = []
        self.movingRMS_cb    = []
        self.movingRMS_t     = []
        self.movingSum_cb    = []
        self.movingSum_t     = []
        self.processLayout   = []
        colors = [(200,200,0), (50,200,0), (100,200,200), (200,0,200)]
        for i in range(self.num_channels):
            graphicsView_data = helperFunctions_UI.makeGraphicsView()
            plotItem_data = graphicsView_data.getPlotItem()
            plotItem_data.setTitle(title='Channel '+str(i))
            plotItem_data.setLabel('left',text='Voltage', units='V')
            plotItem_data.showLabel('left',show=True)
            if i == (self.num_channels-1):
                plotItem_data.setLabel('bottom',text='Time',units='s')
            else:
                plotItem_data.setLabel('bottom')
            plotItem_data.showLabel('bottom',show=True)
            plotItem_data.showGrid(x=True,y=True)

            plot_data = plotItem_data.plot()
            plot_data.setPen(colors[i%len(colors)])

            graphicsView_freq = helperFunctions_UI.makeGraphicsView()
            plotItem_freq = graphicsView_freq.getPlotItem()
            plotItem_freq.setTitle(title='Channel '+str(i)+' PSD')
            plotItem_freq.setLabel('left',text='Magnitude', units='dB/Hz')
            plotItem_freq.showLabel('left',show=True)
            plotItem_freq.setLabel('bottom')
            plotItem_freq.showLabel('bottom',show=True)
            plotItem_freq.showGrid(x=False,y=True)
            plotItem_freq.setLogMode(x=False,y=True)

            plot_freq = plotItem_freq.plot()
            plot_freq.setPen(colors[i%len(colors)])

            graphicsView_data.setVisible(True)
            graphicsView_freq.setVisible(False)

            self.graphicsViews.append( [graphicsView_data, graphicsView_freq])
            self.plotItems.append( [plotItem_data, plotItem_freq])
            self.plots.append( [plot_data, plot_freq])
            self.graphLayout.addWidget(self.graphicsViews[i][0], i, 0)
            
        
            parent_gb       = helperFunctions_UI.makeGroupBox('Channel '+str(i))
            channel_t       = helperFunctions_UI.makeTextBox(parent_gb, text='[Optional Label]')
            channel_dd      = helperFunctions_UI.makeComboBox(parent_gb, items=self.ai_channel_order)
            filterL_l       = helperFunctions_UI.makeLabel('Filter Cutoffs  Low:', parent_gb)
            filterL_t       = helperFunctions_UI.makeTextBox(parent_gb)
            filterH_l       = helperFunctions_UI.makeLabel('  High:', parent_gb)
            filterH_t       = helperFunctions_UI.makeTextBox(parent_gb)
            filterOrder_s   = helperFunctions_UI.makeSliderBar(parent_gb, 1, 10, 1, initialValue=5)
            psd_cb          = helperFunctions_UI.makeCheckBox('Calculate PSD', parent_gb)
            abs_cb          = helperFunctions_UI.makeCheckBox('Absolute Value', parent_gb)
            zNormalize_cb   = helperFunctions_UI.makeCheckBox('Z-Score Normalize', parent_gb)
            detectPeaks_cb  = helperFunctions_UI.makeCheckBox('Detect Peaks', parent_gb)
            detectPeaksCC_l = helperFunctions_UI.makeLabel('  Count: ', parent_gb)
            detectPeaksC_l  = helperFunctions_UI.makeLabel('--', parent_gb)
            detectPeaksRR_l = helperFunctions_UI.makeLabel('  Rate: ', parent_gb)
            detectPeaksR_l  = helperFunctions_UI.makeLabel('--', parent_gb)
            movingMean_cb   = helperFunctions_UI.makeCheckBox('Moving Mean  window(ms):', parent_gb)
            movingMean_t    = helperFunctions_UI.makeTextBox(parent_gb)
            movingMedian_cb = helperFunctions_UI.makeCheckBox('Moving Median  window(ms):', parent_gb)
            movingMedian_t  = helperFunctions_UI.makeTextBox(parent_gb)
            movingRMS_cb    = helperFunctions_UI.makeCheckBox('Moving RMS  window(ms):', parent_gb)
            movingRMS_t     = helperFunctions_UI.makeTextBox(parent_gb)
            movingSum_cb    = helperFunctions_UI.makeCheckBox('Moving Sum  window(ms):', parent_gb)
            movingSum_t     = helperFunctions_UI.makeTextBox(parent_gb)
            peakHeight_l     = helperFunctions_UI.makeLabel('    peak Height: ', parent_gb)
            peakHeight_t     = helperFunctions_UI.makeTextBox(parent_gb)
            peakThreshold_l  = helperFunctions_UI.makeLabel('    peak Threshold: ', parent_gb)
            peakThreshold_t  = helperFunctions_UI.makeTextBox(parent_gb)
            peakDistance_l   = helperFunctions_UI.makeLabel('    peak Distance: ', parent_gb)
            peakDistance_t   = helperFunctions_UI.makeTextBox(parent_gb)
            peakProminence_l = helperFunctions_UI.makeLabel('    peak Prominence: ', parent_gb)
            peakProminence_t = helperFunctions_UI.makeTextBox(parent_gb)
            peakWidth_l      = helperFunctions_UI.makeLabel('    peak Width: ', parent_gb)
            peakWidth_t      = helperFunctions_UI.makeTextBox(parent_gb)

            try:
                channel_dd.setCurrentIndex(self.ai_channel_order.index(self.defaultChannels[i]))
            except:
                print('Default Channel(s) not found.')
                channel_dd.setCurrentIndex(i)
            placeholder_l   = helperFunctions_UI.makeLabel(' ',parent_gb)
            channel_dd.currentIndexChanged.connect(self.initializeGraphing)
            filterL_t.textChanged.connect( self.calculateFilterCoef)
            filterH_t.textChanged.connect( self.calculateFilterCoef)
            filterOrder_s.valueChanged.connect( self.calculateFilterCoef)
            abs_cb.stateChanged.connect( self.updateAbs)
            zNormalize_cb.stateChanged.connect( self.updateZNormalize)
            movingMean_cb.stateChanged.connect( self.calculateMovingMeanWindow)
            movingMean_t.textChanged.connect( self.calculateMovingMeanWindow)
            movingMedian_cb.stateChanged.connect( self.calculateMovingMedianWindow)
            movingMedian_t.textChanged.connect( self.calculateMovingMedianWindow)
            movingRMS_cb.stateChanged.connect( self.calculateMovingRMSWindow)
            movingRMS_t.textChanged.connect( self.calculateMovingRMSWindow)
            movingSum_cb.stateChanged.connect( self.calculateMovingSumWindow)
            movingSum_t.textChanged.connect( self.calculateMovingSumWindow)
            psd_cb.stateChanged.connect( self.updateFreq)
            detectPeaks_cb.stateChanged.connect( self.updatePeaks)
            peakHeight_t.textChanged.connect( self.updatePeaks)
            peakThreshold_t.textChanged.connect( self.updatePeaks)
            peakDistance_t.textChanged.connect( self.updatePeaks)
            peakProminence_t.textChanged.connect( self.updatePeaks)
            peakWidth_t.textChanged.connect( self.updatePeaks)

            channelLayout    = helperFunctions_UI.makeHorizontalLayout([channel_t,channel_dd])
            filterLayout     = helperFunctions_UI.makeHorizontalLayout([filterL_l,filterL_t,filterH_l,filterH_t])
            movingMeanLayout = helperFunctions_UI.makeHorizontalLayout([movingMean_cb,movingMean_t])
            movingMedianLayout = helperFunctions_UI.makeHorizontalLayout([movingMedian_cb,movingMedian_t])
            movingRMSLayout  = helperFunctions_UI.makeHorizontalLayout([movingRMS_cb,movingRMS_t])
            movingSumLayout  = helperFunctions_UI.makeHorizontalLayout([movingSum_cb,movingSum_t])
            detectPeaksLayout   = helperFunctions_UI.makeHorizontalLayout([detectPeaks_cb,detectPeaksCC_l,detectPeaksC_l,detectPeaksRR_l,detectPeaksR_l])
            peakSettings1Layout = helperFunctions_UI.makeHorizontalLayout([peakHeight_l,peakHeight_t,peakWidth_l,peakWidth_t]) # peakThreshold_l,peakThreshold_t,
            peakSettings2Layout = helperFunctions_UI.makeHorizontalLayout([peakProminence_l,peakProminence_t,peakDistance_l,peakDistance_t])
            processLayout = helperFunctions_UI.makeVerticalLayout([ channelLayout,
                                                                    filterLayout,
                                                                    filterOrder_s,
                                                                    abs_cb,
                                                                    zNormalize_cb,
                                                                    movingMeanLayout,
                                                                    movingMedianLayout,
                                                                    movingRMSLayout,
                                                                    movingSumLayout,
                                                                    psd_cb,
                                                                    detectPeaksLayout,
                                                                    peakSettings1Layout,
                                                                    peakSettings2Layout,
                                                                    ])
            placeholder_layout = helperFunctions_UI.makeVerticalLayout([placeholder_l])
            processLayout.addLayout(placeholder_layout,99)
            
            self.process_gb.append(parent_gb)
            self.channel_dd.append(channel_dd)
            self.filterL_t.append(filterL_t)
            self.filterH_t.append(filterH_t)
            self.filterOrder_s.append(filterOrder_s)
            self.psd_cb.append(psd_cb)
            self.abs_cb.append(abs_cb)
            self.zNormalize_cb.append(zNormalize_cb)
            self.detectPeaks_cb.append(detectPeaks_cb)
            self.detectPeaksC_l.append(detectPeaksC_l)
            self.detectPeaksR_l.append(detectPeaksR_l)
            self.peakHeight.append(peakHeight_t)
            self.peakThreshold.append(peakThreshold_t)
            self.peakDistance.append(peakDistance_t)
            self.peakProminence.append(peakProminence_t)
            self.peakWidth.append(peakWidth_t)
            self.movingMean_cb.append(movingMean_cb)
            self.movingMean_t.append(movingMean_t)
            self.movingMedian_cb.append(movingMedian_cb)
            self.movingMedian_t.append(movingMedian_t)
            self.movingRMS_cb.append(movingRMS_cb)
            self.movingRMS_t.append(movingRMS_t)
            self.movingSum_cb.append(movingSum_cb)
            self.movingSum_t.append(movingSum_t)
            self.processLayout.append(processLayout)

        self.processLayout  = helperFunctions_UI.makeVerticalLayout(self.processLayout)
        self.channelLayout  = helperFunctions_UI.makeHorizontalLayout([self.processLayout])
        self.channelLayout.addLayout(self.graphLayout, 99)

        # Metadata Layout
        self.metadata_gb    = helperFunctions_UI.makeGroupBox('Metadata')
        self.niDevice_l     = helperFunctions_UI.makeLabel('Device', self.metadata_gb )
        self.niDevice_t     = helperFunctions_UI.makeTextBox( self.metadata_gb)
        self.niDevice_t.setText(self.niDevice)
        self.niDevice_t.textChanged.connect(self.initializeGraphing)
        self.voltageRail_dd = helperFunctions_UI.makeComboBox(self.metadata_gb, items=['0.1 V','0.2 V','0.5 V','1 V','2 V','5 V','10 V'])
        self.voltageRail_dd.setCurrentIndex(5)
        self.voltageRail_dd.currentIndexChanged.connect(self.initializeGraphing)
        self.sampleRate_l   = helperFunctions_UI.makeLabel('Sample Rate (kHz)', self.metadata_gb )
        self.sampleRate_s   = helperFunctions_UI.makeSpinBox( self.metadata_gb, 1, 250, 0, 10)
        self.sampleRate_s.setValue(self.fs/(1e3))
        self.sampleRate_s.valueChanged.connect(self.initializeGraphing)
        self.refreshRate_l  = helperFunctions_UI.makeLabel('Refresh Rate (Hz)', self.metadata_gb )
        self.refreshRate_s  = helperFunctions_UI.makeSpinBox( self.metadata_gb, 1, 120, 0, 1)
        self.refreshRate_s.setValue(self.plot_refreshRate)
        self.refreshRate_s.valueChanged.connect(self.initializeGraphing)
        self.displayTime_l  = helperFunctions_UI.makeLabel('Display Time (s)', self.metadata_gb )
        self.displayTime_t  = helperFunctions_UI.makeTextBox(self.metadata_gb )
        self.displayTime_t.setText('2')
        self.displayTime_t.textChanged.connect(self.initializeGraphing)
        self.pause_b        = helperFunctions_UI.makeButton('Pause', self.metadata_gb)
        self.pause_b.clicked.connect(self.click_pause)
        self.rescale_b      = helperFunctions_UI.makeButton('Rescale Plots', self.metadata_gb)
        self.rescale_b.clicked.connect(self.click_rescale)
        self.linkAxes_cb    = helperFunctions_UI.makeCheckBox('Link Axes', self.metadata_gb)
        self.linkAxes_cb.stateChanged.connect(  self.check_linkAxes)
        self.labelLow_b     = helperFunctions_UI.makeButton('Label 0', self.metadata_gb)
        self.labelHigh_b    = helperFunctions_UI.makeButton('Label as:', self.metadata_gb)
        self.labelNumber_sb = helperFunctions_UI.makeSpinBox( self.metadata_gb, 1, 2**16, 0, 1)
        self.pathSave_l     = helperFunctions_UI.makeLabel('Save To:', self.metadata_gb )
        self.pathSave_t     = helperFunctions_UI.makeTextBox(self.metadata_gb )
        self.pathSave_b     = helperFunctions_UI.makeButton('Browse', self.metadata_gb )
        self.pathSave_t.setText(self.pathSave_default)
        self.pathSave_b.clicked.connect(self.click_pathSave)
        self.record_b       = helperFunctions_UI.makeButton('Record', self.metadata_gb )
        self.record_b.clicked.connect(self.click_record)
        width = 100
        # self.displayTime_l.setFixedWidth(width)
        # self.pause_b.setFixedWidth(width*2)
        # self.rescale_b.setFixedWidth(width*2)
        # self.linkAxes_cb.setFixedWidth(width*2)
        # self.labelNumber_sb.setFixedWidth(width)
        # self.displayTime_t.setFixedWidth(width*2)
        # self.pathSave_l.setFixedWidth(width)
        # self.pathSave_b.setFixedWidth(width)

        self.experimentLayout = helperFunctions_UI.makeHorizontalLayout([self.niDevice_l,
                                                                         self.niDevice_t,
                                                                         self.voltageRail_dd,
                                                                         self.sampleRate_l,
                                                                         self.sampleRate_s,
                                                                         self.refreshRate_l,
                                                                         self.refreshRate_s,
                                                                         self.displayTime_l,
                                                                         self.displayTime_t, 
                                                                         self.pause_b, 
                                                                         self.rescale_b, 
                                                                         self.linkAxes_cb, 
                                                                         self.pathSave_l, 
                                                                         self.pathSave_t, 
                                                                         self.pathSave_b,
                                                                         self.record_b])

        self.mainLayout = helperFunctions_UI.makeVerticalLayout([ self.experimentLayout, self.channelLayout])

        self.setLayout(self.mainLayout)

        self.displayData = None
        self.displayIndicies  = None
        self.plotX_data = None
        self.filterCoef = None
        self.calculateAbs = None
        self.calculateZNormalize = None
        self.movingMeanWindow   = None
        self.movingMedianWindow = None
        self.movingRMSWindow    = None
        self.movingSumWindow    = None
        self.calculatePSD = None
        self.detectPeaks  = None
        self.peaksKwargs  = None

        self.thread    = None
        self.plotTimer = QtCore.QTimer()
        self.plotTimer.timeout.connect(self.updatePlot)
        self.initializeGraphing()


    def initializeGraphing(self):
        self.stopStream = True
        if self.recording:
            self.click_record()
        # Formally close the thread
        if self.thread is not None:
            self.plotTimer.stop()
            self.thread.join(timeout=1)

        self.fs               = self.sampleRate_s.value() * 1e3
        self.plot_refreshRate = self.refreshRate_s.value()
        self.displayTime      = float(self.displayTime_t.toPlainText())
        self.niDevice         = self.niDevice_t.toPlainText()

        self.displayData = -2 * np.ones(int(self.num_channels*self.displayTime*self.fs))
        self.displayIndicies  = np.arange(0, int(self.displayTime * self.fs * self.num_channels))
        self.plotX_data = np.linspace(-1*self.displayTime, 0, num=int(self.displayTime*self.fs)).astype(float)
        self.filterCoef = [None] * self.num_channels
        self.calculateAbs = [False] * self.num_channels
        self.calculateZNormalize = [False] * self.num_channels
        self.movingMeanWindow   = [0] * self.num_channels
        self.movingMedianWindow = [0] * self.num_channels
        self.movingRMSWindow    = [0] * self.num_channels
        self.movingSumWindow    = [0] * self.num_channels
        self.calculatePSD = [False] * self.num_channels
        self.detectPeaks  = [False] * self.num_channels
        self.peaksKwargs  = [{}] * self.num_channels

        self.thread = threading.Thread(target=self.stream)
        self.thread.daemon = True

        self.stopStream = False
        self.thread.start()
        self.plotTimer.start(int((1/self.plot_refreshRate)*1e3))
        
        self.calculateFilterCoef()
        self.updateAbs()
        self.updateZNormalize()
        self.calculateMovingMeanWindow()
        self.calculateMovingMedianWindow()
        self.calculateMovingRMSWindow()
        self.calculateMovingSumWindow()
        self.updateFreq()
        self.updatePeaks()


    def stream(self):
        channelIn = ','.join([self.niDevice+'/'+ch.currentText().strip().split()[0].lower() for ch in self.channel_dd])

        with nidaqmx.Task() as task:
            samples_per_buffer = int(self.fs // self.plot_refreshRate)

            voltage_rail = float(self.voltageRail_dd.currentText()[:-2])
            task.ai_channels.add_ai_voltage_chan(channelIn, min_val=-1*voltage_rail, max_val=voltage_rail)
            task.timing.cfg_samp_clk_timing(rate=self.fs, source='OnboardClock', samps_per_chan=samples_per_buffer, sample_mode=constants.AcquisitionType.CONTINUOUS)
            # [daqInData1, daqInData2] = task.read(samples_per_buffer)
            
            reader = stream_readers.AnalogMultiChannelReader(task.in_stream)
            # writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(task.out_stream)

            def reading_task_callback(task_idx, event_type, num_samples, callback_data=None):
                '''After data has been read into the NI buffer this callback is called to read in the data from the buffer.
                Args:
                    task_idx (int): Task handle index value
                    event_type (nidaqmx.constants.EveryNSamplesEventType): ACQUIRED_INTO_BUFFER
                    num_samples (int): Number of samples that was read into the buffer.
                    callback_data (object)[None]: No idea. Documentation says: The callback_data parameter contains the value
                        you passed in the callback_data parameter of this function.
                '''
                buffer = np.zeros((self.num_channels, num_samples), dtype=np.float64)
                reader.read_many_sample(buffer, num_samples, timeout=nidaqmx.constants.WAIT_INFINITELY)

                # Convert the data from channel as a row order to channel as a column
                self.dataToDisplay.append(buffer)
                return 0

            task.register_every_n_samples_acquired_into_buffer_event(samples_per_buffer, reading_task_callback)
            task.start()
            
            while not self.stopStream:
                time.sleep(0.2)
                continue


    def updatePlot(self):
        if len(self.dataToDisplay) == 0:
            return
        
        newData = self.dataToDisplay
        self.dataToDisplay = []
        newData = np.concatenate(newData,axis=-1).flatten(order='F')

        if self.recording:
            self.fileRecording.write(newData.tobytes())

        # Necessary for autoscaling
        if len(self.plotX_data) != int(self.displayTime * self.fs) or (len(self.plotX_data) * self.num_channels) != len(self.displayIndicies):
            # Updates display indicies
            self.displayIndicies  = np.arange(0, int(self.displayTime * self.fs * self.num_channels))
            # Sets up the x axis for the data based on display time
            self.plotX_data = np.linspace(-1*self.displayTime, 0, num=int(self.displayTime * self.fs)).astype(float)

        # Shows if new data overflows
        if newData.shape[-1] > (self.displayTime*self.fs*self.num_channels):
            print('overflow!')
            newData = newData[-1*int(self.displayTime*self.fs*self.num_channels):]
        
        # Calculates the index where new data starts
        newDataSize = newData.shape[-1]
        newDataIndex = np.arange( self.displayIndex, self.displayIndex + newDataSize)
        # Puts new data into the new data indices
        np.put( self.displayData,  newDataIndex, newData,  mode='wrap')
        # Updates display index
        self.displayIndex += newDataSize
        # Sets up current display indices
        displayIndiciesCurrent = np.add( self.displayIndicies, self.displayIndex)
        # Gets the y axis for the data
        displayChannel  = np.take( self.displayData,  displayIndiciesCurrent, mode='wrap').astype(np.float64)

        if self.recording:
            self.plotX_data = np.add(self.plotX_data, newDataSize/(self.fs*self.num_channels))

        # print('\nmax: ', max(displayChannel), max(displayChannel) * self.convertToInt)
        # print('min: ', min(displayChannel), min(displayChannel) * self.convertToInt)
        if not self.displayPaused:
            for i in range(self.num_channels):
                d = displayChannel[i::self.num_channels]
                
                if self.filterCoef[i] is not None:
                    m = np.mean(d)
                    d = filtfilt( self.filterCoef[i][0], self.filterCoef[i][1], d-m) + m
                    
                if self.calculateAbs[i]:
                    d = np.abs(d)
                if self.calculateZNormalize[i]:
                    d = (d - np.mean(d)) / np.std(d)
                if self.movingMeanWindow[i] > 0:
                    d = move_mean( d, self.movingMeanWindow[i], min_count=1)
                if self.movingMedianWindow[i] > 0:
                    d = move_median( d, self.movingMedianWindow[i], min_count=1)
                if self.movingRMSWindow[i] > 0:
                    d = np.sqrt( move_mean( np.square(d), self.movingRMSWindow[i], min_count=1))
                if self.movingSumWindow[i] > 0:
                    d = move_sum( d, self.movingSumWindow[i], min_count=1)
                if self.calculatePSD[i]:
                    d_freq = self.coefficientPSD * np.square( np.abs( fft(d - np.mean(d))[:d.shape[-1]//2]))
                    d_freq[0] = d_freq[1:].min()
                    self.plots[i][1].setData(y=d_freq, x=self.plotX_freq)
                if self.detectPeaks[i]:
                    peak_i, _ = find_peaks( d, **self.peaksKwargs[i])
                    self.detectPeaksC_l[i].setText(str(peak_i.shape[-1]))
                    if peak_i.shape[-1] > 0:
                        rate_Hz = self.fs / np.mean(np.diff(peak_i))
                    else:
                        rate_Hz = 0
                    self.detectPeaksR_l[i].setText('{:.3f} Hz | {:.3f} /min'.format(rate_Hz,rate_Hz*60))
                
                self.plots[i][0].setData(y=d, x=self.plotX_data)


    def calculateFilterCoef(self):
        nyq = self.fs * 0.5
        for i in range(self.num_channels):
            low  = self.filterL_t[i].toPlainText().strip()
            high = self.filterH_t[i].toPlainText().strip()
            order = self.filterOrder_s[i].value()
            try:
                if len(low) > 0:
                    if len(high) > 0:
                        self.filterCoef[i] = butter( order, [float(low)/nyq,float(high)/nyq], btype='bandpass', analog=False, output='ba')
                    else:
                        self.filterCoef[i] = butter( order, float(low)/nyq, btype='high', analog=False, output='ba')
                elif len(high) > 0:
                    self.filterCoef[i] = butter( order, float(high)/nyq, btype='low', analog=False, output='ba')
                else:
                    self.filterCoef[i] = None
            except:
                print('ERROR: Unable to calculate coefficients for "', low, '" and "', high, '"')
                self.filterCoef[i] = None


    def updateAbs(self):
        for i in range(self.num_channels):
            self.calculateAbs[i] = self.abs_cb[i].isChecked()
    

    def updateZNormalize(self):
        for i in range(self.num_channels):
            self.calculateZNormalize[i] = self.zNormalize_cb[i].isChecked()
    

    def calculateMovingMeanWindow(self):
        for i in range(self.num_channels):
            if self.movingMean_cb[i].isChecked():
                self.movingMean_t[i].setVisible(True)
                try:
                    self.movingMeanWindow[i] = int((float(self.movingMean_t[i].toPlainText()) / 1e3) * self.fs)
                except:
                    print('ERROR: Unable to calculate window size for "', self.movingMean_t[i].toPlainText(), '"')
                    self.movingMeanWindow[i] = 0
            else:
                self.movingMean_t[i].setVisible(False)
                self.movingMeanWindow[i] = 0


    def calculateMovingMedianWindow(self):
        for i in range(self.num_channels):
            if self.movingMedian_cb[i].isChecked():
                self.movingMedian_t[i].setVisible(True)
                try:
                    self.movingMedianWindow[i] = int((float(self.movingMedian_t[i].toPlainText()) / 1e3) * self.fs)
                except:
                    print('ERROR: Unable to calculate window size for "', self.movingMedian_t[i].toPlainText(), '"')
                    self.movingMedianWindow[i] = 0
            else:
                self.movingMedian_t[i].setVisible(False)
                self.movingMedianWindow[i] = 0


    def calculateMovingRMSWindow(self):
        for i in range(self.num_channels):
            if self.movingRMS_cb[i].isChecked():
                self.movingRMS_t[i].setVisible(True)
                try:
                    self.movingRMSWindow[i] = int((float(self.movingRMS_t[i].toPlainText()) / 1e3) * self.fs)
                except:
                    print('ERROR: Unable to calculate window size for "', self.movingRMS_t[i].toPlainText(), '"')
                    self.movingRMSWindow[i] = 0
            else:
                self.movingRMS_t[i].setVisible(False)
                self.movingRMSWindow[i] = 0


    def calculateMovingSumWindow(self):
        for i in range(self.num_channels):
            if self.movingSum_cb[i].isChecked():
                self.movingSum_t[i].setVisible(True)
                try:
                    self.movingSumWindow[i] = int((float(self.movingSum_t[i].toPlainText()) / 1e3) * self.fs)
                except:
                    print('ERROR: Unable to calculate window size for "', self.movingSum_t[i].toPlainText(), '"')
                    self.movingSumWindow[i] = 0
            else:
                self.movingSum_t[i].setVisible(False)
                self.movingSumWindow[i] = 0
    

    def updateFreq(self):
        self.plotX_freq = fftfreq(self.plotX_data.shape[-1], d=1/self.fs)[:self.plotX_data.shape[-1]//2]
        self.coefficientPSD = 2/(self.fs*self.plotX_data.shape[-1])
        for i in range(self.num_channels):
            if self.psd_cb[i].isChecked():
                self.graphicsViews[i][1].setVisible(True)
                self.plotItems[i][1].setXRange(0,5e3)
                self.calculatePSD[i] = True
            else:
                self.graphicsViews[i][1].setVisible(False)
                self.calculatePSD[i] = False

    
    def updatePeaks(self):
        for i in range(self.num_channels):
            if self.detectPeaks_cb[i].isChecked():
                self.detectPeaks[i] = True
                self.peakHeight[i].setVisible(True)
                self.peakThreshold[i].setVisible(True)
                self.peakDistance[i].setVisible(True)
                self.peakProminence[i].setVisible(True)
                self.peakWidth[i].setVisible(True)

                self.peaksKwargs[i] = {}
                h = self.peakHeight[i].toPlainText()
                t = self.peakThreshold[i].toPlainText()
                d = self.peakDistance[i].toPlainText()
                p = self.peakProminence[i].toPlainText()
                w = self.peakWidth[i].toPlainText()
                if len(h) > 0:
                    self.peaksKwargs[i]['height']     = float(h)
                if len(t) > 0:
                    self.peaksKwargs[i]['threshold']  = float(t)
                if len(d) > 0:
                    self.peaksKwargs[i]['distance']   = float(d) * self.fs
                if len(p) > 0:
                    self.peaksKwargs[i]['prominence'] = float(p)
                if len(w) > 0:
                    self.peaksKwargs[i]['width']      = float(w) * self.fs
                
            else:
                self.detectPeaks[i] = False
                self.peakHeight[i].setVisible(False)
                self.peakThreshold[i].setVisible(False)
                self.peakDistance[i].setVisible(False)
                self.peakProminence[i].setVisible(False)
                self.peakWidth[i].setVisible(False)


    def click_pathSave( self):
        folderName = QtWidgets.QFileDialog.getExistingDirectory( self, 'Select a Folder', )
        self.pathSave_t.setText(folderName)

    def click_rescale( self):
        for g in self.graphicsViews:
            for h in g:
                h.enableAutoRange()

    def click_pause(self):
        if self.displayPaused:
            self.pause_b.setText('Pause')
            self.displayPaused = False
        else:
            self.pause_b.setText('Resume')
            self.displayPaused = True
    
    def click_record(self):
        if self.recording:
            self.recording = False
            self.plotX_data = np.linspace(-1*self.displayTime, 0, num=int(self.displayTime*self.fs)).astype(float)
            self.fileRecording.close()
            self.record_b.setText('Record')
            self.record_b.setStyleSheet("background-color: grey;")
        else:
            ts = datetime.datetime.now()
            filename = '{:04d}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}_nidaq.bin'.format(ts.year,ts.month,ts.day,ts.hour,ts.minute,ts.second)
            self.fileRecording = open( os.path.join( self.pathSave_t.toPlainText(), filename), 'wb')
            headerByteArray = bytearray([ 1, # Version Number
                                         (int(self.fs)>>16)&0xFF, 
                                         (int(self.fs)>>8)&0xFF, 
                                          int(self.fs)&0xFF, 
                                          self.num_channels, 
                                          ts.year>>8, 
                                          ts.year&0xFF, 
                                          ts.month, 
                                          ts.day, 
                                          ts.hour, 
                                          ts.minute, 
                                          ts.second, 
                                          (int(ts.microsecond)>>16)&0xFF, 
                                          (int(ts.microsecond)>>8)&0xFF, 
                                          int(ts.microsecond)&0xFF])
            self.fileRecording.write(headerByteArray)
            self.recording = True
            self.record_b.setText('Stop Recording')
            self.record_b.setStyleSheet("background-color: red;")

    def check_linkAxes( self):
        # Toggles whether the axis are linked
        if self.linkAxes_cb.isChecked():
            for p in self.plotItems:
                p[0].setXLink(self.plotItems[0][0])
        else:
            for p in self.plotItems:
                p[0].setXLink(p[0])

    def closeEvent(self, event):
        print('closing...')

        try:
            self.stopStream = True
            if self.recording:
                self.recording = False
                self.fileRecording.close()
            # Formally close the thread
            if self.thread is not None:
                self.thread.join(timeout=1)
        except Exception as e:
            print('errors on closing...')
            print(e)

        print('Successfully closed.')
        event.accept()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = mainWidget()
    ui.show()
    sys.exit(app.exec())