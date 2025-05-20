from PyQt6 import QtWidgets
import helperFunctions_UI
import nidaq_stim2ch
import numpy as np



class Stim_Square(QtWidgets.QWidget):

    def __init__( self, parent=None):
        super(Stim_Square, self).__init__(parent)
        self.parent_nosave = parent
        self.stim_type = 'Square'
        self.stim_cnt = 0

        self.nidaq = nidaq_stim2ch.NIDAQ_Stim2ch()
        self.buildUI()
        self.setDefaultParams()


    def run(self, stimID=None, **kwargs):
        self.generateStim(**kwargs)
        self.runNidaq(stimID=stimID)


    def generateStim(self, **kwargs):
        self.amplitude          = kwargs.get( 'amplitude'          , float(self.ui_amplitude_t.toPlainText()))       * 1e-6  # uA
        self.pulse_width        = kwargs.get( 'pulse_width'        , float(self.ui_pulseWidth_t.toPlainText()))      * 1e-6  # microsec
        self.time_interpulse    = kwargs.get( 'time_interpulse'    , float(self.ui_time_interpulse_t.toPlainText())) * 1e-6  # microsec
        self.frequency          = kwargs.get( 'frequency'          , float(self.ui_frequency_t.toPlainText()))               # Hz
        self.repeat             = kwargs.get( 'repeat'             , int(float(self.ui_repeat_t.toPlainText())))             # count
        self.time_preTrial      = kwargs.get( 'time_preTrial'      , float(self.ui_time_preTrial_t.toPlainText()))           # sec
        self.time_betweenTrial  = kwargs.get( 'time_betweenTrial'  , float(self.ui_time_betweenTrial_t.toPlainText()))       # sec
        self.time_postTrial     = kwargs.get( 'time_postTrial'     , float(self.ui_time_postTrial_t.toPlainText()))          # sec
        self.Vref               = kwargs.get( 'Vref'               , 0 )                                                     # Volts
        if self.ui_usePulseCount_cb.isChecked():
            self.pulse_amplitudeFull = kwargs.get( 'pulse_amplitudeFull', int(float(self.ui_numOrTime_amplitudeFull_t.toPlainText())))    # count
            self.time_amplitudeFull  = -1
        else:
            self.pulse_amplitudeFull = -1
            self.time_amplitudeFull  = kwargs.get(  'time_amplitudeFull', float(self.ui_numOrTime_amplitudeFull_t.toPlainText()))  # sec
        if False: # self.ui_addJitter_cb.isChecked():
            self.time_jitterRange = kwargs.get( 'time_jitterRange'   , 0.5 * (2*self.time_betweenTrial))      # sec
        else:
            self.time_jitterRange = 0

        if self.time_amplitudeFull > 0:
            time_all = self.time_preTrial + self.time_postTrial + (self.time_jitterRange * self.repeat)
            time_all += self.time_amplitudeFull + (self.repeat-1)*(self.time_amplitudeFull + self.time_betweenTrial)
        elif self.pulse_amplitudeFull > 0:
            time_ampFull = (1/self.frequency) * self.pulse_amplitudeFull
            time_all = self.time_preTrial + self.time_postTrial + (self.time_jitterRange * self.repeat)
            time_all += time_ampFull + (self.repeat-1)*(time_ampFull + self.time_betweenTrial)

        amp = self.amplitude * 100
        amp = amp * 6.2 * 0.96 # calibration equation

        stimWave       = [-amp]*int(self.pulse_width*self.nidaq.fs) + [0]*int(self.time_interpulse*self.nidaq.fs) + [amp]*int(self.pulse_width*self.nidaq.fs)
        stimOnTemplate = [  1 ]*len(stimWave)
        stimWave       += [0]*int((self.nidaq.fs/self.frequency)-len(stimWave))
        stimOnTemplate += [0]*int((self.nidaq.fs/self.frequency)-len(stimOnTemplate))
        if self.ui_usePulseCount_cb.isChecked():
            stimWave       = stimWave * self.pulse_amplitudeFull
            stimOnTemplate = stimOnTemplate * self.pulse_amplitudeFull
        else:
            numWave = int((self.time_amplitudeFull * self.nidaq.fs) // len(stimWave))
            stimWave       = stimWave * numWave
            stimOnTemplate = stimOnTemplate * numWave

        stimVec1 = [0]*int(self.time_preTrial*self.nidaq.fs) + stimWave
        stimOn   = [0]*int(self.time_preTrial*self.nidaq.fs) + stimOnTemplate
        for i in range(self.repeat-1):
            time_between = int((self.time_betweenTrial + np.random.uniform(-1*self.time_jitterRange,self.time_jitterRange)) * self.nidaq.fs)
            print(time_between/self.nidaq.fs)
            stimVec1 += [0]*time_between + stimWave
            stimOn   += [0]*time_between + stimOnTemplate
        stimVec1 += [0]*int(time_all*self.nidaq.fs-len(stimVec1))
        stimOn   += [0]*int(time_all*self.nidaq.fs-len(stimOn))

        self.output_ch1 = np.array(stimVec1) + self.Vref
        self.output_ch2 = stimVec1.copy()
        self.stimOn     = np.array(stimOn)


    def runNidaq(self, stimID=None, waitForCamera=False):
        print(self.stim_type, ': Stimulating at ', self.amplitude*1e6, 'uA, ', self.frequency, 'Hz')
        if stimID is None:
            self.stim_cnt += 1
            stimID = self.stim_cnt
        self.nidaq.run( self.output_ch1, self.output_ch2, stimID=stimID, waitForCamera=waitForCamera, channelIn=None)
        print('Done.')
        if self.parent_nosave is not None:
            # try:
                self.parent_nosave.finish_experiment(self)
            # except:
            #     print('ERROR: Unable to save Trial')


    def get_matVars(self):
        matVars = self.nidaq.get_matVars()

        class_vars = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not key.startswith('ui') and not key.startswith('nidaq') and not key.endswith('_nosave') and not callable(key)}
        matVars.update( class_vars )

        return matVars


    def plot_stimAvg(self):
        data = np.concatenate(( self.nidaq.recording_ch1, self.nidaq.recording_ch2, self.nidaq.recording_ch3 )).reshape(3,-1)
        index_stim = np.argwhere( np.diff(self.stimOn) == 1)

        data_stim = []
        i_before = int( self.nidaq.fs / self.frequency)
        i_after  = int( self.nidaq.fs / self.frequency)
        for i in index_stim:
            if (i-i_before > 0) and (i+i_after < data.shape[-1]):
                data_stim.append( data[: , int(i-i_before) : int(i+i_after)])
        data_avg = np.mean( data_stim, axis=0)
        time_avg = np.linspace(-1/self.frequency, 1/self.frequency, data_avg.shape[-1])

        self.parent_nosave.pNW.setData( x=time_avg, y=data_avg[0,:])
        self.parent_nosave.pSW.setData( x=time_avg, y=data_avg[1,:])
        self.parent_nosave.pNE.setData( x=time_avg, y=data_avg[2,:])
        self.parent_nosave.pSE.setData( x=time_avg, y=data_avg[3,:])


    def click_numOrTimePulses(self):
        if self.ui_usePulseCount_cb.isChecked():
            self.ui_numOrTime_amplitudeFull_l.setText('Number Pulses (count):')
        else:
            self.ui_numOrTime_amplitudeFull_l.setText('Time Pulseing (sec):')


    def buildUI(self):
        spacing_sm = 5
        spacing_lg = 10
        
        self.ui_main_gb                    = helperFunctions_UI.makeGroupBox('Stimulation Params')
        self.ui_time_preTrial_l            = helperFunctions_UI.makeLabel( 'Pre-Trial Time (sec):', self.ui_main_gb)
        self.ui_time_preTrial_t            = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_time_betweenTrial_l        = helperFunctions_UI.makeLabel( 'Between Trial Time (sec):', self.ui_main_gb)
        self.ui_time_betweenTrial_t        = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_time_postTrial_l           = helperFunctions_UI.makeLabel( 'Post-Trial Time (sec):', self.ui_main_gb)
        self.ui_time_postTrial_t           = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_repeat_l                   = helperFunctions_UI.makeLabel( 'Number Trials (count):', self.ui_main_gb)
        self.ui_repeat_t                   = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_numOrTime_amplitudeFull_l  = helperFunctions_UI.makeLabel( 'Number Pulses (count):', self.ui_main_gb)
        self.ui_numOrTime_amplitudeFull_t  = helperFunctions_UI.makeTextBox( self.ui_main_gb)

        self.ui_amplitude_l                = helperFunctions_UI.makeLabel( 'Amplitude (uA):', self.ui_main_gb)
        self.ui_amplitude_t                = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_frequency_l                = helperFunctions_UI.makeLabel( 'Pulse Frequency (Hz):', self.ui_main_gb)
        self.ui_frequency_t                = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_pulseWidth_l               = helperFunctions_UI.makeLabel( 'Pulse Width (usec):', self.ui_main_gb)
        self.ui_pulseWidth_t               = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_time_interpulse_l          = helperFunctions_UI.makeLabel( 'Interpulse Time (usec):', self.ui_main_gb)
        self.ui_time_interpulse_t          = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_usePulseCount_cb           = helperFunctions_UI.makeCheckBox('Use Pulse Count', self.ui_main_gb)
        self.ui_usePulseCount_cb.stateChanged.connect( self.click_numOrTimePulses)

        self.ui_run_b = helperFunctions_UI.makeButton( 'Run', self.ui_main_gb)
        self.ui_run_b.clicked.connect( self.run)

        self.ui_checkboxes  = helperFunctions_UI.makeHorizontalLayout([self.ui_usePulseCount_cb])
        self.ui_main1Layout = helperFunctions_UI.makeHorizontalLayout([  helperFunctions_UI.makeVerticalLayout([ self.ui_time_preTrial_l, 
                                                                                                                self.ui_time_preTrial_t, 
                                                                                                                self.ui_time_betweenTrial_l, 
                                                                                                                self.ui_time_betweenTrial_t,  
                                                                                                                self.ui_time_postTrial_l, 
                                                                                                                self.ui_time_postTrial_t,
                                                                                                                self.ui_repeat_l, 
                                                                                                                self.ui_repeat_t,
                                                                                                                self.ui_numOrTime_amplitudeFull_l, 
                                                                                                                self.ui_numOrTime_amplitudeFull_t],   contentsMargins=(spacing_sm,0,spacing_sm,0)),
                                                                        helperFunctions_UI.makeVerticalLayout([ self.ui_amplitude_l, 
                                                                                                                self.ui_amplitude_t, 
                                                                                                                self.ui_frequency_l, 
                                                                                                                self.ui_frequency_t,
                                                                                                                self.ui_pulseWidth_l,
                                                                                                                self.ui_pulseWidth_t,
                                                                                                                self.ui_time_interpulse_l,
                                                                                                                self.ui_time_interpulse_t,
                                                                                                                self.ui_checkboxes,
                                                                                                                self.ui_run_b], contentsMargins=(spacing_sm,0,spacing_sm,0)),
                                                                    ])
        self.ui_mainLayout = helperFunctions_UI.makeVerticalLayout([self.ui_main1Layout])
        self.ui_main1Layout.setContentsMargins(spacing_sm,spacing_sm,spacing_sm,spacing_sm)
        self.ui_mainLayout.setContentsMargins( spacing_sm,spacing_sm,spacing_sm,spacing_sm)
        self.ui_main_gb.setLayout(self.ui_mainLayout)


    def setDefaultParams(self):
        self.ui_time_preTrial_t.setText('0.7')
        self.ui_time_betweenTrial_t.setText('0.25')
        self.ui_time_postTrial_t.setText('2')
        self.ui_repeat_t.setText('1')
        self.ui_numOrTime_amplitudeFull_t.setText('50')
        self.ui_amplitude_t.setText('500')
        self.ui_frequency_t.setText('20')
        self.ui_pulseWidth_t.setText('500')
        self.ui_time_interpulse_t.setText('200')
        self.ui_usePulseCount_cb.setChecked(True)
        self.click_numOrTimePulses()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Stim_Square()
    ui.setLayout(ui.ui_mainLayout)
    ui.show()
    sys.exit(app.exec())