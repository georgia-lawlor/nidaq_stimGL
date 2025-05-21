from PyQt6 import QtWidgets
import helperFunctions_UI
import nidaq_stim2ch
import numpy as np
import os



class Stim_Voltametry(QtWidgets.QWidget):

    def __init__( self, parent=None):
        super(Stim_Voltametry, self).__init__(parent)
        self.parent_nosave = parent
        self.stim_type = 'Voltametry'
        self.stim_cnt = 0

        self.nidaq = nidaq_stim2ch.NIDAQ_Stim2ch()
        self.buildUI()
        self.setDefaultParams()


    def run(self, stimID=None, **kwargs):
        self.generateStim(**kwargs)
        self.runNidaq(stimID=stimID)


    def generateStim(self, **kwargs):
        self.startVoltage       = kwargs.get( 'startVoltage'       , float(self.ui_startVoltage_t.toPlainText()))    * 1e-3  # mV
        self.stopVoltage        = kwargs.get( 'stopVoltage'        , float(self.ui_stopVoltage_t.toPlainText()))     * 1e-3  # mV
        self.stepVoltage        = kwargs.get( 'stepVoltage'        , float(self.ui_stepVoltage_t.toPlainText()))     * 1e-3  # mV
        self.pulseHeight        = kwargs.get( 'pulseHeight'        , float(self.ui_pulseHeight_t.toPlainText()))     * 1e-3  # mV
        self.pulseWidth         = kwargs.get( 'pulseWidth'         , float(self.ui_pulseWidth_t.toPlainText()))      * 1e-3  # milliseconds
        # self.currentRange       = kwargs.get( 'currentRange'       , float(self.ui_currentRange_t.toPlainText()))    * 1e-6  # uA

        self.time_preTrial      = kwargs.get( 'time_preTrial'      , float(self.ui_time_preTrial_t.toPlainText()))           # sec
        self.time_betweenTrial  = kwargs.get( 'time_betweenTrial'  , float(self.ui_time_betweenTrial_t.toPlainText()))       # sec
        self.time_postTrial     = kwargs.get( 'time_postTrial'     , float(self.ui_time_postTrial_t.toPlainText()))          # sec
        self.repeat             = kwargs.get( 'repeat'             , int(float(self.ui_repeat_t.toPlainText())))             # count
        self.Vref               = kwargs.get( 'Vref'               , 0.9 )                                                   # Volts

        if self.startVoltage > self.stopVoltage: direction = -1
        else: direction = 1
        self.stepVoltage = direction * np.abs(self.stepVoltage) # ensure is in direction of startVoltage to stopVoltage
        self.pulseHeight = direction * np.abs(self.pulseHeight) # ensure is in direction of startVoltage to stopVoltage

        self.time_waveFull = np.floor(np.abs((self.stopVoltage - self.startVoltage) / self.stepVoltage)) * (self.pulseWidth * 2)

        time_all = self.time_preTrial + self.time_postTrial
        time_all += self.time_waveFull + (self.repeat-1)*(self.time_waveFull + self.time_betweenTrial)

        amp = self.pulseHeight * 1.0 # calibration equation

        stimPulse      = [ amp]*int(self.pulseWidth*self.nidaq.fs) + [ -amp]*int(self.pulseWidth*self.nidaq.fs)
        stimOnTemplate = [ 1 ]*int(self.pulseWidth*self.nidaq.fs) + [ -1 ]*int(self.pulseWidth*self.nidaq.fs)
        stimWave       = []
        stimOnWave     = []
        stimStepWave   = []
        for offset in np.arange(self.startVoltage, self.stopVoltage, self.stepVoltage):
            stimWave     += [offset + s for s in stimPulse]
            stimOnWave   += stimOnTemplate
            stimStepWave += [offset] * len(stimPulse)

        stimVec1    = [self.startVoltage]*int(self.time_preTrial*self.nidaq.fs) + stimWave
        stimOnVec   = [self.startVoltage]*int(self.time_preTrial*self.nidaq.fs) + stimOnWave
        stimStepVec = [self.startVoltage]*int(self.time_preTrial*self.nidaq.fs) + stimStepWave
        for i in range(self.repeat-1):
            time_between = int(self.time_betweenTrial * self.nidaq.fs)
            stimVec1    += [self.stopVoltage]*time_between + stimWave
            stimOnVec   += [self.stopVoltage]*time_between + stimOnWave
            stimStepVec += [self.stopVoltage]*time_between + stimStepWave
        stimVec1    += [self.stopVoltage]*int(time_all*self.nidaq.fs-len(stimVec1))
        stimOnVec   += [self.stopVoltage]*int(time_all*self.nidaq.fs-len(stimOnVec))
        stimStepVec += [self.stopVoltage]*int(time_all*self.nidaq.fs-len(stimStepVec))

        self.output_ch1 = np.array(stimVec1) + self.Vref
        self.output_ch2 = self.output_ch1.copy()
        self.stimOn     = np.array(stimOnVec)
        self.stimStep   = np.array(stimStepVec)


    def runNidaq(self, stimID=None, waitForCamera=False):
        print(self.stim_type, ': Stimulating Now!')
        if stimID is None:
            self.stim_cnt += 1
            stimID = self.stim_cnt
        self.nidaq.run( self.output_ch1, self.output_ch2, stimID=stimID, waitForCamera=waitForCamera, channelIn=None)
        print('Done.')
        if self.parent_nosave is not None:
            try:
                self.parent_nosave.finish_experiment(self)
            except:
                print('ERROR: Unable to save Trial')


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
        self.ui_pulseWidth_l                = helperFunctions_UI.makeLabel( 'Puls Width (ms):', self.ui_main_gb)
        self.ui_pulseWidth_t                = helperFunctions_UI.makeTextBox( self.ui_main_gb)

        self.ui_startVoltage_l             = helperFunctions_UI.makeLabel( 'Start Voltage (mV):', self.ui_main_gb)
        self.ui_startVoltage_t             = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_stopVoltage_l              = helperFunctions_UI.makeLabel( 'Stop Voltage (mV):', self.ui_main_gb)
        self.ui_stopVoltage_t              = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_stepVoltage_l              = helperFunctions_UI.makeLabel( 'Step Voltage (mV):', self.ui_main_gb)
        self.ui_stepVoltage_t              = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        self.ui_pulseHeight_l              = helperFunctions_UI.makeLabel( 'Pulse Height (mV):', self.ui_main_gb)
        self.ui_pulseHeight_t              = helperFunctions_UI.makeTextBox( self.ui_main_gb)
        # self.ui_currentRange_l             = helperFunctions_UI.makeLabel( 'Current Range (usec):', self.ui_main_gb)
        # self.ui_currentRange_t             = helperFunctions_UI.makeTextBox( self.ui_main_gb)

        self.ui_run_b = helperFunctions_UI.makeButton( 'Run', self.ui_main_gb)
        self.ui_run_b.clicked.connect( self.run)

        self.ui_main1Layout = helperFunctions_UI.makeHorizontalLayout([  helperFunctions_UI.makeVerticalLayout([ self.ui_time_preTrial_l, 
                                                                                                                self.ui_time_preTrial_t, 
                                                                                                                self.ui_time_betweenTrial_l, 
                                                                                                                self.ui_time_betweenTrial_t,  
                                                                                                                self.ui_time_postTrial_l, 
                                                                                                                self.ui_time_postTrial_t,
                                                                                                                self.ui_repeat_l, 
                                                                                                                self.ui_repeat_t,
                                                                                                                self.ui_pulseWidth_l, 
                                                                                                                self.ui_pulseWidth_t],   contentsMargins=(spacing_sm,0,spacing_sm,0)),
                                                                        helperFunctions_UI.makeVerticalLayout([ self.ui_startVoltage_l,
                                                                                                                self.ui_startVoltage_t,
                                                                                                                self.ui_stopVoltage_l,
                                                                                                                self.ui_stopVoltage_t,
                                                                                                                self.ui_stepVoltage_l,
                                                                                                                self.ui_stepVoltage_t,
                                                                                                                self.ui_pulseHeight_l, 
                                                                                                                self.ui_pulseHeight_t, 
                                                                                                                # self.ui_currentRange_l, 
                                                                                                                # self.ui_currentRange_t,
                                                                                                                self.ui_run_b], contentsMargins=(spacing_sm,0,spacing_sm,0)),
                                                                    ])
        self.ui_mainLayout = helperFunctions_UI.makeVerticalLayout([self.ui_main1Layout])
        self.ui_main1Layout.setContentsMargins(spacing_sm,spacing_sm,spacing_sm,spacing_sm)
        self.ui_mainLayout.setContentsMargins( spacing_sm,spacing_sm,spacing_sm,spacing_sm)
        self.ui_main_gb.setLayout(self.ui_mainLayout)


    def setDefaultParams(self):
        self.ui_time_preTrial_t.setText('1')
        self.ui_time_betweenTrial_t.setText('0.5')
        self.ui_time_postTrial_t.setText('1')
        self.ui_repeat_t.setText('1')
        self.ui_pulseWidth_t.setText('16')
        self.ui_startVoltage_t.setText('0')
        self.ui_stopVoltage_t.setText('400')
        self.ui_stepVoltage_t.setText('4')
        self.ui_pulseHeight_t.setText('40')
        # self.ui_currentRange_t.setText('100')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Stim_Voltametry()
    ui.setLayout(ui.ui_mainLayout)
    ui.show()
    sys.exit(app.exec())