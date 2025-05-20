from PyQt6.QtCore import pyqtSignal, pyqtSlot
from scipy.io import savemat, loadmat
from datetime import datetime
from PyQt6 import QtWidgets
import helperFunctions_UI
import time
import sys
import os

import stim_square, stim_voltametry


class mainWidget(QtWidgets.QWidget):
    incrementFileTag = pyqtSignal()

    def __init__(self, parent=None):
        super(mainWidget, self).__init__(parent)
        self.setWindowTitle("Livermore Square Wave Voltametry")
        self.pathSave_default = "C://Users//georg//OneDrive - Johns Hopkins//Epileptogenesis//Scripts//Livermore"
        self.stimulation = None
        self.closing = False
        self.thread = None

        if not os.path.isdir(self.pathSave_default):
            os.mkdir(self.pathSave_default)

        self.incrementFileTag.connect(self.incrementFileTagFcn)
        
        self.buildUI()
        self.set_defaultParams()
        self.click_rescale()

    def save( self, matVars, newTimestamp=False, path=None):
        if path is None:
            path = self.pathSave_t.toPlainText()
        
        matVars.update( self.get_matVars())
        
        if newTimestamp:
            filename = '_'.join(( matVars['filePrefix'], matVars['fileTag'],                                datetime.now().strftime("%H.%M.%S"), str(self.stimulation.stim_type)+'.mat'))
        else:
            filename = '_'.join(( matVars['filePrefix'], matVars['fileTag'], datetime.fromtimestamp(matVars['time_start']).strftime("%H.%M.%S"), str(self.stimulation.stim_type)+'.mat'))
        matVars['origFileName'] = filename
        print('Saving...')

        try:
            savemat( os.path.join(path, filename), matVars)
        except:
            print('ERROR accessing save destination.')
        self.incrementFileTag.emit()


    def incrementFileTagFcn(self):
        tag = self.fileTag_t.toPlainText()
        if len(tag) == 2:
            if tag[-1].lower() == 'z':
                tag = chr(ord(tag[-2])+1) + 'A'
            else:
                tag = tag[-2] + chr(ord(tag[-1])+1)
            self.fileTag_t.setText(tag)


    def finish_experiment(self, trial):
        self.save(matVars=trial.get_matVars())
        self.update_trialPlots(trial)
        self.flag_finishExperiment = True


    def update_trialPlots(self, trial):
        self.pNW.setData(   x=trial.nidaq.time, y=trial.nidaq.output_ch1)
        self.pSW.setData(   x=trial.nidaq.time, y=trial.nidaq.recording_ch1)
        self.pNE.setData(   x=trial.nidaq.time, y=trial.nidaq.recording_ch2)
        self.pSE.setData(   x=trial.nidaq.time, y=trial.nidaq.recording_ch3)
        self.click_rescale()
        print('Updated Plots.')


    def click_pathSave( self):
        folderName = QtWidgets.QFileDialog.getExistingDirectory( self, 'Select a Folder', )
        self.pathSave_t.setText(folderName)


    def click_rescale( self):
        # self.pw_NW.setXRange(0, self.latestTrial.displayTime, padding=0.05)
        # self.pw_NW.setYRange(0, 1, padding=0.05)
        self.graphicsView_NW.enableAutoRange()

        # self.pw_NE.setXRange(0, self.latestTrial.displayTime, padding=0.05)
        # self.pw_NE.setYRange(0, 1, padding=0.05)
        self.graphicsView_NE.enableAutoRange()

        # self.pw_SW.setXRange(0, self.latestTrial.displayTime, padding=0.05)
        # self.pw_SW.setYRange(0, 1, padding=0.05)
        self.graphicsView_SW.enableAutoRange()

        # self.pw_SE.setXRange(0, self.latestTrial.displayTime, padding=0.05)
        # self.pw_SE.setYRange(0, 1, padding=0.05)
        self.graphicsView_SE.enableAutoRange()


    def check_linkAxes( self):
        # Toggles whether the axis are linked
        if self.linkAxes_cb.isChecked():
            self.pw_NW.setXLink(self.pw_NW)
            self.pw_NE.setXLink(self.pw_NW)
            self.pw_SW.setXLink(self.pw_NW)
            self.pw_SE.setXLink(self.pw_NW)
        else:
            self.pw_NW.setXLink(self.pw_NW)
            self.pw_NE.setXLink(self.pw_NE)
            self.pw_SW.setXLink(self.pw_SW)
            self.pw_SE.setXLink(self.pw_SE)


    def get_matVars(self):
        matVars = {}
        matVars['filePrefix'] = self.filePrefix_t.toPlainText()
        matVars['electrode']  = self.electrode_t.toPlainText()
        matVars['expType']    = self.experimentType_t.toPlainText()
        matVars['fileTag']    = self.fileTag_t.toPlainText()
        
        return matVars


    def buildUI(self):
        spacing_sm = 5
        spacing_lg = 10

        # Metadata Layout
        self.metadata_gb           = helperFunctions_UI.makeGroupBox('Metadata')
        self.experimentClass_dd    = helperFunctions_UI.makeComboBox(self.metadata_gb ,items=['Voltametry','Square'])
        self.pathSave_l            = helperFunctions_UI.makeLabel('Save Path', self.metadata_gb )
        self.pathSave_t            = helperFunctions_UI.makeTextBox(self.metadata_gb )
        self.pathSave_b            = helperFunctions_UI.makeButton('Browse', self.metadata_gb )
        self.experimentClass_dd.setFixedWidth(100)
        self.pathSave_l.setFixedWidth(100)
        self.pathSave_b.setFixedWidth(100)
        self.pathSave_t.setText(self.pathSave_default)
        self.pathSave_b.clicked.connect(self.click_pathSave)
        self.experiment_layout = helperFunctions_UI.makeHorizontalLayout([ self.experimentClass_dd, self.pathSave_l, self.pathSave_t, self.pathSave_b])

        self.electrode_l           = helperFunctions_UI.makeLabel( 'Electrode:', self.metadata_gb)
        self.electrode_t           = helperFunctions_UI.makeTextBox( self.metadata_gb)
        self.experimentType_l      = helperFunctions_UI.makeLabel( 'Experiment Type:', self.metadata_gb)
        self.experimentType_t      = helperFunctions_UI.makeTextBox( self.metadata_gb)
        self.filePrefix_l          = helperFunctions_UI.makeLabel( 'File Prefix:', self.metadata_gb)
        self.filePrefix_t          = helperFunctions_UI.makeTextBox( self.metadata_gb)
        self.fileTag_l             = helperFunctions_UI.makeLabel( 'File Tag:', self.metadata_gb)
        self.fileTag_t             = helperFunctions_UI.makeTextBox( self.metadata_gb)
        self.notes_layout       = helperFunctions_UI.makeHorizontalLayout([ helperFunctions_UI.makeVerticalLayout([  self.electrode_l,  self.electrode_t, self.experimentType_l, self.experimentType_t], contentsMargins=(spacing_sm,0,spacing_sm,0)),
                                                                            helperFunctions_UI.makeVerticalLayout([ self.filePrefix_l, self.filePrefix_t,        self.fileTag_l,        self.fileTag_t], contentsMargins=(spacing_sm,0,spacing_sm,0)),
                                                                           ])
        
        self.metadata_layout = helperFunctions_UI.makeVerticalLayout([self.notes_layout,self.experiment_layout])
        self.metadata_layout.setContentsMargins(spacing_sm,spacing_sm,spacing_sm,spacing_sm)
        self.metadata_gb.setLayout(self.metadata_layout)

        self.rescale_b = helperFunctions_UI.makeButton('Rescale Plots', self)
        self.rescale_b.clicked.connect(self.click_rescale)
        self.linkAxes_cb = helperFunctions_UI.makeCheckBox('Link Axes', self)
        self.linkAxes_cb.stateChanged.connect(  self.check_linkAxes)
        self.control_layout = helperFunctions_UI.makeHorizontalLayout([ self.rescale_b, self.linkAxes_cb])

        # Create the graphs
        self.graphLayout = QtWidgets.QGridLayout()
        self.graphicsView_NW  = helperFunctions_UI.makeGraphicsView() # Graphics View 2 (Top Left)
        self.graphLayout.addWidget(self.graphicsView_NW, 0,0)
        self.graphicsView_NE  = helperFunctions_UI.makeGraphicsView() # Graphics View 1 (Top Right)
        self.graphLayout.addWidget(self.graphicsView_NE, 0,1)
        self.graphicsView_SW  = helperFunctions_UI.makeGraphicsView() # Graphics View 1 (Bottom Right)
        self.graphLayout.addWidget(self.graphicsView_SW, 1,0)
        self.graphicsView_SE  = helperFunctions_UI.makeGraphicsView() # Graphics View 2 (Bottom Left)
        self.graphLayout.addWidget(self.graphicsView_SE, 1,1)

        # Set up the graph item
        self.pw_NW = self.graphicsView_NW.getPlotItem()
        self.pw_NW.setTitle(title='Stimulation Output')
        self.pw_NW.setLabel('bottom',text='Time',units='s')
        self.pw_NW.showLabel('bottom',show=True)
        self.pw_NW.setLabel('left',text='Voltage', units='V')
        self.pw_NW.showLabel('left',show=True)
        self.pw_NW.showGrid(x=True,y=True)
        
        self.pw_NE = self.graphicsView_NE.getPlotItem()
        self.pw_NE.setTitle(title='Recording Ch 1')
        self.pw_NE.setLabel('bottom',text='Time',units='s')
        self.pw_NE.showLabel('bottom',show=True)
        self.pw_NE.setLabel('left',text='Voltage', units='V')
        self.pw_NE.showLabel('left',show=True)
        self.pw_NE.showGrid(x=True,y=True)

        self.pw_SW = self.graphicsView_SW.getPlotItem()
        self.pw_SW.setTitle(title='Recording Ch 2')
        self.pw_SW.setLabel('bottom',text='Time',units='s')
        self.pw_SW.showLabel('bottom',show=True)
        self.pw_SW.setLabel('left',text='Voltage', units='V')
        self.pw_SW.showLabel('left',show=True)
        self.pw_SW.showGrid(x=True,y=True)

        self.pw_SE = self.graphicsView_SE.getPlotItem()
        self.pw_SE.setTitle(title='Recording Ch 3')
        self.pw_SE.setLabel('bottom',text='Time',units='s')
        self.pw_SE.showLabel('bottom',show=True)
        self.pw_SE.setLabel('left',text='Voltage', units='V')
        self.pw_SE.showLabel('left',show=True)
        self.pw_SE.showGrid(x=True,y=True)

        # Extract the plotter object directly for faster data updates
        self.pNW = self.pw_NW.plot()
        self.pNW.setPen((200,200,0))
        self.pNE = self.pw_NE.plot()
        self.pNE.setPen((200,200,0))
        self.pSW = self.pw_SW.plot()
        self.pSW.setPen((200,200,0))
        self.pSE = self.pw_SE.plot()
        self.pSE.setPen((200,200,0))

        # Create the stimulation GUIs
        self.stimulation_square = stim_square.Stim_Square(self)
        self.stimulation_voltametry = stim_voltametry.Stim_Voltametry(self)

        # Construct the final layout
        self.mainLayout = helperFunctions_UI.makeVerticalLayout([ self.metadata_gb, self.stimulation_square.ui_main_gb, self.stimulation_voltametry.ui_main_gb, self.control_layout])
        self.mainLayout.addLayout(self.graphLayout, 99)
        self.setLayout(self.mainLayout)
        self.experimentClass_dd.currentIndexChanged.connect( self.refreshUI)
        self.refreshUI()


    def refreshUI(self):
        spacing_lg = 10

        if self.experimentClass_dd.currentText() == 'Voltametry':
            self.stimulation = self.stimulation_voltametry
            self.stimulation_square.ui_main_gb.setVisible(False)
            self.stimulation_voltametry.ui_main_gb.setVisible(True)
        if self.experimentClass_dd.currentText() == 'Square':
            self.stimulation = self.stimulation_square
            self.stimulation_voltametry.ui_main_gb.setVisible(False)
            self.stimulation_square.ui_main_gb.setVisible(True)
        print('<'+self.experimentClass_dd.currentText()+'>')
            

    def set_defaultParams(self):
        self.filePrefix_t.setText('JORJA000')
        self.electrode_t.setText('custom')
        self.experimentType_t.setText('test')
        self.fileTag_t.setText('AA')


    def closeEvent(self, event):
        print('closing...')

        try:
            self.closing = True
            # Wait for thread to exit gracefully
            startTime = time.time()
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
