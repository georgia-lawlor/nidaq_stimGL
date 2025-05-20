from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget


'''
In previous implementations, all labels looked like this:
    >>> self.label_13 = QtWidgets.QLabel(self.BaseStationParameters_gb)
    >>> font = QtGui.QFont()
    >>> font.setPointSize(10)
    >>> self.label_13.setFont(font)
    >>> self.label_13.setText("Charge Freq")
    >>> self.label_13.setObjectName("label_13")

This function shrinks a lot of lines into one call, shortening files drastically
'''

dataView_fontsize = 10
dataView_heightMultiplier = 4
dataView_UniversalFont = QtGui.QFont()
dataView_UniversalFont.setPointSize( dataView_fontsize)

def makeLabel( text, parent):
    label_obj = QtWidgets.QLabel( parent)
    label_obj.setFont(dataView_UniversalFont)
    label_obj.setText(text)

    return label_obj


def makeUnit( text, parent):
    label_obj = makeLabel( text, parent)

    label_obj.setFixedHeight(dataView_fontsize*dataView_heightMultiplier)
    label_obj.setFixedWidth( dataView_fontsize*len(text))

    return label_obj


def makeGroupBox( name):
    groupboxObj = QtWidgets.QGroupBox(name)
    groupboxObj.setFont(dataView_UniversalFont)
    groupboxObj.setTitle(name)
    groupboxObj.setFlat(False)

    return groupboxObj


def makeButton( text, parent, enabled=None):
    buttonObj = QtWidgets.QPushButton( parent)
    buttonObj.setFont(dataView_UniversalFont)
    buttonObj.setText(text)

    if enabled is not None:
        buttonObj.setEnabled(enabled)

    return buttonObj


def makeCheckBox( text, parent, checked=None):
    checkboxObj = QtWidgets.QCheckBox(parent)
    checkboxObj.setFont(dataView_UniversalFont)
    checkboxObj.setText(text)
    checkboxObj.setFixedHeight(dataView_fontsize*dataView_heightMultiplier)

    if checked is not None:
        checkboxObj.setChecked(checked)

    return checkboxObj


def makeComboBox( parent, items=[]):
    comboboxObj = QtWidgets.QComboBox(parent)
    comboboxObj.setFont(dataView_UniversalFont)

    for item in items:
        comboboxObj.addItem(item)

    return comboboxObj


def makeTextBox( parent, text=None):
    textboxObj = QtWidgets.QTextEdit(parent)
    textboxObj.setFont(dataView_UniversalFont)
    try: # PyQt6
        textboxObj.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        textboxObj.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        textboxObj.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    except: # PyQt5
        textboxObj.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        textboxObj.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        textboxObj.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    textboxObj.setTabChangesFocus(True)
    textboxObj.setFixedHeight(dataView_fontsize*dataView_heightMultiplier)

    if text is not None:
        textboxObj.setText(text)

    return textboxObj


def makeSpinBox( parent, minimum, maximum, decimals, step):
    spinboxObj = QtWidgets.QDoubleSpinBox(parent)
    spinboxObj.setFont(dataView_UniversalFont)
    spinboxObj.setMinimum(minimum)
    spinboxObj.setMaximum(maximum)
    spinboxObj.setDecimals(decimals)
    spinboxObj.setSingleStep(step)

    return spinboxObj


def makeSliderBar( parent, minimum, maximum, interval, vertical=False, initialValue=None):
    if vertical:
        try: # PyQt6
            sliderObj = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical, parent)
        except: # PyQt5
            sliderObj = QtWidgets.QSlider(QtCore.Qt.Vertical, parent)
    else:
        try: # PyQt6
            sliderObj = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, parent)
        except: # PyQt5
            sliderObj = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent)
    sliderObj.setMinimum(minimum)
    sliderObj.setMaximum(maximum)
    sliderObj.setTickInterval(interval)
    if initialValue is not None:
        sliderObj.setValue(initialValue)

    return sliderObj


def makeHorizontalLayout( listOfWidgets, contentsMargins=(0,0,0,0), spacing=5):
    # Content Margin is (Left, Top, Right, Bottom)
    horizontalLayout = QtWidgets.QHBoxLayout()
    for widget in listOfWidgets:
        try:
            horizontalLayout.addWidget( widget, 4)
        except:
            try:
                horizontalLayout.addLayout( widget, 4)
            except:
                print('Error adding ', widget, '  - Incorrect Type.')
    
    horizontalLayout.setContentsMargins(*contentsMargins)
    horizontalLayout.setSpacing(spacing)

    return horizontalLayout

def makeVerticalLayout( listOfWidgets, contentsMargins=(0,0,0,0), spacing=0):
    # Content Margin is (Left, Top, Right, Bottom)
    verticalLayout = QtWidgets.QVBoxLayout()
    for widget in listOfWidgets:
        try:
            verticalLayout.addWidget( widget, 4)
        except:
            try:
                verticalLayout.addLayout( widget, 4)
            except:
                print('Error adding ', widget, '  - Incorrect Type.')

    verticalLayout.setContentsMargins(*contentsMargins)
    verticalLayout.setSpacing(spacing)

    return verticalLayout


def makeGraphicsView():
    graphicsViewObj = PlotWidget()
    graphicsViewObj.setFont(dataView_UniversalFont)
    try: # PyQt6
        graphicsViewObj.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus) # PyQt6
        graphicsViewObj.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
    except: # PyQt5
        graphicsViewObj.setFocusPolicy(QtCore.Qt.ClickFocus) # PyQt5

    return graphicsViewObj