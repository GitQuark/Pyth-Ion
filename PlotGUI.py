# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plotgui.ui'
#
# Created: Wed Nov 05 09:55:50 2014
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_PythIon(object):
    def setupUi(self, PythIon):
        PythIon.setObjectName(_fromUtf8("PythIon"))
        PythIon.resize(1129, 757)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PythIon.sizePolicy().hasHeightForWidth())
        PythIon.setSizePolicy(sizePolicy)
        PythIon.setAutoFillBackground(False)
        self.centralwidget = QtGui.QWidget(PythIon)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 160, 1121, 251))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.verticalLayoutWidget.setFont(font)
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.signalplot = GraphicsLayoutWidget(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.signalplot.setFont(font)
        self.signalplot.setObjectName(_fromUtf8("signalplot"))
        self.verticalLayout.addWidget(self.signalplot)
        self.verticalLayoutWidget_5 = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(569, 23, 551, 131))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.verticalLayoutWidget_5.setFont(font)
        self.verticalLayoutWidget_5.setObjectName(_fromUtf8("verticalLayoutWidget_5"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_6.setMargin(0)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.meandelilabel = QtGui.QLabel(self.verticalLayoutWidget_5)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.meandelilabel.setFont(font)
        self.meandelilabel.setText(_fromUtf8(""))
        self.meandelilabel.setObjectName(_fromUtf8("meandelilabel"))
        self.verticalLayout_6.addWidget(self.meandelilabel)
        self.meandwelllabel = QtGui.QLabel(self.verticalLayoutWidget_5)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.meandwelllabel.setFont(font)
        self.meandwelllabel.setText(_fromUtf8(""))
        self.meandwelllabel.setObjectName(_fromUtf8("meandwelllabel"))
        self.verticalLayout_6.addWidget(self.meandwelllabel)
        self.meandtlabel = QtGui.QLabel(self.verticalLayoutWidget_5)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.meandtlabel.setFont(font)
        self.meandtlabel.setText(_fromUtf8(""))
        self.meandtlabel.setObjectName(_fromUtf8("meandtlabel"))
        self.verticalLayout_6.addWidget(self.meandtlabel)
        self.eventcounterlabel = QtGui.QLabel(self.verticalLayoutWidget_5)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.eventcounterlabel.setFont(font)
        self.eventcounterlabel.setText(_fromUtf8(""))
        self.eventcounterlabel.setObjectName(_fromUtf8("eventcounterlabel"))
        self.verticalLayout_6.addWidget(self.eventcounterlabel)
        self.filelabel = QtGui.QLabel(self.verticalLayoutWidget_5)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filelabel.sizePolicy().hasHeightForWidth())
        self.filelabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.filelabel.setFont(font)
        self.filelabel.setText(_fromUtf8(""))
        self.filelabel.setObjectName(_fromUtf8("filelabel"))
        self.verticalLayout_6.addWidget(self.filelabel)
        self.horizontalLayoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 417, 1121, 280))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.horizontalLayoutWidget_2.setFont(font)
        self.horizontalLayoutWidget_2.setObjectName(_fromUtf8("horizontalLayoutWidget_2"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setMargin(0)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.eventplot = GraphicsLayoutWidget(self.horizontalLayoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.eventplot.sizePolicy().hasHeightForWidth())
        self.eventplot.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Kozuka Gothic Pro L"))
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(3)
        self.eventplot.setFont(font)
        self.eventplot.setStyleSheet(_fromUtf8("font: 25 20pt \"Kozuka Gothic Pro L\";"))
        self.eventplot.setObjectName(_fromUtf8("eventplot"))
        self.horizontalLayout_3.addWidget(self.eventplot)
        self.scatterplot = GraphicsLayoutWidget(self.horizontalLayoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scatterplot.sizePolicy().hasHeightForWidth())
        self.scatterplot.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Kozuka Gothic Pro L"))
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(3)
        self.scatterplot.setFont(font)
        self.scatterplot.setStyleSheet(_fromUtf8("font: 25 20pt \"Kozuka Gothic Pro L\";"))
        self.scatterplot.setObjectName(_fromUtf8("scatterplot"))
        self.horizontalLayout_3.addWidget(self.scatterplot)
        self.groupBox_3 = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(700, 2, 261, 71))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setAutoFillBackground(False)
        self.groupBox_3.setTitle(_fromUtf8(""))
        self.groupBox_3.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_3.setFlat(False)
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.horizontalLayoutWidget_3 = QtGui.QWidget(self.groupBox_3)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 40, 239, 26))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.horizontalLayoutWidget_3.setFont(font)
        self.horizontalLayoutWidget_3.setObjectName(_fromUtf8("horizontalLayoutWidget_3"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_4.setMargin(0)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.previousbutton = QtGui.QPushButton(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.previousbutton.setFont(font)
        self.previousbutton.setObjectName(_fromUtf8("previousbutton"))
        self.horizontalLayout_4.addWidget(self.previousbutton)
        self.gobutton = QtGui.QPushButton(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.gobutton.setFont(font)
        self.gobutton.setObjectName(_fromUtf8("gobutton"))
        self.horizontalLayout_4.addWidget(self.gobutton)
        self.nextbutton = QtGui.QPushButton(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.nextbutton.setFont(font)
        self.nextbutton.setObjectName(_fromUtf8("nextbutton"))
        self.horizontalLayout_4.addWidget(self.nextbutton)
        self.horizontalLayoutWidget_4 = QtGui.QWidget(self.groupBox_3)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(10, 10, 241, 24))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.horizontalLayoutWidget_4.setFont(font)
        self.horizontalLayoutWidget_4.setObjectName(_fromUtf8("horizontalLayoutWidget_4"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_5.setMargin(0)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_4 = QtGui.QLabel(self.horizontalLayoutWidget_4)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_5.addWidget(self.label_4)
        self.eventnumberentry = QtGui.QLineEdit(self.horizontalLayoutWidget_4)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.eventnumberentry.setFont(font)
        self.eventnumberentry.setObjectName(_fromUtf8("eventnumberentry"))
        self.horizontalLayout_5.addWidget(self.eventnumberentry)
        self.groupBox_2 = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(240, 2, 321, 129))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setTitle(_fromUtf8(""))
        self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.layoutWidget = QtGui.QWidget(self.groupBox_2)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 306, 110))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.layoutWidget.setFont(font)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.label = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_5.addWidget(self.label)
        self.label_2 = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_5.addWidget(self.label_2)
        self.label_5 = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout_5.addWidget(self.label_5)
        self.label_3 = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout_5.addWidget(self.label_3)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.LPentry = QtGui.QLineEdit(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LPentry.sizePolicy().hasHeightForWidth())
        self.LPentry.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.LPentry.setFont(font)
        self.LPentry.setObjectName(_fromUtf8("LPentry"))
        self.verticalLayout_4.addWidget(self.LPentry)
        self.outputsamplerateentry = QtGui.QLineEdit(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.outputsamplerateentry.sizePolicy().hasHeightForWidth())
        self.outputsamplerateentry.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.outputsamplerateentry.setFont(font)
        self.outputsamplerateentry.setObjectName(_fromUtf8("outputsamplerateentry"))
        self.verticalLayout_4.addWidget(self.outputsamplerateentry)
        self.eventbufferentry = QtGui.QLineEdit(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.eventbufferentry.sizePolicy().hasHeightForWidth())
        self.eventbufferentry.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.eventbufferentry.setFont(font)
        self.eventbufferentry.setObjectName(_fromUtf8("eventbufferentry"))
        self.verticalLayout_4.addWidget(self.eventbufferentry)
        self.thresholdentry = QtGui.QLineEdit(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thresholdentry.sizePolicy().hasHeightForWidth())
        self.thresholdentry.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.thresholdentry.setFont(font)
        self.thresholdentry.setObjectName(_fromUtf8("thresholdentry"))
        self.verticalLayout_4.addWidget(self.thresholdentry)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(10, 2, 221, 129))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMaximumSize(QtCore.QSize(449, 16777215))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox.setFont(font)
        self.groupBox.setMouseTracking(True)
        self.groupBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.groupBox.setTitle(_fromUtf8(""))
        self.groupBox.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.layoutWidget1 = QtGui.QWidget(self.groupBox)
        self.layoutWidget1.setGeometry(QtCore.QRect(6, 10, 101, 112))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.layoutWidget1.setFont(font)
        self.layoutWidget1.setObjectName(_fromUtf8("layoutWidget1"))
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_8.setMargin(0)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.loadbutton = QtGui.QPushButton(self.layoutWidget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadbutton.sizePolicy().hasHeightForWidth())
        self.loadbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.loadbutton.setFont(font)
        self.loadbutton.setObjectName(_fromUtf8("loadbutton"))
        self.verticalLayout_8.addWidget(self.loadbutton)
        self.nextfilebutton = QtGui.QPushButton(self.layoutWidget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nextfilebutton.sizePolicy().hasHeightForWidth())
        self.nextfilebutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.nextfilebutton.setFont(font)
        self.nextfilebutton.setObjectName(_fromUtf8("nextfilebutton"))
        self.verticalLayout_8.addWidget(self.nextfilebutton)
        self.previousfilebutton = QtGui.QPushButton(self.layoutWidget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previousfilebutton.sizePolicy().hasHeightForWidth())
        self.previousfilebutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.previousfilebutton.setFont(font)
        self.previousfilebutton.setObjectName(_fromUtf8("previousfilebutton"))
        self.verticalLayout_8.addWidget(self.previousfilebutton)
        self.multifilebutton = QtGui.QPushButton(self.layoutWidget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.multifilebutton.sizePolicy().hasHeightForWidth())
        self.multifilebutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.multifilebutton.setFont(font)
        self.multifilebutton.setObjectName(_fromUtf8("multifilebutton"))
        self.verticalLayout_8.addWidget(self.multifilebutton)
        self.layoutWidget2 = QtGui.QWidget(self.groupBox)
        self.layoutWidget2.setGeometry(QtCore.QRect(110, 10, 101, 112))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.layoutWidget2.setFont(font)
        self.layoutWidget2.setObjectName(_fromUtf8("layoutWidget2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_3.setMargin(0)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.cutbutton = QtGui.QPushButton(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cutbutton.sizePolicy().hasHeightForWidth())
        self.cutbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.cutbutton.setFont(font)
        self.cutbutton.setObjectName(_fromUtf8("cutbutton"))
        self.verticalLayout_3.addWidget(self.cutbutton)
        self.baselinebutton = QtGui.QPushButton(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.baselinebutton.sizePolicy().hasHeightForWidth())
        self.baselinebutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.baselinebutton.setFont(font)
        self.baselinebutton.setObjectName(_fromUtf8("baselinebutton"))
        self.verticalLayout_3.addWidget(self.baselinebutton)
        self.analyzebutton = QtGui.QPushButton(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analyzebutton.sizePolicy().hasHeightForWidth())
        self.analyzebutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.analyzebutton.setFont(font)
        self.analyzebutton.setObjectName(_fromUtf8("analyzebutton"))
        self.verticalLayout_3.addWidget(self.analyzebutton)
        self.savebutton = QtGui.QPushButton(self.layoutWidget2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.savebutton.sizePolicy().hasHeightForWidth())
        self.savebutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.savebutton.setFont(font)
        self.savebutton.setObjectName(_fromUtf8("savebutton"))
        self.verticalLayout_3.addWidget(self.savebutton)
        self.eventinfolabel = QtGui.QLabel(self.centralwidget)
        self.eventinfolabel.setGeometry(QtCore.QRect(4, 700, 551, 16))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.eventinfolabel.setFont(font)
        self.eventinfolabel.setText(_fromUtf8(""))
        self.eventinfolabel.setObjectName(_fromUtf8("eventinfolabel"))
        self.groupBox_4 = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_4.setEnabled(True)
        self.groupBox_4.setGeometry(QtCore.QRect(980, 2, 141, 151))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.groupBox_4.setMaximumSize(QtCore.QSize(449, 16777215))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setMouseTracking(True)
        self.groupBox_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.groupBox_4.setTitle(_fromUtf8(""))
        self.groupBox_4.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.verticalLayoutWidget_6 = QtGui.QWidget(self.groupBox_4)
        self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(10, 10, 121, 131))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.verticalLayoutWidget_6.setFont(font)
        self.verticalLayoutWidget_6.setObjectName(_fromUtf8("verticalLayoutWidget_6"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_7.setMargin(0)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.invertbutton = QtGui.QPushButton(self.verticalLayoutWidget_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.invertbutton.sizePolicy().hasHeightForWidth())
        self.invertbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.invertbutton.setFont(font)
        self.invertbutton.setObjectName(_fromUtf8("invertbutton"))
        self.verticalLayout_7.addWidget(self.invertbutton)
        self.concatenatebutton = QtGui.QPushButton(self.verticalLayoutWidget_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.concatenatebutton.sizePolicy().hasHeightForWidth())
        self.concatenatebutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.concatenatebutton.setFont(font)
        self.concatenatebutton.setObjectName(_fromUtf8("concatenatebutton"))
        self.verticalLayout_7.addWidget(self.concatenatebutton)
        self.deleteeventbutton = QtGui.QPushButton(self.verticalLayoutWidget_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.deleteeventbutton.sizePolicy().hasHeightForWidth())
        self.deleteeventbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.deleteeventbutton.setFont(font)
        self.deleteeventbutton.setObjectName(_fromUtf8("deleteeventbutton"))
        self.verticalLayout_7.addWidget(self.deleteeventbutton)
        self.clearscatterbutton = QtGui.QPushButton(self.verticalLayoutWidget_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clearscatterbutton.sizePolicy().hasHeightForWidth())
        self.clearscatterbutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.clearscatterbutton.setFont(font)
        self.clearscatterbutton.setObjectName(_fromUtf8("clearscatterbutton"))
        self.verticalLayout_7.addWidget(self.clearscatterbutton)
        self.savetracebutton = QtGui.QPushButton(self.verticalLayoutWidget_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.savetracebutton.sizePolicy().hasHeightForWidth())
        self.savetracebutton.setSizePolicy(sizePolicy)
        self.savetracebutton.setObjectName(_fromUtf8("savetracebutton"))
        self.verticalLayout_7.addWidget(self.savetracebutton)
        self.saveasmatbutton = QtGui.QPushButton(self.verticalLayoutWidget_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveasmatbutton.sizePolicy().hasHeightForWidth())
        self.saveasmatbutton.setSizePolicy(sizePolicy)
        self.saveasmatbutton.setObjectName(_fromUtf8("saveasmatbutton"))
        self.verticalLayout_7.addWidget(self.saveasmatbutton)
        PythIon.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(PythIon)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1129, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuEdit = QtGui.QMenu(self.menubar)
        self.menuEdit.setObjectName(_fromUtf8("menuEdit"))
        PythIon.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(PythIon)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        PythIon.setStatusBar(self.statusbar)
        self.actionInvert = QtGui.QAction(PythIon)
        self.actionInvert.setObjectName(_fromUtf8("actionInvert"))
        self.actionsetbuffer = QtGui.QAction(PythIon)
        self.actionsetbuffer.setObjectName(_fromUtf8("actionsetbuffer"))
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(PythIon)
        QtCore.QMetaObject.connectSlotsByName(PythIon)

    def retranslateUi(self, PythIon):
        PythIon.setWindowTitle(_translate("PythIon", "PythIon", None))
        self.previousbutton.setText(_translate("PythIon", "Previous", None))
        self.gobutton.setText(_translate("PythIon", "Go", None))
        self.nextbutton.setText(_translate("PythIon", "Next", None))
        self.label_4.setText(_translate("PythIon", "Enter Event Number To View", None))
        self.eventnumberentry.setText(_translate("PythIon", "0", None))
        self.label.setText(_translate("PythIon", "Low Pass Filter (kHz)", None))
        self.label_2.setText(_translate("PythIon", "Output Samplerate (kHz)", None))
        self.label_5.setText(_translate("PythIon", "Event Buffer (samples)", None))
        self.label_3.setText(_translate("PythIon", "Threshold (nA)", None))
        self.LPentry.setText(_translate("PythIon", "100", None))
        self.outputsamplerateentry.setText(_translate("PythIon", "4166.67", None))
        self.eventbufferentry.setText(_translate("PythIon", "1000", None))
        self.thresholdentry.setText(_translate("PythIon", ".8", None))
        self.loadbutton.setText(_translate("PythIon", "LOAD", None))
        self.nextfilebutton.setText(_translate("PythIon", "NEXT", None))
        self.previousfilebutton.setText(_translate("PythIon", "PREVIOUS", None))
        self.multifilebutton.setText(_translate("PythIon", "MULTIFILE", None))
        self.cutbutton.setText(_translate("PythIon", "CUT", None))
        self.baselinebutton.setText(_translate("PythIon", "BASELINE", None))
        self.analyzebutton.setText(_translate("PythIon", "ANALYZE", None))
        self.savebutton.setText(_translate("PythIon", "SAVE", None))
        self.invertbutton.setText(_translate("PythIon", "INVERT", None))
        self.concatenatebutton.setText(_translate("PythIon", "CONCATENATE", None))
        self.deleteeventbutton.setText(_translate("PythIon", "DELETE EVENT", None))
        self.clearscatterbutton.setText(_translate("PythIon", "CLEAR SCATTER", None))
        self.savetracebutton.setText(_translate("PythIon", "SAVE TRACE", None))
        self.saveasmatbutton.setText(_translate("PythIon", "SAVE AS .MAT", None))
        self.menuFile.setTitle(_translate("PythIon", "File", None))
        self.menuEdit.setTitle(_translate("PythIon", "Edit", None))
        self.actionInvert.setText(_translate("PythIon", "Invert data", None))
        self.actionsetbuffer.setText(_translate("PythIon", "Set Event Buffer Size", None))

from pyqtgraph import GraphicsLayoutWidget
