# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'batchinfowidget.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_batchinfodialog(object):
    def setupUi(self, batchinfodialog):
        batchinfodialog.setObjectName("batch_info_dialog")
        batchinfodialog.resize(375, 409)
        self.gridLayout_2 = QtWidgets.QGridLayout(batchinfodialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox = QtWidgets.QGroupBox(batchinfodialog)
        self.groupBox.setAutoFillBackground(False)
        self.groupBox.setStyleSheet("QGroupBox { \n"
"     border: 2px solid gray; \n"
"     border-radius: 3px; \n"
" } ")
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.minfracbox = QtWidgets.QLineEdit(self.groupBox)
        self.minfracbox.setAlignment(QtCore.Qt.AlignCenter)
        self.minfracbox.setObjectName("minfracbox")
        self.gridLayout.addWidget(self.minfracbox, 2, 1, 1, 1)
        self.sampratebox = QtWidgets.QLineEdit(self.groupBox)
        self.sampratebox.setAlignment(QtCore.Qt.AlignCenter)
        self.sampratebox.setObjectName("sampratebox")
        self.gridLayout.addWidget(self.sampratebox, 0, 1, 1, 1)
        self.LPfilterbox = QtWidgets.QLineEdit(self.groupBox)
        self.LPfilterbox.setAlignment(QtCore.Qt.AlignCenter)
        self.LPfilterbox.setObjectName("LPfilterbox")
        self.gridLayout.addWidget(self.LPfilterbox, 1, 1, 1, 1)
        self.mindwellbox = QtWidgets.QLineEdit(self.groupBox)
        self.mindwellbox.setAlignment(QtCore.Qt.AlignCenter)
        self.mindwellbox.setObjectName("mindwellbox")
        self.gridLayout.addWidget(self.mindwellbox, 3, 1, 1, 1)
        self.cusumstepentry = QtWidgets.QLineEdit(self.groupBox)
        self.cusumstepentry.setAlignment(QtCore.Qt.AlignCenter)
        self.cusumstepentry.setObjectName("cusumstepentry")
        self.gridLayout.addWidget(self.cusumstepentry, 4, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.cusumthreshentry = QtWidgets.QLineEdit(self.groupBox)
        self.cusumthreshentry.setAlignment(QtCore.Qt.AlignCenter)
        self.cusumthreshentry.setObjectName("cusumthreshentry")
        self.gridLayout.addWidget(self.cusumthreshentry, 5, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 6, 0, 1, 1)
        self.minleveltbox = QtWidgets.QLineEdit(self.groupBox)
        self.minleveltbox.setAlignment(QtCore.Qt.AlignCenter)
        self.minleveltbox.setObjectName("minleveltbox")
        self.gridLayout.addWidget(self.minleveltbox, 6, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 8, 0, 1, 1)
        self.maxLevelsBox = QtWidgets.QLineEdit(self.groupBox)
        self.maxLevelsBox.setAlignment(QtCore.Qt.AlignCenter)
        self.maxLevelsBox.setObjectName("maxLevelsBox")
        self.gridLayout.addWidget(self.maxLevelsBox, 8, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.invertCheckBox = QtWidgets.QCheckBox(batchinfodialog)
        self.invertCheckBox.setObjectName("invertCheckBox")
        self.verticalLayout.addWidget(self.invertCheckBox)
        self.selfCorrectCheckBox = QtWidgets.QCheckBox(batchinfodialog)
        self.selfCorrectCheckBox.setObjectName("selfCorrectCheckBox")
        self.verticalLayout.addWidget(self.selfCorrectCheckBox)
        self.gridLayout_2.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.okbutton = QtWidgets.QPushButton(batchinfodialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.okbutton.sizePolicy().hasHeightForWidth())
        self.okbutton.setSizePolicy(sizePolicy)
        self.okbutton.setMinimumSize(QtCore.QSize(0, 0))
        self.okbutton.setMaximumSize(QtCore.QSize(200, 50))
        self.okbutton.setObjectName("okbutton")
        self.gridLayout_2.addWidget(self.okbutton, 1, 1, 1, 1)
        self.cancelbutton = QtWidgets.QPushButton(batchinfodialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cancelbutton.sizePolicy().hasHeightForWidth())
        self.cancelbutton.setSizePolicy(sizePolicy)
        self.cancelbutton.setMinimumSize(QtCore.QSize(0, 0))
        self.cancelbutton.setMaximumSize(QtCore.QSize(200, 50))
        self.cancelbutton.setObjectName("cancelbutton")
        self.gridLayout_2.addWidget(self.cancelbutton, 1, 2, 1, 1)

        self.retranslateUi(batchinfodialog)
        QtCore.QMetaObject.connectSlotsByName(batchinfodialog)

    def retranslateUi(self, batchinfodialog):
        _translate = QtCore.QCoreApplication.translate
        batchinfodialog.setWindowTitle(_translate("batch_info_dialog", "Dialog"))
        self.label.setText(_translate("batch_info_dialog", "Sampling Rate (kHz):"))
        self.label_6.setText(_translate("batch_info_dialog", "cusum Threshold:"))
        self.label_3.setText(_translate("batch_info_dialog", "Min. Fractional Blockade:"))
        self.label_4.setText(_translate("batch_info_dialog", "Min. Dwell Time (μs):"))
        self.label_2.setText(_translate("batch_info_dialog", "Low-Pass Filter (kHz):"))
        self.minfracbox.setText(_translate("batch_info_dialog", "0"))
        self.sampratebox.setText(_translate("batch_info_dialog", "4166.67"))
        self.LPfilterbox.setText(_translate("batch_info_dialog", "150"))
        self.mindwellbox.setText(_translate("batch_info_dialog", "0"))
        self.cusumstepentry.setText(_translate("batch_info_dialog", "20"))
        self.label_5.setText(_translate("batch_info_dialog", "cusum Step:"))
        self.cusumthreshentry.setText(_translate("batch_info_dialog", "10"))
        self.label_7.setText(_translate("batch_info_dialog", "Min. Level time (μs)"))
        self.minleveltbox.setText(_translate("batch_info_dialog", "5"))
        self.label_8.setText(_translate("batch_info_dialog", "Max Levels per Event"))
        self.maxLevelsBox.setText(_translate("batch_info_dialog", "100"))
        self.invertCheckBox.setText(_translate("batch_info_dialog", "Invert"))
        self.selfCorrectCheckBox.setText(_translate("batch_info_dialog", "Self-Correct"))
        self.okbutton.setText(_translate("batch_info_dialog", "OK"))
        self.cancelbutton.setText(_translate("batch_info_dialog", "Cancel"))

