# -*- coding: utf8 -*-
import sys
import numpy as np
from PyQt5 import QtWidgets

from PythIon.Widgets.PoreSizerWidget import *
# from PyQt5 import QtGui


class PoreSizer(QtWidgets.QWidget):

    def __init__(self, master=None):
        QtWidgets.QWidget.__init__(self, master)
        self.ui_pore_size = Ui_PoreSizerWidget()
        self.ui_pore_size.setup_ui(self)
        
        self.ui_pore_size.compute_button.clicked.connect(self.sizepore)

    def sizepore(self, V=200, I_open=0.7, deltaI=0.5,
                 sigma=50, d_mol=2.2, prec=0.1):

        V = 200
        I_open = 0.7
        deltaI =0.5
        sigma = 50
        d_mol = 2.2
        prec = 0.1

        ui = self.ui_pore_size

        V = np.float64(ui.voltage.text())
        I_open = np.float64(ui.open_pore_current.text())
        deltaI = np.float64(ui.current_blockade.text())
        sigma = np.float64(ui.buffer_conductance.text())
        d_mol = np.float64(ui.analyte_diameter.text())
        prec = np.float64(ui.precision.text())
        
        d = list(np.arange(d_mol+prec, 20, prec))
        t = list(np.arange(0.1, 20, prec))
        I_b = I_open - deltaI
        diff_list = []
        dia_list = []
        thick_list = []
        
        for dia in d:
            d_eff = np.sqrt(dia**2-d_mol**2);
            for thi in t:
                q1 = sigma * V/1000 * ((4*thi)/(np.pi*dia**2) + 1/dia)**-1
                q2 = sigma * V/1000 * ((4*thi)/(np.pi*d_eff**2) + 1/d_eff)**-1
                diff1 = (I_open - q1/10)**2
                diff2 = (I_b - q2/10)**2
                diff_list.append(diff1 + diff2)
                dia_list.append(dia)
                thick_list.append(thi)
        
        ind = np.argmin(diff_list)
        diameter = dia_list[ind]
        thickness = thick_list[ind]
        
        self.ui_pore_size.pore_diameter.setText(str(diameter))
        self.ui_pore_size.pore_eff_thickness.setText(str(thickness))


if __name__ == "__main__":
    global myapp_sc
    myapp_sc = QtWidgets.QApplication(sys.argv)
    myapp_sc = PoreSizer()
    myapp_sc.show()
    sys.exit(app_sc.exec_())
