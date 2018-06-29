# -*- coding: utf8 -*-
import sys
from PythIon.Widgets.batchinfowidget import *


class BatchProcessor(QtGui.QWidget):

    def __init__(self, master=None):
        QtGui.QWidget.__init__(self,master)
        self.uibp = Ui_batchinfodialog()
        self.uibp.setup_ui(self)
        
        self.uibp.cancelbutton.clicked.connect(self.close)
        
    def close(self):
        self.destroy()
    
    
if __name__ == "__main__":
    global myapp_bp
    app_bp = QtGui.QApplication(sys.argv)
    myapp_bp = BatchProcessor()
    myapp_bp.show()
    sys.exit(app_bp.exec_())
