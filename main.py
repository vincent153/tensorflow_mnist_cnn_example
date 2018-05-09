import sys
from PyQt5 import QtCore
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget,QPushButton,QMessageBox

from PyQt5.QtGui import QPainter,QPainterPath,QPen,QImage
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
import load_and_predict
class Drawer(QWidget):
    newPoint = pyqtSignal(QtCore.QPoint)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.path = QPainterPath()
        self.btn = QPushButton('classify', self)
        self.btn.clicked.connect(self.btn_event)
        self._left = False
        self.setStyleSheet("background-color:white;")
        self.img_data = None
        print(dir(self.newPoint))

    def btn_event(self):
        digit = load_and_predict.classify(self.img_data)
        msg = QMessageBox(self)
        msg.setText('you are writing:{}'.format(digit))
        msg.show()
    def paintEvent(self, event):
        p = QPen(QtCore.Qt.black)
        p.setWidth(20)
        painter = QPainter(self)
        painter.setPen(p)
        painter.drawPath(self.path)
        painter.end()
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.path.moveTo(event.pos())
            self._left = True
            pass
        elif event.button() == QtCore.Qt.RightButton:
            pass
        #print(dir(event))
        self.update()

    def mouseMoveEvent(self, event):
        if self._left:
            self.path.lineTo(event.pos())
            self.newPoint.emit(event.pos())

            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            px = self.grab()
            img = px.toImage()
            img = img.convertToFormat(QImage.Format_Grayscale8)
            file_name = 'tmp.jpg'
            img.save(file_name)
            cvMat=cv2.imread(file_name,-1)
            cvMat = 255-cvMat
            self.img_data = cvMat
            #load_and_predict.classify(cvMat)
            self._left = False
            pass
        elif event.button() == QtCore.Qt.RightButton:
            self.path = QPainterPath()

            pass
        self.update()
        pass


    def sizeHint(self):
        return QtCore.QSize(300, 300)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Drawer()
    w.show()
    sys.exit(app.exec_())
