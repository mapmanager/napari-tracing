from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QComboBox,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
)


class ComboBox(QComboBox):
    # https://code.qt.io/cgit/qt/qtbase.git/tree/src/widgets/widgets/qcombobox.cpp?h=5.15.2#n3173
    def paintEvent(self, event):

        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QPalette.Text))

        # draw the combobox frame, focusrect and selected etc.
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)

        if self.currentIndex() < 0:
            opt.palette.setBrush(
                QPalette.ButtonText,
                opt.palette.brush(QPalette.ButtonText).color().lighter(),
            )
            if self.placeholderText():
                opt.currentText = self.placeholderText()

        # draw the icon and text
        painter.drawControl(QStyle.CE_ComboBoxLabel, opt)
