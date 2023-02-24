from qtpy.QtCore import Signal
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class SaveTracingWidget(QDialog):
    saveTracing = Signal()
    discardTracing = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Tracing Result")

        QBtn = QDialogButtonBox.Yes | QDialogButtonBox.No

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.on_reject)

        self.layout = QVBoxLayout()
        message = QLabel("Do you want to save your tracing to a CSV file?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def on_accept(self):
        self.saveTracing.emit()
        self.close()

    def on_reject(self):
        self.discardTracing.emit()
        self.close()
