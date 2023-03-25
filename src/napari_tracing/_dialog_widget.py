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
        message = QLabel(
            "Do you want to save your tracing(s) to a CSV file before deleting this layer?"  # noqa
        )
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def on_accept(self):
        self.saveTracing.emit()
        self.close()

    def on_reject(self):
        self.discardTracing.emit()
        self.close()


class AcceptTracingWidget(QDialog):
    acceptTracing = Signal()
    rejectTracing = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Accept/Reject Tracing")

        QBtn = QDialogButtonBox.Yes | QDialogButtonBox.No

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.on_reject)

        self.layout = QVBoxLayout()
        message = QLabel("Do you accept this tracing?")  # noqa
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def on_accept(self):
        self.acceptTracing.emit()
        self.close()

    def on_reject(self):
        self.rejectTracing.emit()
        self.close()
