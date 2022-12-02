import warnings
from typing import Optional

import numpy as np
from napari.layers import Layer
from napari.utils.events import Event
from napari.viewer import Viewer
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ._logger import logger
from .utils import Utils

RED = np.array([1, 0, 0, 1])
GREEN = np.array([0, 1, 0, 1])
WHITE = np.array([1, 1, 1, 1])
BLACK = np.array([0, 0, 0, 1])
PURPLE = np.array([0.5, 0, 0.5, 1])
ORANGE = np.array([1, 0.65, 0, 1])
TURQUOISE = np.array([0.25, 0.88, 0.82, 1])


class TracerWidget(QWidget):
    def __init__(self, viewer: Viewer, layer: Optional[Layer] = None) -> None:
        """
        Note that each layer has its command manager, which is why
        we are creating a dictionary of command managers
        which maps layerID to command manager instance.
        """
        super().__init__()

        warnings.filterwarnings(action="ignore", category=FutureWarning)

        self.viewer = viewer
        self.layer = None
        self.layer_data = np.array([])
        self.start = None
        self.end = None
        self.configure_gui()

        if layer:
            logger.info("layer found")
            logger.info(layer)
            self.layer = layer
        else:
            logger.info("layer not found. Finding active layer")
            active_layer = self.find_active_layers()
            if active_layer:
                logger.info("active layer found")
                logger.info(active_layer)
                self.layer = active_layer

        self.viewer.layers.events.inserted.connect(self.slot_insert_layer)
        self.viewer.layers.events.removed.connect(self.slot_remove_layer)
        self.viewer.layers.selection.events.changed.connect(
            self.slot_select_layer
        )

    def configure_gui(self) -> None:
        """
        Configure a QHBoxLayout to hold the undo and redo buttons.
        """
        layout = QHBoxLayout()
        trace_button = QPushButton("Trace")
        trace_button.clicked.connect(self.trace)
        layout.addWidget(trace_button)
        self.setLayout(layout)

    def trace(self):
        logger.info("Beginning tracing...")

    def find_active_layers(self) -> Optional[Layer]:
        """
        Find pre-existing selected layer.
        """
        currently_selected_layer = self.viewer.layers.selection.active
        if currently_selected_layer:
            return currently_selected_layer
        return None

    def connect_layer(self, layer: Layer) -> None:
        """
        Connect to a layer's events.
        Here, we're connecting to the data change event.

                Args:
                        layer: Layer (Layer to connect to.)
        """
        if self.layer:
            self.layer.events.data.disconnect(self.slot_points_data_change)
            # self.layer.events.highlight.disconnect(self.slot_user_selected_point)
            self.layer_data = np.array([])

        self.layer = layer
        if "data" in vars(layer).keys() and layer.data.any():
            self.layer_data = layer.data.copy()
        self.layer.events.data.connect(self.slot_points_data_change)
        # self.layer.events.highlight.connect(self.slot_user_selected_point)

    def slot_points_data_change(self, event: Event) -> None:
        """
        Modify the start/end points for a* search
        based on point adding/deletion
        """
        if Utils.are_sets_equal(event.source.selected_data, self.layer_data):
            # no change
            return

        if len(event.source.data) > len(self.layer_data):
            logger.info("new point added")
            idx, point = Utils.get_diff(event.source.data, self.layer_data)

            if self.start is None and not np.array_equal(point, self.end):
                logger.info(f"Making {point} as our start")
                self.start = point
                self.change_color(idx, ORANGE)

            elif self.end is None and not np.array_equal(point, self.start):
                logger.info(f"Making {point} as our end")
                self.end = point
                self.change_color(idx, TURQUOISE)

            self.layer_data = event.source.data.copy()

        elif len(event.source.data) < len(self.layer_data):
            logger.info("point is deleted")
            idx, point = Utils.get_diff(
                self.layer_data, event.source.data.copy()
            )
            if np.array_equal(point, self.start):
                logger.info("resetting start since it was deleted")
                self.start = None

            elif np.array_equal(point, self.end):
                logger.info("resetting end since it was deleted")
                self.end = None

            self.layer_data = event.source.data.copy()

    def slot_insert_layer(self, event: Event) -> None:
        """
        Respond to new layer in viewer.

        Args:
            event (Event): event.type == 'inserted'
        """
        logger.info(
            f'New layer "{event.source}" was inserted at index {event.index}'
        )
        newly_inserted_layer = event.value
        if newly_inserted_layer:
            self.connect_layer(newly_inserted_layer)

    def slot_remove_layer(self, event: Event) -> None:
        """
        Respond to layer delete in viewer.

        Args:
            event (Event): event.type == 'removed'
        """
        logger.info(f'Removed layer "{event.source}"')
        currently_selected_layer = self.find_active_layers()
        if currently_selected_layer and currently_selected_layer != self.layer:
            self.connect_layer(currently_selected_layer)

    def slot_select_layer(self, event: Event) -> None:
        """Respond to layer selection in viewer.

        Args:
            event (Event): event.type == 'changed'
        """
        currently_selected_layer = self.find_active_layers()
        if currently_selected_layer and currently_selected_layer != self.layer:
            logger.info(f"New layer selected: {currently_selected_layer}")
            self.connect_layer(currently_selected_layer)

    def change_color(self, idx: int, color: np.array):
        layer_colors = self.layer._face.colors
        logger.info(f"layer colors: {layer_colors}")
        layer_colors[idx] = color
        self.layer._face.current_color = WHITE
        self.layer.refresh()
