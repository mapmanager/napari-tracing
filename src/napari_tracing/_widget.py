"""
TODO:
- Add a points layer for the terminal points when the plugin is initialized
- Store a reference to this points layer as self.SOMETHING
- Store the data of this points layer as self.SOMETHING_data
- Listen to this points layer in a different way (different methods) compared
to the results points layer. Maybe don't listen to results points layer at all.

GUI related:
- Reset points layer by removing start and end points when Cancel is clicked
- Reset button
- Vertically center the gui layout on the screen
- Read point size (for path width) from a textbox?
- Read trace path color and size from input
- Save trace button
- restore edge color and point size after plotting complete to prev color/size
"""
import sys

sys.path.append("/Users/vasudhajha/Documents/mapmanager/brightest-path-lib")
sys.path.append(
    "/Users/vasudhajha/Documents/mapmanager/brightest-path-lib/brightest_path_lib"  # noqa
)
sys.path.append(
    "/Users/vasudhajha/Documents/mapmanager/brightest-path-lib/brightest_path_lib/algorithm"  # noqa
)
import warnings  # noqa
from typing import List, Optional  # noqa

import napari  # noqa
import numpy as np  # noqa
from algorithm import AStarSearch, NBAStarSearch  # noqa
from brightest_path_lib.algorithm import AStarSearch, NBAStarSearch  # noqa
from napari.layers import Layer  # noqa
from napari.qt.threading import thread_worker  # noqa
from napari.utils.events import Event  # noqa
from napari.viewer import Viewer  # noqa
from qtpy.QtWidgets import (  # noqa
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._logger import logger  # noqa
from .utils import Utils  # noqa

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
        self.terminal_points_layer = None
        self.tracing_result_layer = None
        self._terminal_points_layer_data = np.array([])
        self._image_layer_data = np.array([])
        self.start_point = None
        self.goal_point = None
        self.start_idx = -1
        self.end_idx = -1
        self.worker = None
        self.tracing_algorithm_name = "A* Search"
        self.configure_gui()

        if layer:
            logger.info("layer found")
            logger.info(layer)
            self.layer = layer
            self.connect_layer(layer)
        else:
            logger.info("layer not found. Finding active layer")
            active_layer = self.find_active_layers()
            if active_layer:
                logger.info("active layer found")
                logger.info(active_layer)
                self.layer = active_layer
                self.connect_layer(active_layer)

        self.viewer.layers.events.inserted.connect(self.slot_insert_layer)
        self.viewer.layers.events.removed.connect(self.slot_remove_layer)
        self.viewer.layers.selection.events.changed.connect(
            self.slot_select_layer
        )

    def configure_gui(self) -> None:
        """
        Configure a QHBoxLayout to hold the trace and cancel buttons.
        """
        main_layout = QVBoxLayout()

        combo_box = QComboBox(self)
        combo_box.addItem("A* Search")
        combo_box.addItem("NBA* Search")
        combo_box.activated[str].connect(self.set_algorithm_for_tracing)

        main_layout.addWidget(combo_box)

        button_layout = QHBoxLayout()
        trace_button = QPushButton("Trace")
        trace_button.clicked.connect(self.trace_brightest_path)
        button_layout.addWidget(trace_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_tracing)
        button_layout.addWidget(cancel_button)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        button_layout.addWidget(reset_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def set_algorithm_for_tracing(self, text):
        logger.info(f"using {text} as tracing algorithm")
        self.tracing_algorithm_name = text

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
            if isinstance(self.layer, napari.layers.points.points.Points):
                self.layer.events.data.disconnect(self.slot_points_data_change)
                # self.layer.events.highlight.disconnect(self.slot_user_selected_point)

                if self.terminal_points_layer is None:
                    self.terminal_points_layer = None
                    self._terminal_points_layer_data = np.array([])

        self.layer = layer
        logger.info(f"Found new layer of type {type(self.layer)}")

        if isinstance(self.layer, napari.layers.points.points.Points):
            # if "data" in vars(layer).keys() and layer.data.any():
            if "_data" in vars(layer).keys() and layer.data.any():
                logger.info("Getting points layer data")

                if self.terminal_points_layer is None:
                    self.terminal_points_layer = layer
                    self._terminal_points_layer_data = layer.data.copy()

            self.layer.events.data.connect(self.slot_points_data_change)

        elif isinstance(self.layer, napari.layers.image.image.Image):
            if "_data" in vars(layer).keys() and layer.data.any():
                logger.info("Getting image layer data")
                self._image_layer_data = layer.data.copy()

        # self.layer.events.highlight.connect(self.slot_user_selected_point)

    def slot_points_data_change(self, event: Event) -> None:
        """
        Modify the start/end points for a* search
        based on point adding/deletion
        """
        logger.info("Inside slot_points_data_change")

        if Utils.are_sets_equal(
            event.source.selected_data, self._terminal_points_layer_data
        ):
            # no change
            return

        if len(event.source.data) > len(self._terminal_points_layer_data):
            logger.info("new point added")
            idx, point = Utils.get_diff(
                event.source.data, self._terminal_points_layer_data
            )
            # point = tuple(map(int, tuple(point[0])))
            point = tuple(point[0])

            if self.start_point is None and point != self.goal_point:
                logger.info(f"Making {point} as our start")
                self.start_idx = len(event.source.data) - 1
                self.start_point = point
                self.change_color(idx, ORANGE)

            elif self.goal_point is None and point != self.start_point:
                logger.info(f"Making {point} as our end")
                self.end_idx = len(event.source.data) - 1
                self.goal_point = point
                self.change_color(idx, TURQUOISE)

            self._terminal_points_layer_data = event.source.data.copy()

        elif len(event.source.data) < len(self._terminal_points_layer_data):
            logger.info("point is deleted")
            idx, point = Utils.get_diff(
                self._terminal_points_layer_data, event.source.data.copy()
            )
            point = tuple(point[0])

            if point == self.start_point:
                logger.info("resetting start since it was deleted")
                self.start_point = None

            elif point == self.goal_point:
                logger.info("resetting end since it was deleted")
                self.goal_point = None

            self._terminal_points_layer_data = event.source.data.copy()

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

    @thread_worker
    def _trace(self):
        if len(self._image_layer_data) == 0:
            logger.info("No image data to trace")
            return
        if self.tracing_algorithm_name == "NBA* Search":
            tracing_algorithm = NBAStarSearch(
                self._image_layer_data, self.start_point, self.goal_point
            )
        else:
            tracing_algorithm = AStarSearch(
                self._image_layer_data, self.start_point, self.goal_point
            )

        result = tracing_algorithm.search()
        logger.info(f"Completed tracing. Found path of length {len(result)}")
        return result

    def trace_brightest_path(self):
        """call the brightest_path_lib to find brightest path
        between start_point and goal_point
        """
        self.worker = self._trace()
        self.worker.returned.connect(self.plot_brightest_path)
        self.worker.start()

    def plot_brightest_path(self, points: List[np.ndarray]):
        logger.info("Plotting brightest path...")
        self.tracing_result_layer = self.viewer.add_points(
            points,
            name="Tracing",
            size=1,
            edge_width=1,
            face_color="green",
            edge_color="green",
        )

    def cancel_tracing(self):
        """Cancel brightest path tracing"""
        if self.worker:
            logger.info("Cancelling tracing...")
            self.worker.quit()
            self.worker = None

    def reset(self):
        """
        reset the UI changes that were made for tracing
        """

        # clear the start and end points
        self.start_point = None
        self.goal_point = None

        if self.terminal_points_layer is not None:
            logger.info(
                f"Resetting the points layer {self.terminal_points_layer}"
            )
            point_indices_to_delete = [
                x for x in range(len(self.terminal_points_layer.data))
            ]
            logger.info(
                f"Deleting points at these indices: {point_indices_to_delete}"
            )
            self.terminal_points_layer.data = np.delete(
                self.terminal_points_layer.data, point_indices_to_delete, 0
            )
            self.terminal_points_layer.refresh()
            self._terminal_points_layer_data = np.array([])

        # remove points layer added for result
        if self.tracing_result_layer is not None:
            self.viewer.layers.remove("Tracing")
            self.tracing_result_layer = None

    def add_point(self, point: np.ndarray, color: Optional[np.ndarray] = None):
        """
        add a new point in the napari layer
        """
        self._terminal_points_layer_data = np.append(
            self._terminal_points_layer_data, np.array([point]), axis=0
        )
        self.layer.data = self._terminal_points_layer_data
        index = len(self._terminal_points_layer_data) - 1

        if color is not None:
            self.change_color(index, color)

        self.layer.refresh()

    # def delete_point(self, point: np.ndarray):
    #     """
    #     delete a point
    #     """
    #     pass

    def change_color(self, idx: int, color: np.array) -> None:
        """
        change the color of a point in the layer
        """
        layer_colors = self.layer._face.colors
        layer_colors[idx] = color
        self.layer._face.current_color = WHITE
        self.layer.refresh()
