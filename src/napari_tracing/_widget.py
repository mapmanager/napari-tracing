"""
TODO:
Features:

- When saving a trace, use the following layout:
    Segment_ID x_start y_start z_start(if applicable) x y z

- save the tracing layer as <image_layer_name_tracing>
- how can you populate the napari UI using a CSV file

GUI related:
- Vertically center the gui layout on the screen
- Read point size (for path width) from a textbox?
- Read trace path color and size from input
- restore edge color and point size after plotting complete to prev color/size
"""
import csv
import os
import sys

sys.path.append("/Users/vasudhajha/Documents/mapmanager/brightest-path-lib")
sys.path.append(
    "/Users/vasudhajha/Documents/mapmanager/brightest-path-lib/brightest_path_lib"  # noqa
)
sys.path.append(
    "/Users/vasudhajha/Documents/mapmanager/brightest-path-lib/brightest_path_lib/algorithm"  # noqa
)
import warnings  # noqa
from typing import Dict, List, Optional  # noqa

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
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._dialog_widget import SaveTracingWidget  # noqa
from ._layer_model import TracingLayers  # noqa
from ._logger import logger  # noqa
from ._segment_model import Segment  # noqa
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
        # stores a reference to the currently active image layer
        self.active_image_layer = None

        # stores a reference to the currently active terminal points layer
        self.active_terminal_points_layer = None

        # stores a reference to the currently active tracing results layer
        self.active_tracing_result_layer = None

        self.active_image_layer_data = np.array([])
        self.active_terminal_points_layer_data = np.array([])
        self.active_tracing_result_layer_data = np.array([])

        self.start_point = None
        self.goal_point = None
        self.start_idx = -1
        self.end_idx = -1

        # algorithm choice for brightest path tracing
        self.tracing_algorithm_name = "A* Search"

        # the result of the current tracing
        self.curr_traced_segment: Segment = None

        # maps the active image_layer hash/id to a TracingLayers object
        # (containing a reference of the terminal points layer and
        # tracing layer mapped to the current layer)
        self.layer_mapping: Dict[int:TracingLayers] = {}

        # maps the active image_layer hash/id
        # to a list of tracings for that layer
        self.traced_segments: Dict[int : List[Segment]] = {}  # noqa

        # maps a layer id to its most recent segment value
        self.most_recent_segment_id: Dict[int:int] = {}

        self.worker = None
        self.current_tracing_result = None

        self.configure_gui()

        self.viewer.layers.events.inserted.connect(
            self.slot_new_layer_inserted
        )
        self.viewer.layers.events.removed.connect(self.slot_layer_removed)

    def find_active_layers(self) -> Optional[Layer]:
        """
        Find pre-existing selected layer.
        """
        currently_selected_layer = self.viewer.layers.selection.active
        if currently_selected_layer:
            return currently_selected_layer
        return None

    def _find_image_layers(self) -> List[Layer]:
        """
        find all the available image layers in the current viewer instance
        """
        image_layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.image.image.Image)
        ]
        return image_layers

    def _add_to_available_img_layers_combo_box(self, img_layers: List[Layer]):
        for img_layer in img_layers:
            self.available_img_layers_combo_box.addItem(img_layer.name)

    def configure_gui(self) -> None:
        """
        Configure a QHBoxLayout to hold the trace and cancel buttons.
        """
        main_layout = QVBoxLayout()

        # this property will highlight a list of available image layers
        # for which tracing algorithms can be run
        self.available_img_layers_combo_box = QComboBox(
            self
        )  # making it a self property because it will be constantly updated
        self.available_img_layers_combo_box.setPlaceholderText(
            "--Available Image Layers for Tracing--"
        )

        # if napari has image layers already present,
        # then show them in the drop-down
        available_image_layers = self._find_image_layers()
        if len(available_image_layers) > 0:
            self._add_to_available_img_layers_combo_box(available_image_layers)
        self.available_img_layers_combo_box.activated[str].connect(
            self.set_img_layer_for_tracing
        )

        main_layout.addWidget(self.available_img_layers_combo_box)

        algo_selection_layout = QVBoxLayout()
        algorithm_selection_combo_box = QComboBox(self)
        algorithm_selection_combo_box.addItem("A* Search")
        algorithm_selection_combo_box.addItem("NBA* Search")
        algorithm_selection_combo_box.activated[str].connect(
            self.set_algorithm_for_tracing
        )
        algo_selection_layout.addWidget(algorithm_selection_combo_box)

        main_layout.addLayout(algo_selection_layout)

        button_layout = QHBoxLayout()
        trace_button = QPushButton("Trace")
        trace_button.clicked.connect(self.trace_brightest_path)
        button_layout.addWidget(trace_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_tracing)
        button_layout.addWidget(cancel_button)

        save_trace_button = QPushButton("Save Trace")
        save_trace_button.clicked.connect(self.save_tracing)
        button_layout.addWidget(save_trace_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def set_img_layer_for_tracing(self, img_layer_name: str):
        """
        Set the global image layer to the one selected in the combo box
        """
        logger.info(
            f"using {img_layer_name} as our layer for tracing the image"
        )
        logger.info(f"Available layers: {self.viewer.layers}")
        img_layers = [
            layer
            for layer in self.viewer.layers
            if layer.name == img_layer_name
        ]

        if len(img_layers) <= 0:
            logger.info("Couldn't find the selected image layer for tracing")
        else:
            self.active_image_layer = img_layers[0]

        logger.info(
            f"selected new image layer for tracing {self.active_image_layer}"
        )

        # configure image layer after selection
        self._configure_layers_for_tracing()

    def _configure_layers_for_tracing(self):
        """
        configures properties of the image_layer that need to be
        tracked once its selected to be
        used for tracing
        """
        # first check if the image layer and its corresponding tracing
        # and results objects already exist
        layer_id = hash(self.active_image_layer)
        if layer_id in self.layer_mapping:
            tracing_layers = self.layer_mapping[layer_id]
            self.active_terminal_points_layer = (
                tracing_layers.terminal_points_layer
            )
        else:
            terminal_layer_name = (
                self.active_image_layer.name + "_terminal_points_layer"
            )
            self.active_terminal_points_layer = self.viewer.add_points(
                name=terminal_layer_name
            )
            tracing_layers = TracingLayers()
            tracing_layers.terminal_points_layer = (
                self.active_terminal_points_layer
            )
            self.layer_mapping[layer_id] = tracing_layers

        # Copy the active layers data. We will use it later on for tracing
        self.active_image_layer_data = self.active_image_layer.data.copy()
        self.active_terminal_points_layer_data = (
            self.active_terminal_points_layer.data.copy()
        )

        # subscribing to data change in the terminal points layer
        self.active_terminal_points_layer.events.data.connect(
            self.slot_points_data_change
        )

    def set_algorithm_for_tracing(self, algo_name: str):
        logger.info(f"using {algo_name} as tracing algorithm")
        self.tracing_algorithm_name = algo_name

    def slot_new_layer_inserted(self, event: Event) -> None:
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
            if isinstance(
                newly_inserted_layer, napari.layers.image.image.Image
            ):
                self.available_img_layers_combo_box.addItem(
                    newly_inserted_layer.name
                )

    def slot_layer_removed(self, event: Event) -> None:
        """
        Respond to layer delete in viewer.

        Args:
            event (Event): event.type == 'removed'
        """
        if isinstance(event.source, napari.layers.image.image.Image):
            logger.info(f'Removed image layer "{event.source}"')
            index = self.available_img_layers_combo_box.findText(
                event.source.name
            )  # find the index of text
            self.available_img_layers_combo_box.removeItem(index)
            logger.info(f"removed {event.source} entry from combo box")

            layer_id = hash(event.source)

            if layer_id in self.layer_mapping:
                del self.layer_mapping[layer_id]

            if layer_id in self.traced_segments:
                del self.traced_segments[layer_id]

            if layer_id in self.most_recent_segment_id:
                del self.most_recent_segment_id[layer_id]

            if event.source == self.active_image_layer:
                logger.info("Removed layer was the active_image_layer")
                layer_id = hash(self.active_image_layer)

                if self.active_terminal_points_layer is not None:
                    self.viewer.layers.remove(
                        self.active_terminal_points_layer.name
                    )
                    self.active_terminal_points_layer = None
                    self.active_terminal_points_layer_data = np.array([])

                if self.active_tracing_result_layer is not None:
                    self.viewer.layers.remove(self.active_tracing_result_layer)
                    self.active_tracing_result_layer = None
                    self.active_tracing_result_layer_data = np.array([])

                self.active_image_layer = None
                self.active_image_layer_data = np.array([])

        elif isinstance(event.source, napari.layers.points.points.Points):
            layer_id = hash(self.active_image_layer_data)
            if event.source == self.active_terminal_points_layer:
                if layer_id in self.layer_mapping:
                    tracing_layers = self.layer_mapping[layer_id]
                    tracing_layers.terminal_points_layer = None

                self.start_point = None
                self.goal_point = None
                self.active_terminal_points_layer_data = np.array([])
                self.active_terminal_points_layer = None

            elif event.source == self.active_tracing_result_layer:
                if layer_id in self.layer_mapping:
                    tracing_layers = self.layer_mapping[layer_id]
                    tracing_layers.result_tracing_layer = None

                if layer_id in self.traced_segments:
                    del self.traced_segments[layer_id]

                self.active_tracing_result_layer_data = np.array([])
                self.active_tracing_result_layer = None

    def slot_points_data_change(self, event: Event) -> None:
        """
        Modify the start/end points for a* search
        based on point adding/deletion
        """
        logger.info("Inside slot_points_data_change")
        if event.source != self.active_terminal_points_layer:
            return

        if Utils.are_sets_equal(
            event.source.selected_data, self.active_terminal_points_layer_data
        ):
            # no change
            return

        if len(event.source.data) > len(
            self.active_terminal_points_layer_data
        ):
            logger.info("new point added")
            idx, points = Utils.get_diff(
                event.source.data, self.active_terminal_points_layer_data
            )
            # point = tuple(map(int, tuple(point[0])))
            point = tuple(points[0])

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

            self.active_terminal_points_layer_data = event.source.data.copy()

        elif len(event.source.data) < len(
            self.active_terminal_points_layer_data
        ):
            logger.info("point is deleted")
            idx, point = Utils.get_diff(
                self.active_terminal_points_layer_data,
                event.source.data.copy(),
            )
            point = tuple(point[0])

            if point == self.start_point:
                logger.info("resetting start since it was deleted")
                self.start_point = None

            elif point == self.goal_point:
                logger.info("resetting end since it was deleted")
                self.goal_point = None

            self.active_terminal_points_layer_data = event.source.data.copy()

    @thread_worker
    def _trace(self):
        if len(self.active_image_layer_data) == 0:
            logger.info("No image data to trace")
            return

        if self.start_point is None or self.goal_point is None:
            logger.info("No start/end point specified")
            return

        if self.tracing_algorithm_name == "NBA* Search":
            tracing_algorithm = NBAStarSearch(
                self.active_image_layer_data, self.start_point, self.goal_point
            )
        else:
            tracing_algorithm = AStarSearch(
                self.active_image_layer_data, self.start_point, self.goal_point
            )

        result = tracing_algorithm.search()
        if not result or len(result) == 0:
            logger.info("Couldn't find any brightest path")
            return []
        else:
            logger.info(
                f"Completed tracing. Found path of length {len(result)}"
            )
            self._save_traced_segment(result)
            self.current_tracing_result = result
            return result

    def _save_traced_segment(self, result: List[np.ndarray]):
        """
        Saves a traced segment for an image layer the traced_segments
        dictionary
        """
        segment_ID = self._get_segment_id()
        logger.info(
            f"Creating a tracing segment with segment_ID: {segment_ID}"
        )
        self.curr_traced_segment = Segment(
            segment_ID=segment_ID,
            start_point=self.start_point,
            goal_point=self.goal_point,
            tracing_result=result,
            tracing_algorithm=self.tracing_algorithm_name,
        )
        logger.info(f"Created a segment: {self.curr_traced_segment}")

        layer_id = hash(self.active_image_layer)
        logger.info("saving tracing for " + f"layer id {layer_id}")

        if layer_id in self.traced_segments:
            self.traced_segments[layer_id].append(self.curr_traced_segment)
        else:
            self.traced_segments[layer_id] = [self.curr_traced_segment]
        logger.info("saved tracing segment")

    def _get_segment_id(self):
        """
        get the most recent tracing segment ID for an image layer from
        """
        layer_id = hash(self.active_image_layer)
        logger.info("getting segmentID for " + f"layer id {layer_id}")
        if layer_id in self.most_recent_segment_id:
            self.most_recent_segment_id[layer_id] += 1
        else:
            self.most_recent_segment_id[layer_id] = 0
        return self.most_recent_segment_id[layer_id]

    def trace_brightest_path(self):
        """call the brightest_path_lib to find brightest path
        between start_point and goal_point
        """
        self.worker = self._trace()
        self.worker.returned.connect(self.plot_brightest_path)
        self.worker.start()

    def plot_brightest_path(self, points: List[np.ndarray]):
        logger.info("Plotting brightest path...")
        if self.active_tracing_result_layer is None:
            self.active_tracing_result_layer = self.viewer.add_points(
                points,
                name=self.active_image_layer.name + "_tracing",
                size=1,
                edge_width=1,
                face_color="green",
                edge_color="green",
            )
        else:
            # append the points numpy array to the data in tracing_result_layer
            self.active_tracing_result_layer.data = np.append(
                self.active_tracing_result_layer.data, points, axis=0
            )
            self.active_tracing_result_layer.refresh()

        self._reset_terminal_points()
        logger.info(
            f"Saved traces for {self.active_image_layer} \
                    are {self.traced_segments}"
        )

    def _reset_terminal_points(self):
        self.start_point = None
        self.goal_point = None
        # add code to switch back to terminal points layer
        # so that user doesn't add new points in tracing result layer

    def cancel_tracing(self):
        """Cancel brightest path tracing"""
        if self.worker:
            logger.info("Cancelling tracing...")
            self.worker.quit()
            self.worker = None

    def save_tracing(self):
        """
        Saves the result of tracings in a CSV file.
        Default mode is "w", if a tracing for that layer doesn't exists already
        otherwise, it appends to an already present file
        """
        if self.start_point is None or self.goal_point is None:
            # we don't have any tracing to save
            logger.info("No tracing done since start and end points are None")
            return

        logger.info("Saving tracing")
        tracing = Segment(
            self.start_point, self.goal_point, self.current_tracing_result
        )
        self.layer_segment_mapping.append(tracing)
        fileName = QFileDialog.getSaveFileName(
            self, "Save Tracing As", "", filter="*.csv"
        )
        if fileName:
            logger.info(f"Saving file as {fileName[0]}")

            file_mode = "w"
            file_exists = os.path.exists(fileName[0])

            if file_exists:
                file_mode = "a"

            with open(fileName[0], file_mode) as f:
                writer = csv.writer(f)
                if not file_exists:
                    column_headers = self.get_column_headers_for_saving_trace()
                    writer.writerow(column_headers)
                writer.writerow(
                    [
                        self.start_point,
                        self.goal_point,
                        self.current_tracing_result,
                    ]
                )

            logger.info("Resetting start and goal point variables")

        self.start_point = None
        self.goal_point = None

    def get_column_headers_for_saving_trace(self) -> List[str]:
        if len(self.start_point) == 2:
            # we're dealing with a 2D image
            return [
                "segment_ID",
                "x_start",
                "y_start",
                "x_goal",
                "y_goal",
                "path_x",
                "path_y",
            ]
        else:
            # we're dealing with a 3D image
            return [
                "segment_ID",
                "x_start",
                "y_start",
                "z_start",
                "x_goal",
                "y_goal",
                "z_goal",
                "path_x",
                "path_y",
                "path_z",
            ]

    def get_row_values_for_saving_trace(self) -> List:
        if len(self.start_point) == 2:
            return []
        else:
            return []

    def discard_tracing(self):
        """
        Discard the result of a tracing
        """
        logger.info("Discard tracing")
        if self.active_tracing_result_layer is not None:
            self.viewer.layers.remove("Tracing")
            self.active_tracing_result_layer = None

    def change_color(self, idx: int, color: np.array) -> None:
        """
        change the color of a point in the layer
        """
        layer_colors = self.active_terminal_points_layer._face.colors
        layer_colors[idx] = color
        self.active_terminal_points_layer._face.current_color = WHITE
        self.active_terminal_points_layer.refresh()
