# flake8: noqa
"""
TODO:
Features:
0. Disable trace button initially if only 1 point is selected
1. Add a mode: -> DONE
    1. Continuous mode: continuing on the same disjoint segment
        - prev goal becomes the new start if there are already two points
2. accept or reject tracing -> DONE
    Do this after you create a tracing.
    1. if tracing is rejected:
        - if a tracing layer already exists for other tracings in the same
          layer, delete this particular tracing
        - if the tracing layer does not exist, and you specifically created
          it for this tracing, then removing the tracing layer itself.
    2. if tracing is accepted, do nothing. don't save path in the CSV yet
       until save trace is clicked
3. Load: -> Done
    - load swc to populate the answer to the tracing on load
    - disjoint tracing into runtime variables interactively
4. Visualization for tracing:
    - Not sure how to do this
5. Change layer selection to terminal points layer after tracing is drawn -> Bug in UI
GUI related:
- Read point size (for path width) from a textbox?
- Read trace path color and size from input
- restore edge color and point size after plotting complete to prev color/size
"""
import warnings  # noqa
from typing import Dict, List, Optional, Tuple  # noqa

import napari  # noqa
import numpy as np  # noqa
from brightest_path_lib.algorithm import AStarSearch, NBAStarSearch  # noqa
from napari.layers import Layer  # noqa
from napari.qt.threading import thread_worker  # noqa
from napari.utils.events import Event  # noqa
from napari.viewer import Viewer  # noqa
from qtpy.QtWidgets import QWidget  # noqa
from qtpy.QtWidgets import (  # noqa
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
)

from ._combobox import ComboBox
from ._dialog_widget import AcceptTracingWidget, SaveTracingWidget  # noqa
from ._layer_model import TracingLayers  # noqa
from ._logger import logger  # noqa
from ._segment_model import Segment  # noqa
from ._trace_loader import TraceLoader
from ._trace_saver import TraceSaver
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
        self.tracing_mode = "Disjoint Tracing Mode"

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

        self.configure_gui()

        self.viewer.layers.events.inserted.connect(
            self.slot_new_layer_inserted
        )
        self.viewer.layers.events.removing.connect(self.slot_removing_layer)
        self.viewer.layers.events.removed.connect(self.slot_layer_removed)

        self.accept_tracing_widget = AcceptTracingWidget()
        self.accept_tracing_widget.acceptTracing.connect(self.accept_tracing)
        self.accept_tracing_widget.rejectTracing.connect(self.reject_tracing)

        self.save_tracing_widget = SaveTracingWidget()
        self.save_tracing_widget.saveTracing.connect(self.save_tracing)
        self.save_tracing_widget.discardTracing.connect(self.discard_tracing)

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
        self.available_img_layers_combo_box = ComboBox(
            self
        )  # making it a self property because it will be constantly updated
        # self.available_img_layers_combo_box.setPlaceholderText(
        #     "--Select Layer for Tracing--"
        # )
        self.available_img_layers_combo_box.setPlaceholderText(
            "Select Image Layer"
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

        algo_and_mode_layout = QHBoxLayout()
        algorithm_selection_combo_box = QComboBox(self)
        algorithm_selection_combo_box.addItem("A* Search")
        algorithm_selection_combo_box.addItem("NBA* Search")
        algorithm_selection_combo_box.activated[str].connect(
            self.set_algorithm_for_tracing
        )
        algo_and_mode_layout.addWidget(algorithm_selection_combo_box)

        mode_selection_combo_box = QComboBox(self)
        mode_selection_combo_box.addItem("Disjoint Tracing Mode")
        mode_selection_combo_box.addItem("Continuous Tracing Mode")
        mode_selection_combo_box.activated[str].connect(self.set_tracing_mode)
        algo_and_mode_layout.addWidget(mode_selection_combo_box)

        # main_layout.addLayout(algo_selection_layout)
        main_layout.addLayout(algo_and_mode_layout)

        trace_button_layout = QHBoxLayout()
        trace_button = QPushButton("Start Tracing")
        trace_button.clicked.connect(self.trace_brightest_path)
        trace_button_layout.addWidget(trace_button)

        cancel_button = QPushButton("Cancel Tracing")
        cancel_button.clicked.connect(self.cancel_tracing)
        trace_button_layout.addWidget(cancel_button)

        save_and_load_button_layout = QHBoxLayout()
        save_trace_button = QPushButton("Save Trace")
        save_trace_button.clicked.connect(self.save_tracing)
        save_and_load_button_layout.addWidget(save_trace_button)

        load_button = QPushButton("Load Trace")
        load_button.clicked.connect(self.load_tracing)
        save_and_load_button_layout.addWidget(load_button)

        main_layout.addLayout(trace_button_layout)
        main_layout.addLayout(save_and_load_button_layout)

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
        self.active_image_layer_data = self.active_image_layer.data.copy()
        # configure image layer after selection
        self._configure_terminal_points_layer()

    def _configure_terminal_points_layer(self, connect_to_events: bool = True):
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
            if tracing_layers.terminal_points_layer is None:
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
            else:
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
        self.active_terminal_points_layer_data = (
            self.active_terminal_points_layer.data.copy()
        )

        # subscribing to data change in the terminal points layer
        if connect_to_events:
            self.active_terminal_points_layer.events.data.connect(
                self.slot_points_data_change
            )

    def set_algorithm_for_tracing(self, algo_name: str):
        logger.info(f"using {algo_name} as tracing algorithm")
        self.tracing_algorithm_name = algo_name

    def set_tracing_mode(self, mode: str):
        logger.info(f"using {mode} mode for tracing")
        self.tracing_mode = mode

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

    def _remove_image_layer_mappings(self, image_layer_id: int):
        if image_layer_id in self.layer_mapping:
            del self.layer_mapping[image_layer_id]

        if image_layer_id in self.traced_segments:
            del self.traced_segments[image_layer_id]

        if image_layer_id in self.most_recent_segment_id:
            del self.most_recent_segment_id[image_layer_id]

    def _reset_active_image_layer(self):
        if self.active_terminal_points_layer is not None:
            logger.info("Removing terminal points layer from viewer")
            self.viewer.layers.remove(self.active_terminal_points_layer.name)
            logger.info("Removed terminal points layer from viewer")
            self.active_terminal_points_layer = None
            self.active_terminal_points_layer_data = np.array([])

        if self.active_tracing_result_layer is not None:
            logger.info("Removing tracing result layer from viewer")
            self.viewer.layers.remove(self.active_tracing_result_layer.name)
            self.active_tracing_result_layer = None
            self.active_tracing_result_layer_data = np.array([])
            logger.info("Removed tracing result layer from viewer")

        self.active_image_layer = None
        self.active_image_layer_data = np.array([])

    def _reset_active_terminal_points_layer(self):
        active_img_layer_id = hash(self.active_image_layer)
        if active_img_layer_id in self.layer_mapping:
            logger.info(
                f"current layer mapping: {self.layer_mapping[active_img_layer_id].terminal_points_layer}"
            )
            tracing_layers = self.layer_mapping[active_img_layer_id]
            tracing_layers.terminal_points_layer = None
            logger.info(
                f"layer mapping after removing terminal pts layer: {self.layer_mapping[active_img_layer_id].terminal_points_layer}"
            )

        self.start_point = None
        self.goal_point = None
        self.active_terminal_points_layer_data = np.array([])
        self.active_terminal_points_layer = None

    def _reset_active_tracing_layer(self):
        active_img_layer_id = hash(self.active_image_layer)
        if active_img_layer_id in self.layer_mapping:
            logger.info(
                f"current layer mapping: {self.layer_mapping[active_img_layer_id].result_tracing_layer}"
            )
            tracing_layers = self.layer_mapping[active_img_layer_id]
            tracing_layers.result_tracing_layer = None
            logger.info(
                f"layer mapping after removing tracing result layer: {self.layer_mapping[active_img_layer_id].result_tracing_layer}"
            )

        if active_img_layer_id in self.traced_segments:
            self.traced_segments[active_img_layer_id] = []

        self.active_tracing_result_layer_data = np.array([])
        self.active_tracing_result_layer = None

    def slot_removing_layer(self, event: Event) -> None:
        """
        Add a dialog box to ask user if they want to save tracing, not save tracing
        or simply cancel (meaning stop deletion)
        """
        logger.info("Removing event emitted...")
        logger.info(vars(event))
        if not event or event.index:
            logger.info("No information about layer to be removed")
            return
        removing_layer = self.viewer.layers[event.index]
        removing_layer_id = hash(removing_layer)
        if removing_layer_id in self.traced_segments:
            if len(self.traced_segments[removing_layer_id]) > 0:
                self.save_tracing_widget.show()
        elif removing_layer == self.active_tracing_result_layer:
            self.save_tracing_widget.show()

    def slot_layer_removed(self, event: Event) -> None:
        """
        Respond to layer delete in viewer.

        Args:
            event (Event): event.type == 'removed'
        """
        logger.info("Removed event emitted....")
        logger.info(f"vars(event) = {vars(event)}")
        removed_layer = None
        if not event.value:
            logger.info("No information found about deleted layer...")
            logger.info("Returning...")
            return
        removed_layer = event.value
        removed_layer_id = hash(removed_layer)

        if isinstance(removed_layer, napari.layers.image.image.Image):
            index = self.available_img_layers_combo_box.findText(
                removed_layer.name
            )  # find the index of text
            self.available_img_layers_combo_box.removeItem(index)
            logger.info(f"removed {removed_layer} entry from combo box")

            self._remove_image_layer_mappings(removed_layer_id)

            if removed_layer == self.active_image_layer:
                logger.info("Removed layer was the active_image_layer")
                self._reset_active_image_layer()

        elif isinstance(removed_layer, napari.layers.points.points.Points):
            if removed_layer == self.active_terminal_points_layer:
                logger.info(
                    "Removed layer was the active_terminal_points_layer"
                )
                self._reset_active_terminal_points_layer()

            elif removed_layer == self.active_tracing_result_layer:
                logger.info(
                    "Removed layer was the active_tracing_result_layer"
                )
                self._reset_active_tracing_layer()

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
                self.change_color(idx, GREEN)

            elif self.goal_point is None and point != self.start_point:
                logger.info(f"Making {point} as our end")
                self.end_idx = len(event.source.data) - 1
                self.goal_point = point
                self.change_color(idx, RED)

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
            return result

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

    def _save_traced_segment(
        self, result: List[np.ndarray], start_point=None, goal_point=None
    ):
        """
        Saves a traced segment for an image layer the traced_segments
        dictionary
        """
        if start_point is None:
            start_point = self.start_point

        if goal_point is None:
            goal_point = self.goal_point

        segment_ID = self._get_segment_id()
        logger.info(
            f"Creating a tracing segment with segment_ID: {segment_ID}"
        )
        self.curr_traced_segment = Segment(
            segment_ID=segment_ID,
            start_point=start_point,
            goal_point=goal_point,
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

    def _reset_terminal_points(self):
        logger.info(
            f"Resetting terminal points based on the {self.tracing_mode}"
        )
        if self.tracing_mode == "Disjoint Tracing Mode":
            self.start_point = None
        else:
            self.start_point = self.goal_point
            idx = Utils.get_idx(
                target=self.goal_point,
                layer_data=self.active_terminal_points_layer_data,
            )
            logger.info(f"found idx {idx} for goal point")
            logger.info("Changing the idx color to orange")
            self.change_color(idx, GREEN)
            # get the index of the point layer in the terminal points layer data,
            # change its color to orange
        self.goal_point = None
        # add code to switch back to terminal points layer
        # so that user doesn't add new points in tracing result layer

    def trace_brightest_path(self):
        """call the brightest_path_lib to find brightest path
        between start_point and goal_point
        """
        self.worker = self._trace()
        self.worker.returned.connect(self.plot_brightest_path)
        self.worker.start()

    def plot_brightest_path(
        self, points: List[np.ndarray], prompt_to_save: bool = True
    ):
        logger.info("Plotting brightest path...")
        image_layer_id = hash(self.active_image_layer)
        if image_layer_id in self.layer_mapping:
            tracing_layers = self.layer_mapping[image_layer_id]
            if tracing_layers.result_tracing_layer is None:
                self.active_tracing_result_layer = self.viewer.add_points(
                    points,
                    name=self.active_image_layer.name + "_tracing",
                    size=1,
                    edge_width=1,
                    face_color="green",
                    edge_color="green",
                )
                tracing_layers.result_tracing_layer = (
                    self.active_tracing_result_layer
                )
            else:
                self.active_tracing_result_layer.data = np.append(
                    self.active_tracing_result_layer.data, points, axis=0
                )
            self.active_tracing_result_layer.refresh()

        if prompt_to_save:
            self.accept_tracing_widget.show()

            logger.info(
                f"Saved traces for {self.active_image_layer} \
                are {self.traced_segments}"
            )

        # switch back to active terminal points layer as current layer
        self.viewer.layers.current = self.active_terminal_points_layer

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

        if (
            self.curr_traced_segment is None
            or len(self.traced_segments.values()) == 0
        ):
            # we don't have any tracing to save
            logger.info("No currently traced segment found")
            logger.info("saving nothing.")
            return

        logger.info("Saving tracing...")

        fileName = QFileDialog.getSaveFileName(
            self,
            "Save Tracing As",
            self.active_tracing_result_layer.name,
            filter="*.csv",
        )
        if fileName:
            logger.info(f"Saving file as {fileName[0]}")
            active_layer_id = hash(self.active_image_layer)
            if active_layer_id in self.traced_segments:
                segments = self.traced_segments[active_layer_id]
                trace_saver = TraceSaver(fileName[0], segments)
                trace_saver.save_trace()

    def _get_row_values_for_saving_trace(self) -> List:
        rows = []

        active_layer_id = hash(self.active_image_layer)
        if active_layer_id in self.traced_segments:
            idx = 0
            for segment in self.traced_segments[active_layer_id]:
                prevIdx = -1
                result = segment.tracing_result
                for point in result:
                    if len(point) == 2:  # (y, x)
                        rows.append(
                            # [idx, "", point[1], point[0], prevIdx] # idx, z, x, y, prevIdx
                            [
                                idx,
                                point[1],
                                point[0],
                                "",
                                prevIdx,
                            ]  # idx, x, y, z, prevIdx
                        )
                    else:  # (z, y, x)
                        rows.append(
                            # [idx, point[0], point[2], point[1], prevIdx] # idx, z, x, y, prevIdx
                            [
                                idx,
                                point[2],
                                point[1],
                                point[0],
                                prevIdx,
                            ]  # idx, x, y, z, prevIdx
                        )

                    idx += 1
                    prevIdx += 1
        return rows

    def discard_tracing(self):
        """
        Discard the result of a tracing
        """
        logger.info("Discard tracing")
        return

    def accept_tracing(self):
        """
        Accept the current tracing
        """
        logger.info("Tracing accepted")
        self._reset_terminal_points()

    def reject_tracing(self):
        """
        discard the current tracing
        """
        # if this is the one and only tracing for that layer, then remove entire layer
        # otherwise just remove the one tracing
        active_img_layer_id = hash(self.active_image_layer)
        if active_img_layer_id in self.traced_segments:
            if len(self.traced_segments[active_img_layer_id]) > 1:
                recent_segment = self.traced_segments[active_img_layer_id][-1]
                start_index_of_recent_result = len(
                    self.active_tracing_result_layer.data
                ) - len(recent_segment.tracing_result)
                end_index_of_recent_result = (
                    len(self.active_tracing_result_layer.data) - 1
                )
                indices_of_result_points = list(
                    range(
                        start_index_of_recent_result,
                        end_index_of_recent_result + 1,
                    )
                )
                self.active_tracing_result_layer.data = np.delete(
                    self.active_tracing_result_layer.data,
                    indices_of_result_points,
                    0,
                )

                del self.traced_segments[active_img_layer_id][-1]
            else:
                if active_img_layer_id in self.layer_mapping:
                    logger.info(
                        f"current layer mapping: {self.layer_mapping[active_img_layer_id].result_tracing_layer}"
                    )
                    tracing_layers = self.layer_mapping[active_img_layer_id]
                    tracing_layers.result_tracing_layer = None
                    logger.info(
                        f"layer mapping after removing tracing result layer: {self.layer_mapping[active_img_layer_id].result_tracing_layer}"
                    )
                    logger.info("Removing tracing result layer from viewer")
                    self.viewer.layers.remove(
                        self.active_tracing_result_layer.name
                    )
                    self.active_tracing_result_layer_data = np.array([])
                    self.active_tracing_result_layer = None

    def load_tracing(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Trace File", "", "CSV files (*.csv)"
        )

        if filename:
            trace_loader = TraceLoader(filename)
            trace_loader.load_trace()
            terminal_points = trace_loader.terminal_points
            tracing_result_points = trace_loader.tracing_result_points

            # load points from SWC file first
            self.plot_terminal_points(points=terminal_points)

            # the below function call would then configure a tracing result layer
            # and plot the brightest path from the SWC file
            self.plot_brightest_path(
                points=tracing_result_points, prompt_to_save=False
            )
            self._reset_terminal_points()

            for tracing_result in trace_loader.tracing_results:
                self._save_traced_segment(
                    result=tracing_result,
                    start_point=tuple(tracing_result[0]),
                    goal_point=tuple(tracing_result[-1]),
                )

    def plot_terminal_points(self, points: List[Tuple]) -> None:
        # below call would create or set the terminal points layer
        self._configure_terminal_points_layer(connect_to_events=False)
        for start_point, goal_point in points:
            self.start_point = tuple(start_point)
            self.active_terminal_points_layer._face.current_color = GREEN
            self.active_terminal_points_layer.data = np.append(
                self.active_terminal_points_layer.data, [start_point], axis=0
            )

            self.goal_point = tuple(goal_point)
            self.active_terminal_points_layer._face.current_color = RED
            self.active_terminal_points_layer.data = np.append(
                self.active_terminal_points_layer.data, [goal_point], axis=0
            )

        self.active_terminal_points_layer._face.current_color = GREEN
        self.active_terminal_points_layer.refresh()
        self.active_terminal_points_layer.events.data.connect(
            self.slot_points_data_change
        )

    def change_color(self, idx: int, color: np.array) -> None:
        """
        change the color of a point in the layer
        """
        layer_colors = self.active_terminal_points_layer._face.colors
        layer_colors[idx] = color
        self.active_terminal_points_layer._face.current_color = WHITE
        self.active_terminal_points_layer.refresh()
