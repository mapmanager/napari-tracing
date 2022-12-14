import warnings
from collections import defaultdict
from queue import PriorityQueue
from typing import Dict, List, Optional, Tuple

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

STEP_SIZE = 0.5


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
        self.start_idx = -1
        self.end_idx = -1
        self.xrange = None
        self.yrange = None
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
        Configure a QHBoxLayout to hold the undo and redo buttons.
        """
        layout = QHBoxLayout()
        trace_button = QPushButton("Trace")
        trace_button.clicked.connect(self.trace)
        layout.addWidget(trace_button)
        self.setLayout(layout)

    def default_value(self) -> float:
        return float("inf")

    def calculate_h_score(self, p1: Tuple, p2: Tuple) -> float:
        """
        Returns the euclidean distance between two points
        """
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)
        # return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def is_close_to_end(
        self, point: Tuple, error_threshold: float = 1.0
    ) -> bool:
        dist = abs(point[0] - self.end[0]) + abs(point[1] - self.end[1])
        return dist <= error_threshold

    def set_ranges(self) -> None:
        """
        setter function for self.xrange and self.yrange
        """
        y_start, x_start = self.start
        y_end, x_end = self.end
        self.yrange = (min(y_start, y_end), max(y_start, y_end))
        self.xrange = (min(x_start, x_end), max(x_start, x_end))

    def get_neighbors(self, point: Tuple) -> List[Tuple]:
        """
        get left, right, top and bottom neighbors of a point
        (When h-func calcs manhattan distance)

        TODO:
        - later adds checks for barriers
        - Might need to calculate diagonal neighbors if h func calculates
        eucledian distance
        """
        y, x = point
        y_min, y_max = self.yrange
        x_min, x_max = self.xrange
        neighbors = []

        # top
        if y > y_min:
            neighbors.append((y - STEP_SIZE, x))

        # bottom
        if y < y_max - 0.5:
            neighbors.append((y + STEP_SIZE, x))

        # left
        if x > x_min:
            neighbors.append((y, x - STEP_SIZE))

        # right
        if x < x_max - 0.5:
            neighbors.append((y, x + STEP_SIZE))

        logger.info(f"found {len(neighbors)} neighbors")
        return neighbors

    def trace(self) -> bool:
        """
        A* tracing algorithm
        """
        logger.info("Starting trace...")
        if self.start is None or self.end is None:
            logger.info("Cannot trace without start/end points.")
            return False

        self.set_ranges()
        count = 0
        open_set = PriorityQueue()
        open_set.put(
            (0, count, self.start, self.start_idx)
        )  # distance, time of occurrence, point tuple
        came_from = {}
        g_score = defaultdict(self.default_value)
        g_score[self.start] = 0
        f_score = defaultdict(self.default_value)
        f_score[self.start] = self.calculate_h_score(self.start, self.end)

        open_set_hash = {self.start}

        while not open_set.empty():
            element = open_set.get()
            current = element[2]
            index = element[3]
            logger.info(f"inside while. current: {current} at index {index}")
            open_set_hash.remove(current)

            # if current == self.end:
            if self.is_close_to_end(current):
                self.draw_path(came_from, current)
                self.change_color(self.end_idx, TURQUOISE)
                self.change_color(self.start_idx, ORANGE)
                logger.info("found end point")
                return True

            neighbors = self.get_neighbors(current)
            logger.info("found neighbors")

            for neighbor in neighbors:
                temp_g_score = g_score[current] + STEP_SIZE

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = (current, index)
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.calculate_h_score(
                        neighbor, self.end
                    )
                    if neighbor not in open_set_hash:
                        count += 1
                        index = self.add_point(
                            np.array(neighbor), GREEN
                        )  # make open
                        open_set.put(
                            (f_score[neighbor], count, neighbor, index)
                        )
                        open_set_hash.add(neighbor)

            # draw()

            if current != self.start:
                # self.add_point(np.array(current), RED) # make closed
                # self.add_point(np.array(current)) # make closed
                self.change_color(index, RED)

        logger.info("Done tracing")
        return False

    def add_point(
        self, point: np.ndarray, color: Optional[np.ndarray] = None
    ) -> int:
        self.layer_data = np.append(self.layer_data, np.array([point]), axis=0)
        self.layer.data = self.layer_data
        index = len(self.layer_data) - 1

        if color is not None:
            self.change_color(index, color)

        self.layer.refresh()
        return index

    def change_color(self, idx: int, color: np.array) -> None:
        layer_colors = self.layer._face.colors
        layer_colors[idx] = color
        self.layer._face.current_color = WHITE
        self.layer.refresh()

    def draw_path(self, came_from: Dict[Tuple, Tuple], current: Tuple) -> None:
        """
        Draws final path by changing color of points in the path to purple.
        """
        while current in came_from:
            current, index = came_from[current]
            self.change_color(index, PURPLE)

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
            # point = tuple(map(int, tuple(point[0])))
            point = tuple(point[0])

            if self.start is None and point != self.end:
                logger.info(f"Making {point} as our start")
                self.start_idx = len(event.source.data) - 1
                self.start = point
                self.change_color(idx, ORANGE)

            elif self.end is None and point != self.start:
                logger.info(f"Making {point} as our end")
                self.end_idx = len(event.source.data) - 1
                self.end = point
                self.change_color(idx, TURQUOISE)

            self.layer_data = event.source.data.copy()

        elif len(event.source.data) < len(self.layer_data):
            logger.info("point is deleted")
            idx, point = Utils.get_diff(
                self.layer_data, event.source.data.copy()
            )
            point = tuple(point[0])

            if point == self.start:
                logger.info("resetting start since it was deleted")
                self.start = None

            elif point == self.end:
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
