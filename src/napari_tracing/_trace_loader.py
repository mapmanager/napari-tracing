import csv
from typing import List, Tuple

import numpy as np


class TraceLoader:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.terminal_points: List[Tuple] = []
        self.tracing_results: List[
            List[np.ndarray]
        ] = []  # stores a list of result points, one list item per segment
        self.tracing_result_points: List[
            np.ndarray
        ] = []  # stores all the result points in one list
        self.last_known_start_point: np.ndarray = np.array([])
        self.last_known_end_point: np.ndarray = np.array([])
        self.previous_row: List = []

    def load_trace(self) -> None:
        with open(self.filename) as file:
            reader = csv.reader(file)
            next(reader)  # skip the header row
            tracing_result = []

            for string_row in reader:  # row format: idx, x, y, z, prevIdx
                row = [
                    int(element) for element in string_row if len(element) > 0
                ]
                point = self._get_point(row)
                self.tracing_result_points.append(point)

                # check if this row is for a start point
                prevIdx = row[-1]
                if prevIdx == -1:
                    if len(self.last_known_start_point) > 0:
                        # a previous start point already exists
                        # and this is the next start point.
                        self.last_known_end_point = self._get_point(
                            self.previous_row
                        )
                        self.terminal_points.append(
                            (
                                self.last_known_start_point.copy(),
                                self.last_known_end_point.copy(),
                            )
                        )
                        self.last_known_end_point = np.array([])

                        self.tracing_results.append(tracing_result)
                        tracing_result = []

                    self.last_known_start_point = point

                tracing_result.append(point)
                self.previous_row = row

        if (
            len(self.last_known_start_point) > 0
            and len(self.last_known_end_point) == 0
        ):
            self.last_known_end_point = self._get_point(self.previous_row)
            self.terminal_points.append(
                (
                    self.last_known_start_point.copy(),
                    self.last_known_end_point.copy(),
                )
            )
            self.tracing_results.append(tracing_result)

    def _get_point(self, row: List[str]) -> np.ndarray:
        if (
            len(row) == 4
        ):  # 2D because we only have columns idx, x, y and prevIdx
            point = np.array([row[2], row[1]])  # (y,x)
        else:
            point = np.array([row[3], row[2], row[1]])  # (z,y,x)

        return point
