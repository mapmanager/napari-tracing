import os
from typing import List

from ._logger import logger  # noqa
from ._segment_model import Segment

TYPE = 1
RADIUS = 1


class TraceSaver:
    def __init__(
        self, directory: str, layer_name: str, segments: List[Segment]
    ) -> None:
        self.directory = directory
        self.layer_name = layer_name
        self.segments = segments
        self.segment_ids_added = set()

    def save_trace(self):
        merged_segments = self._merge_segments()
        for segment in merged_segments:
            file_path = os.path.join(
                self.directory, f"{self.layer_name}_{segment.segment_ID}.swc"
            )
            with open(file_path, "w") as f:
                # column_headers: idx, type, x, y, z, radius, prevIdx
                for row in self._get_row_values_for_saving_trace(segment):
                    f.write(" ".join(str(element) for element in row))
                    f.write("\n")

    def _get_row_values_for_saving_trace(self, segment: "Segment") -> List:
        rows = []
        idx = 0

        if segment.segment_ID in self.segment_ids_added:
            return

        prevIdx = -1
        for point in segment.tracing_result:
            if len(point) == 2:  # (y, x)
                rows.append(
                    # idx, type, x, y, z, radius, prevIdx
                    [
                        idx,
                        TYPE,
                        point[1],
                        point[0],
                        "",
                        RADIUS,
                        prevIdx,
                    ]
                )
            else:  # (z, y, x)
                rows.append(
                    # idx, type, x, y, z, radius, prevIdx
                    [
                        idx,
                        TYPE,
                        point[2],
                        point[1],
                        point[0],
                        RADIUS,
                        prevIdx,
                    ]
                )

            idx += 1
            prevIdx += 1

        self.segment_ids_added.add(segment.segment_ID)

        for child in segment.children:
            if child.segment_ID in self.segment_ids_added:
                continue

            prevIdx = self._find_prevIdx(child, rows)
            logger.info(f"prefIdx: {prevIdx}")

            for point in child.tracing_result[1:]:
                if len(point) == 2:  # (y, x)
                    rows.append(
                        # idx, z, x, y, prevIdx
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
                        # idx, z, x, y, prevIdx
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

            self.segment_ids_added.add(child.segment_ID)

        return rows

    def _merge_segments(self) -> List["Segment"]:
        """
        1. Merges segments that form a continuous tracing
        (like A->B, B->C merged into A->C)
        2. otherwise returns disjoint segments
        (like A->B, B->C, X->Y, Y->Z merged into A->C is merged,
        X->Y remains same since that is disjoint)
        """
        if len(self.segments) == 1:
            return self.segments

        merged_segments = []
        # we can assume that our list is always sorted
        # i,e, (1, 5), (5, 6) not (5, 6), (1, 5)
        prev_segment = self.segments[0]
        prev_start = prev_segment.start_point
        prev_goal = prev_segment.goal_point
        merged_segments.append(prev_segment)

        for curr_segment in self.segments[1:]:
            curr_start = curr_segment.start_point
            curr_goal = curr_segment.goal_point

            if prev_goal == curr_start:
                extended_tracing_result = (
                    prev_segment.tracing_result + curr_segment.tracing_result
                )

                new_segment = Segment(
                    segment_ID=prev_segment.segment_ID,
                    start_point=prev_start,
                    goal_point=curr_goal,
                    tracing_result=extended_tracing_result,
                    children=prev_segment.children + curr_segment.children,
                )
                merged_segments[
                    -1
                ] = new_segment  # we merged a segment into the last segment

                prev_segment = new_segment
                prev_start = prev_segment.start_point
                prev_goal = prev_segment.goal_point
            else:
                # no merge, move ahead
                merged_segments.append(curr_segment)

                prev_segment = curr_segment
                prev_start = curr_start
                prev_goal = curr_goal

        return merged_segments

    def _find_prevIdx(self, child: Segment, rows: List[List]) -> int:
        start_point = child.start_point
        # logger.info(f"start_point: {start_point}")
        for row in rows:
            if len(start_point) == 2:
                # logger.info(f"Start point matches {row[1], row[2]}?")
                if row[1] == start_point[1] and row[2] == start_point[0]:
                    # logger.info("Match found")
                    return row[0]

            if len(start_point) == 3:
                if (
                    row[1] == start_point[2]
                    and row[2] == start_point[1]
                    and row[3] == start_point[0]
                ):
                    return row[0]
