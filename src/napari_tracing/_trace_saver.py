import csv
from typing import List

from ._segment_model import Segment


class TraceSaver:
    def __init__(self, filename: str, segments: List[Segment]) -> None:
        self.filename = filename
        self.segments = segments

    def save_trace(self):
        with open(self.filename, "w") as f:
            writer = csv.writer(f)
            column_headers = ["idx", "x", "y", "z", "prevIdx"]
            writer.writerow(column_headers)
            for row in self._get_row_values_for_saving_trace():
                writer.writerow(row)

    def _get_row_values_for_saving_trace(self) -> List:
        rows = []
        idx = 0
        merged_segments = self._merge_segments()
        for segment in merged_segments:
            prevIdx = -1
            result = segment.tracing_result
            for point in result:
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
        return rows

    def _merge_segments(self) -> List:
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
