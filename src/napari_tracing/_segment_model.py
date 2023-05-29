from typing import List, Tuple

import numpy as np


class Segment:
    def __init__(
        self,
        segment_ID: int,
        start_point: Tuple,
        goal_point: Tuple,
        tracing_result: List[np.ndarray] = [],
        tracing_algorithm: str = "A* Search",
        children: List["Segment"] = [],
    ) -> None:
        self.segment_ID = segment_ID
        self.start_point = start_point
        self.goal_point = goal_point
        self.tracing_result = tracing_result
        self.tracing_algorithm = tracing_algorithm
        self.children = children

    def add_child(self, child: "Segment") -> None:
        self.children.append(child)


# class Tree:
#     def __init__(self, node: Segment, left: Segment, right: Segment) -> None:
#         self.node = node
#         self.left = left
#         self.right = right
