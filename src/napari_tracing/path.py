from typing import List, Tuple

import numpy as np


class PathModel:
    def __init__(
        self,
        start_point: Tuple,
        goal_point: Tuple,
        tracing_result: List[np.ndarray] = [],
    ) -> None:
        self.start_point = start_point
        self.goal_point = goal_point
        self.tracing_result = tracing_result
