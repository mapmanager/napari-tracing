from typing import List, Set, Tuple

import numpy as np

from ._logger import logger


class Utils:
    def are_sets_equal(a: Set, b: Set):
        """
        Compare if two sets are equal or not
        """
        if a is None or b is None:
            return False

        if len(a) != len(b):
            return False

        for x in a:
            if x not in b:
                return False

        return True

    def get_diff(
        layer1_data: np.ndarray, layer2_data: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        - Get difference between two dataframes
        - this method assumes layer1_data is larger than layer2_data
        """
        ptr1, ptr2 = 0, 0
        differing_indices = []
        diff_data = []

        while ptr1 < len(layer1_data) and ptr2 < len(layer2_data):
            diff = layer1_data[ptr1] - layer2_data[ptr2]
            if diff.any():  # if there is any difference between the two values
                differing_indices.append(ptr1)
                diff_data.append((layer1_data[ptr1]).tolist())
                ptr1 += 1
            else:  # the two values are equal, proceed
                ptr1 += 1
                ptr2 += 1

        while ptr1 < len(layer1_data):
            differing_indices.append(ptr1)
            diff_data.append((layer1_data[ptr1]).tolist())
            ptr1 += 1

        res = (differing_indices, np.array(diff_data))
        logger.info(res)
        return res

    def get_idx(target: np.ndarray, layer_data: np.ndarray) -> int:
        idx = len(layer_data) - 1
        while idx > 0:
            diff = layer_data[idx] - target
            if diff.any():
                idx -= 1
            else:
                return idx
