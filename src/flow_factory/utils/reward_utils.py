# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from itertools import permutations

# src/flow_factory/utils/reward_utils.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


# -------------------------------------Grid Utils-------------------------------------
def divide_prompt(prompt: str) -> List[str]:
    # Separators look like ". [TOP-LEFT]:" or 'xxx." [BOTTOM-RIGHT]:'
    match_sep = re.compile(r"[\.\"]\s+[A-Z0-9-\[\]]+:")
    seps = match_sep.findall(prompt)
    # Add '.' for each sentence
    sub_prompts = [
        p + "." if p.strip()[-1] != "." else p
        for p in re.split("|".join(map(re.escape, seps)), prompt)
    ]
    return sub_prompts


def divide_image(image, grid_info: tuple[int, int]) -> List[Image.Image]:
    assert len(grid_info) == 2, "grid_info must be a tuple of two integers (a, b)"

    a, b = grid_info
    width, height = image.size

    grid_cells = []
    cell_height = height // a
    cell_width = width // b

    # 2x2 grid
    # | 1 | 2 |
    # | 3 | 4 |
    # [
    # (0, 0, cell_width, cell_height),
    # (cell_width, 0, 2 * cell_width, cell_height),
    # (0, cell_height, cell_width, 2 * cell_height),
    # (cell_width, cell_height, 2 * cell_width, 2 * cell_height)
    # ]

    for i in range(a):
        for j in range(b):
            upper = i * cell_height
            left = j * cell_width
            right = left + cell_width
            lower = upper + cell_height
            grid_cells.append(image.crop((left, upper, right, lower)))

    return grid_cells


def extract_grid_info(prompt: str) -> tuple[int, int]:
    # Grid can be represented as int x int, or int ⨉ int. ⨉ has unicode \u2a09
    match = re.findall(r"(\d+)\s*[x⨉]\s*(\d+)", prompt)
    if len(match) == 0:
        return (1, 1)

    return (int(match[0][0]), int(match[0][1]))


# -------------------------------------Reward Computation Utils-------------------------------------
def is_symmetric_matrix(matrix: np.ndarray) -> bool:
    """Check if the matrix is symmetric.

    Args:
        matrix: Square numpy array.

    Returns:
        True if symmetric, False otherwise.
    """
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        # Must be square
        return False

    return np.all(matrix == matrix.T)


def is_antisymmetric_matrix(matrix: np.ndarray, diagonal_zero=True) -> bool:
    """Check if the matrix is anti-symmetric.

    Args:
        matrix: Square numpy array.
        diagonal_zero: If True, also require zero diagonal elements; otherwise ignore the
            diagonal.

    Returns:
        True if anti-symmetric, False otherwise.
    """
    matrix = np.array(matrix)
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        # Must be square
        return False

    summation = matrix.T + matrix
    if diagonal_zero:
        # Check if all elements are zero
        return np.all(summation == 0)
    else:
        # Assign diagonal to zero and check
        summation[np.diag_indices_from(summation)] = 0
        if np.any(summation != 0):
            return False

    return True


def is_transitive_matrix(
    matrix: np.ndarray, return_violations=False
) -> Union[bool, tuple[bool, List[tuple[int, int, int]]]]:
    """Check if the binary relation matrix is transitive.

    Args:
        matrix: Square numpy array with binary values (0 or 1).
        return_violations: If True, also return every violating ``(i, j, k)`` triple.

    Returns:
        True if transitive, False otherwise. With ``return_violations``, a square matrix
        returns ``(is_transitive, violations)``.

    Raises:
        ValueError: If the matrix is not binary.
    """
    matrix = np.array(matrix)
    n = len(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        # Must be square
        return False

    if not np.all(np.isin(matrix, [0, 1])):
        # Must be binary
        raise ValueError("`transitiveMatrixQ` requires matrix must be binary (0 or 1)")

    # Check transitivity: if A[i][j] == 1 and A[j][k] == 1, then A[i][k] must be 1
    violations = []
    for i, j, k in permutations(range(n), 3):
        # Check all 3-tuples
        if matrix[i][j] == 1 and matrix[j][k] == 1 and matrix[i][k] != 1:
            if not return_violations:
                return False

            violations.append((i, j, k))

    if return_violations:
        return len(violations) == 0, violations

    return len(violations) == 0
