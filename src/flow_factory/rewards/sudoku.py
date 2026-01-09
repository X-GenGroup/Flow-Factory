# src/flow_factory/rewards/sudoku.py
from accelerate import Accelerator
from typing import Optional, List, Union
from PIL import Image
import torch
import copy
import numpy as np

from transformers import AutoProcessor, AutoModelForImageTextToText

from .abc import BaseRewardModel, RewardModelOutput
from ..hparams import *


class SudokuRewardModel(BaseRewardModel):
    def __init__(self, reward_args: RewardArguments, accelerator: Accelerator):
        super().__init__(reward_args, accelerator)
        self.model = AutoModelForImageTextToText.from_pretrained(
            "stepfun-ai/GOT-OCR-2.0-hf", device_map=self.device, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

    def _to_pil(self, img: Union[Image.Image, torch.Tensor, np.ndarray, List]) -> List[Image.Image]:
        """Convert tensor/ndarray/PIL to list of PIL Images."""
        if isinstance(img, list):
            return sum([self._to_pil(x) for x in img], [])
        if isinstance(img, Image.Image):
            return [img]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = img[None, ..., None]
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
                img = np.transpose(img, (1, 2, 0))
            img = img[None]
        elif img.ndim == 4 and img.shape[1] in (1, 3, 4) and img.shape[1] < img.shape[-1]:
            img = np.transpose(img, (0, 2, 3, 1))
        vmin, vmax = img.min(), img.max()
        if vmin >= -1.0 and vmax <= 1.0 and vmin < 0:
            img = (img + 1) * 127.5
        elif vmax <= 1.0:
            img = img * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return [Image.fromarray(x.squeeze(-1) if x.shape[-1] == 1 else x) for x in img]

    @torch.no_grad()
    def _batch_ocr(self, images: List[Image.Image]) -> List[str]:
        """Batch OCR whole images using GOT-OCR-2.0."""
        results = []
        for i in range(0, len(images), self.reward_args.batch_size):
            batch = images[i : i + self.reward_args.batch_size]
            inputs = self.processor(batch, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=256,
            )
            texts = self.processor.batch_decode(
                generate_ids[:, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            results.extend(texts)
        return results

    def _extract_digits(self, text: str) -> str:
        """Extract digits from OCR result, pad/truncate to 81 chars."""
        digits = ''.join(ch for ch in text if ch.isdigit())
        return digits.ljust(81, '0')[:81]

    def _to_grid(self, digits: str) -> List[List[int]]:
        """Convert 81-char string to 9x9 grid."""
        return [[int(digits[i * 9 + j]) for j in range(9)] for i in range(9)]

    def _grid_to_str(self, grid: List[List[int]]) -> str:
        """Convert 9x9 grid to 81-char string."""
        return ''.join(str(grid[i][j]) for i in range(9) for j in range(9))

    def _find_solution(self, puzzle: List[List[int]]) -> Optional[List[List[int]]]:
        """Backtracking solver, returns first solution or None."""
        grid = copy.deepcopy(puzzle)
        
        def is_valid(r, c, num):
            if num in grid[r]: return False
            if num in [grid[i][c] for i in range(9)]: return False
            br, bc = 3 * (r // 3), 3 * (c // 3)
            return all(grid[br+i][bc+j] != num for i in range(3) for j in range(3))
        
        def solve():
            for i in range(9):
                for j in range(9):
                    if grid[i][j] == 0:
                        for num in range(1, 10):
                            if is_valid(i, j, num):
                                grid[i][j] = num
                                if solve():
                                    return True
                                grid[i][j] = 0
                        return False
            return True
        
        return grid if solve() else None

    def _is_valid_placement(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) violates sudoku rules."""
        if num == 0:
            return True
        # Check row
        for j in range(9):
            if j != col and grid[row][j] == num:
                return False
        # Check column
        for i in range(9):
            if i != row and grid[i][col] == num:
                return False
        # Check 3x3 box
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if (br + i, bc + j) != (row, col) and grid[br + i][bc + j] == num:
                    return False
        return True

    def _compute_reward(self, puzzle_str: str, solution_str: str) -> float:
        """
        Compute reward based on diff:
        - Delete/modify original digit: -1
        - Add non-conflicting digit: +20
        - Add conflicting digit: +5 - encourages exploration
        """
        p = self._extract_digits(puzzle_str)
        s = self._extract_digits(solution_str)
        puzzle_grid = self._to_grid(p)
        solution_grid = self._to_grid(s)
        
        score = 0
        for idx in range(81):
            row, col = divmod(idx, 9)
            if p[idx] != '0':  # Original cell
                if s[idx] != p[idx]:
                    score -= 1  # Deleted or modified
            else:  # Empty cell
                if s[idx] != '0':
                    if self._is_valid_placement(puzzle_grid, row, col, int(s[idx])):
                        score += 20  # Valid addition
                    else:
                        score += 5  # Conflicting addition - encourages exploration
        
        return float(score)

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        condition_images = [self._to_pil(cond_imgs) for cond_imgs in condition_images]
        batch_size = len(prompt)
        
        puzzles = [cond[0] for cond in condition_images]
        solutions = list(image)
        
        # Batch OCR all images
        all_texts = self._batch_ocr(puzzles + solutions)
        puzzle_texts = all_texts[:batch_size]
        solution_texts = all_texts[batch_size:]
        
        # Compute rewards
        rewards = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            rewards[i] = self._compute_reward(puzzle_texts[i], solution_texts[i])
        
        return RewardModelOutput(rewards=rewards, extra_info={})