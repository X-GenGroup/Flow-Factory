# src/flow_factory/rewards/sudoku.py
from accelerate import Accelerator
from typing import Optional, List, Union, Literal
from PIL import Image
import torch
import copy
import numpy as np

from .abc import BaseRewardModel, RewardModelOutput
from ..hparams import *


class SudokuRewardModel(BaseRewardModel):
    def __init__(
        self,
        config: RewardArguments,
        accelerator: Accelerator,
        ocr_backend: Literal["got", "paddle"] = "paddle",
    ):
        super().__init__(config, accelerator)
        self.ocr_backend = ocr_backend
        
        if ocr_backend == "got":
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                "stepfun-ai/GOT-OCR-2.0-hf", device_map=self.device, torch_dtype=torch.bfloat16
            )
            self.processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)
        else:  # paddle
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_angle_cls=False,
            )

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

    def _split_grid(self, img: Image.Image) -> List[Image.Image]:
        """Split 9x9 sudoku image into 81 cell images."""
        w, h = img.size
        cw, ch = w // 9, h // 9
        return [img.crop((j * cw, i * ch, (j + 1) * cw, (i + 1) * ch)) for i in range(9) for j in range(9)]

    @torch.no_grad()
    def _ocr_grids_got(self, images: List[Image.Image]) -> List[str]:
        """OCR using GOT-OCR model."""
        if not images:
            return []
        
        all_cells = [cell for img in images for cell in self._split_grid(img)]
        all_digits = []
        
        for i in range(0, len(all_cells), self.config.batch_size):
            batch = all_cells[i : i + self.config.batch_size]
            inputs = self.processor(batch, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=1,
            )
            texts = self.processor.batch_decode(
                generate_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            all_digits.extend(next((c for c in t if c.isdigit()), '0') for t in texts)
        
        return [''.join(all_digits[i * 81 : (i + 1) * 81]) for i in range(len(images))]

    def _ocr_grids_paddle(self, images: List[Image.Image]) -> List[str]:
        """OCR using PaddleOCR - process full grid and map by coordinates."""
        if not images:
            return []
        
        results = []
        for img in images:
            w, h = img.size
            cw, ch = w / 9, h / 9
            img_np = np.array(img.convert('RGB'))
            
            # OCR full image at once
            ocr_result = self.ocr.ocr(img_np, cls=False)
            
            # Initialize 9x9 grid with zeros
            grid = [['0'] * 9 for _ in range(9)]
            
            if ocr_result and ocr_result[0]:
                for item in ocr_result[0]:
                    box, (text, conf) = item[0], item[1]
                    # Get center of bounding box
                    cx = sum(p[0] for p in box) / 4
                    cy = sum(p[1] for p in box) / 4
                    # Map to grid position
                    col, row = int(cx // cw), int(cy // ch)
                    if 0 <= row < 9 and 0 <= col < 9:
                        digit = next((c for c in text if c.isdigit()), None)
                        if digit:
                            grid[row][col] = digit
            
            results.append(''.join(''.join(row) for row in grid))
        return results

    def _ocr_grids(self, images: List[Image.Image]) -> List[str]:
        """OCR dispatcher based on backend."""
        if self.ocr_backend == "got":
            return self._ocr_grids_got(images)
        return self._ocr_grids_paddle(images)

    def _to_grid(self, digits: str) -> List[List[int]]:
        return [[int(digits[i * 9 + j]) for j in range(9)] for i in range(9)]

    def _find_solution(self, puzzle: List[List[int]]) -> Optional[List[List[int]]]:
        grid = copy.deepcopy(puzzle)
        def is_valid(r, c, num):
            if num in grid[r] or num in [grid[i][c] for i in range(9)]:
                return False
            br, bc = 3 * (r // 3), 3 * (c // 3)
            return all(grid[br + i][bc + j] != num for i in range(3) for j in range(3))
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
        if num == 0:
            return True
        for j in range(9):
            if j != col and grid[row][j] == num:
                return False
        for i in range(9):
            if i != row and grid[i][col] == num:
                return False
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if (br + i, bc + j) != (row, col) and grid[br + i][bc + j] == num:
                    return False
        return True

    def _compute_reward(self, puzzle_str: str, solution_str: str) -> float:
        puzzle_grid = self._to_grid(puzzle_str)
        score = 0
        for idx in range(81):
            row, col = divmod(idx, 9)
            p, s = puzzle_str[idx], solution_str[idx]
            if p != '0':
                score -= 1 if s != p else 0
            elif s != '0':
                score += 20 if self._is_valid_placement(puzzle_grid, row, col, int(s)) else 5
        return float(score)

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        condition_images = [self._to_pil(cond_imgs) for cond_imgs in condition_images]
        puzzles = [cond[0] for cond in condition_images]
        
        puzzle_texts = self._ocr_grids(puzzles)
        solution_texts = self._ocr_grids(list(image))
        
        rewards = torch.tensor(
            [self._compute_reward(p, s) for p, s in zip(puzzle_texts, solution_texts)],
            device=self.device,
        )
        return RewardModelOutput(rewards=rewards, extra_info={})