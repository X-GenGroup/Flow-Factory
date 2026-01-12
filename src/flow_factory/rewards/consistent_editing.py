
import os
import re
import json
from socket import timeout
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
from itertools import combinations
import math
import time

import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
from PIL import Image

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

def pil_image_to_base64(image : Image.Image, format="JPEG") -> str:
    """
        Convert a PIL Image to a base64-encoded string.
        Args:
            image (Image.Image): PIL Image object
            format (str): Image format, e.g., "JPEG", "PNG"
        Returns:
            base64 string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_image = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_image

def get_yes_cond_prob_from_completion(completion, canonicalize=False) -> float:
    """
        Extract the conditional probability of "yes" from an OpenAI ChatCompletion response.
        Args:
            completion (openai.ChatCompletion): The completion response from OpenAI API.
            canonicalize (bool): If True, aggregate probabilities for all case variations of "yes" and "no".
        Returns:
            float: The conditional probability of "yes". Returns 0.0 if "yes" or "no" cannot be determined.
    """
    if completion is None:
        return 0.0

    logprobs = completion.choices[0].logprobs
    if logprobs:
        # Use logprobs to compute, score = P('yes') / (P('yes') + P('no'))
        # score = 1 / (1 + exp(logprob('no') -  logprob('yes')))
        # Same formular for logits as well. Since the sum term will cancel out.
        # Use uppercase only here.
        if not canonicalize:
            token_logprobs = {t.token: t.logprob for t in logprobs.content[0].top_logprobs}
            yes_logprob = token_logprobs.get('Yes', float('-inf'))
            no_logprob = token_logprobs.get('No', float('-inf'))
            if yes_logprob == float('-inf') and no_logprob == float('-inf'):
                # When inf - inf encountered, give 0.0 score.
                yes_cond_prob = 0.0 # 0.0
            else:
                diff = torch.tensor(yes_logprob - no_logprob, dtype=torch.float64)
                yes_cond_prob = torch.sigmoid(diff).item()
        else:
            # Sum all possible cases together
            # 'yes', 'Yes', 'YES', 'yes ',....
            # 'no', 'No', 'NO',....
            token_probs = {t.token: np.exp(t.logprob, dtype=np.float64) for t in logprobs.content[0].top_logprobs}
            
            # Vectorized computation
            tokens = np.array(list(token_probs.keys()))
            probs = np.array(list(token_probs.values()))
            
            # Strip and lower the tokens for matching
            tokens_stripped = np.array([token.strip().lower() for token in tokens])
            
            yes_mask = tokens_stripped == "yes"
            no_mask = tokens_stripped == "no"
            
            yes_prob_sum = probs[yes_mask].sum()
            no_prob_sum = probs[no_mask].sum()
            
            total = yes_prob_sum + no_prob_sum

            if total == 0.0:
                yes_cond_prob = 0.0
            else:
                yes_cond_prob = yes_prob_sum / total
    else:
        # log_prob cannot be derived here. Return 0.0.
        yes_cond_prob = 0.0

    return yes_cond_prob

class ConsistentEditingRewardModel:
    """
    Consistent Editing Reward Model using OpenAI API.
    This model evaluates the consistency of edited images with respect to the original images
    based on given prompts. It computes two scores: `image consistency` and `prompt-following accuracy`,
    and combines them to produce a final reward score.
    """
    def __init__(
            self,
            client: AsyncOpenAI,
            model='PaCo-Reward-7B',
            max_concurrent=100,
            max_retries=10,
            timeout=60,
            max_cache_size=1024,
        ):
        self.client = client
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.global_semaphore = asyncio.Semaphore(self.max_concurrent)
        self.max_cache_size = max_cache_size if max_cache_size is not None else math.inf

    def __call__(
            self,
            prompt : List[str],
            image: List[Image.Image],
            condition_images: Union[List[Image.Image], List[List[Image.Image]]]=None,
            canonicalize: bool = False
        ) -> List[float]:
        return asyncio.run(self.__async_call__(prompt, image, condition_images, canonicalize))

    @torch.no_grad()
    async def __async_call__(
        self,
        prompt : List[str],
        image : List[Image.Image],
        condition_images: Union[List[Image.Image], List[List[Image.Image]]],
        canonicalize: bool = False
    ) -> List[float]:
        if condition_images is None:
            condition_images = [None] * len(image)

        assert len(prompt) == len(image) == len(condition_images), \
            "Length of prompt, image, and condition_images must be the same."
        
        async def process_single_image(prompt, image, cond_images):
            consistency_text_prompt = (
                f"Compare the edited image (second) with the original image (first). "
                f"Instruction: '{prompt}'. "
                f"Except for the parts that are intentionally changed according to the instruction, "
                f"does the edited image remain consistent with the original in style, logic, and identity? "
                f"Answer 'Yes' or 'No' first, then provide detailed reasons."
            )
            prompt_following_text_prompt = (
                f"Compare the edited image (second) with the original image (first). "
                f"Instruction: '{prompt}'. "
                f"Does the edited image accurately follow this instruction? "
                f"Answer 'Yes' or 'No' first, then provide detailed reasons."
            )
            consistency_score = await self._async_compute_image_consistency(
                criteria_text=consistency_text_prompt,
                condition_images=cond_images,
                image=image,
                canonicalize=canonicalize
            )
            prompt_following_score = await self._async_compute_image_consistency(
                criteria_text=prompt_following_text_prompt,
                condition_images=cond_images,
                image=image,
                canonicalize=canonicalize
            )
            # Combine the two scores (e.g., geometric mean following EditScore)
            combined_score = math.sqrt(consistency_score * prompt_following_score)
            return combined_score
        
        tasks = [
            process_single_image(p, i, c)
            for p, i, c in zip(prompt, image, condition_images)
        ]
        scores = await asyncio.gather(*tasks)
        return scores

    async def _async_compute_image_consistency(
            self,
            criteria_text: str,
            condition_images: Union[Image.Image, List[Image.Image]],
            image: Image.Image,
            top_logprobs: int = 20,
            canonicalize: bool = False
        ) -> float:
        """
        Async version of compute_image_consistency with concurrency control.
        """

        content = []

        if condition_images is not None:
            if not isinstance(condition_images, list):
                condition_images = [condition_images]
            content +=  [
                {"type": "image_url", "image_url": {"url": pil_image_to_base64(cond_img)}}
                for cond_img in condition_images
            ]

        content.append(
            {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}}
        )
        content.append(
            {"type": "text", "text": criteria_text}
        )
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        for attempt in range(self.max_retries):
            try:
                async with self.global_semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0, # Deterministic result, no use for logprobs, actually.
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        timeout=self.timeout
                    )

                score = get_yes_cond_prob_from_completion(completion, canonicalize=canonicalize)
                # self.add_to_cache(cache_key, score)
                break

            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    score = 0.0  # Default score on failure        
        return score    

