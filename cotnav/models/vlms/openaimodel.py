# openaimodel.py
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union
import os, time, random

import io
import PIL
from PIL import Image
from pathlib import Path
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError
import tempfile, numpy as np, os
from enum import Enum

from cotnav.utils.log_utils import logging

# ----- Types -----
# Part = Dict[str, Any]                 # {"type":"input_text",...} | {"type":"input_image",...} | {"type":"input_file",...}
# Message = Dict[str, Any]
# Prompt = Union[str, Sequence[Part]]   # plain string OR list of parts


# Canonical content type enum
class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"

# Common roles enum
class Role(str, Enum):
    USER = "user"
    DEVELOPER = "developer"
    SYSTEM = "system"
    ASSISTANT = "assistant"

@dataclass
class ChatQuery:
    type: str = ContentType.TEXT.value
    role: str = Role.USER.value
    content: Any = field(default_factory=str)

    def __post_init__(self) -> None:
        # Normalize simple content values into a list for consistent downstream use.
        if isinstance(self.content, (str, dict)):
            self.content = self.content

    def as_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "role": self.role, "content": self.content}

class ResponsesMessage:
    def input_text_message(self, text):
        return {
            "type": "input_text",
            "text": text
        }
    def file_message(self, file_data, filename):
        return {
            "type": "input_file",
            "file_data": file_data,
            "filename": filename
        }
    def input_file_id(self, file_id):
        return {
            "type": "input_image",
            "file_id": file_id
        }
    def output_text_message(self, text):
        return {
            "type": "output_text",
            "text": text
        }

RM = ResponsesMessage()

def get_openai_cost(
    model_name, input_tokens=0, cached_tokens=0, output_tokens=0
):
    """Get the cost of an OpenAI response."""
    # Price table: https://openai.com/api/pricing/
    # Prices are per 1M tokens, as of July 12, 2025.
    if model_name == "o3":
        price_input_tokens = 2.0
        price_cached_tokens = 0.5
        price_output_tokens = 8.0
    elif model_name == "o3-mini":
        price_input_tokens = 1.1
        price_cached_tokens = 0.55
        price_output_tokens = 4.4
    elif model_name == "o4-mini":
        price_input_tokens = 1.1
        price_cached_tokens = 0.275
        price_output_tokens = 4.4
    elif model_name == "gpt-5":
        price_input_tokens = 1.25
        price_cached_tokens = 0.125
        price_output_tokens = 10.0
    else:
        raise ValueError(f"Model {model_name} pricing not known")

    input_cost = (input_tokens / 1_000_000) * price_input_tokens
    cached_cost = (cached_tokens / 1_000_000) * price_cached_tokens
    output_cost = (output_tokens / 1_000_000) * price_output_tokens

    total_cost = input_cost + cached_cost + output_cost

    return total_cost, {
        "input_cost": input_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost
    }

class OpenAIModel:
    """
    Minimal Responses-API wrapper.
    Now also owns:
      - default_model_args (persisted per instance)
      - preprocess(...) to build messages
      - generate(messages, **kwargs) to return output_text
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        service_tier: Optional[str] = None,  # e.g., "flex"
        default_model_args: Optional[Dict[str, Any]] = None,
        default_role: str = "user",
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.timeout = timeout
        self.service_tier = service_tier
        self.default_model_args = default_model_args or {}
        self._default_role = default_role

    # -------------- Owned conveniences --------------

    def set_default_model_args(self, **updates: Any) -> None:
        self.default_model_args.update({k: v for k, v in updates.items() if v is not None})

    def preprocess_image(self, img):
        assert isinstance(img, Image.Image), "Only supports preprocess PIL.Image for now" 
        max_dim = 480
        w, h = img.size
        largest = max(w, h)
        if largest > max_dim:
            scale = max_dim / float(largest)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = img.resize(new_size, resample=Image.LANCZOS)
        return img

    def format_content(self, prompt):
        if prompt.type == ContentType.TEXT:
            # Use output_text for assistant turns, input_text otherwise
            if str(prompt.role) == Role.ASSISTANT.value or str(prompt.role) == "assistant":
                return RM.output_text_message(prompt.content)
            else:
                return RM.input_text_message(prompt.content)

        elif prompt.type == ContentType.IMAGE:
            # images are user inputs; upload & reference by file_id
            image = self.preprocess_image(prompt.content)
            file_id = self.upload_file(image)
            return RM.input_file_id(file_id)

        else:
            return None

    def compile_prompt(self, prompts: ChatThread):
        messages = []
        for prompt in prompts:
            part = self.format_content(prompt)
            assert part is not None, "Missing part in compile_prompt()"
            messages.append({"role": prompt.role, "content": [part]})
        return messages

    def generate_response(self, instructions: str, inputs: List[str], **kwargs: Any) -> str:
        """
        One-call text generation using this instance's defaults.
        Per-call overrides:
          - instructions
          - model_args={...}  (merged into this instance's defaults for THIS CALL only)
          - timeout, service_tier, max_retries, auto_bump_tokens, max_bump_cap
          - any Responses params (e.g., max_output_tokens, stop_sequences, temperature/top_p if supported)
        """
        model_args = kwargs.pop("model_args", self.default_model_args)
        assert model_args is not None, "Model args were not provided"

        instructions  = kwargs.pop("instructions", None)
        timeout       = kwargs.pop("timeout", self.timeout)
        service_tier  = kwargs.pop("service_tier", None)
        max_retries   = int(kwargs.pop("max_retries", 3))
        auto_bump     = kwargs.pop("auto_bump_tokens", True)
        max_bump_cap  = int(kwargs.pop("max_bump_cap", 8192))

        for i in range(max_retries):
            try:
                response = self.client.with_options(timeout=timeout).responses.create(
                    **model_args,
                    instructions=instructions,
                    input=inputs,
                    service_tier=service_tier
                )
                return response
            except Exception as e:
                logging.warning(f"Error in response {i+1}/{max_retries}: {e}")
                import pdb; pdb.set_trace()
                sleep_time = 16 * (2**i) + random.uniform(1, 2)
                logging.info(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                logging.info(f"Retrying...")

            
        raise Exception(f"Failed to get response after {max_retries} retries")

    def upload_file(self, file: Any, purpose: str = "user_data") -> str:
        if isinstance(file, str) or isinstance(file, Path):
            buf = open(str(file), "rb")
        else:
            buf = _img_to_buf(file)

        f = self.client.files.create(file=buf, purpose=purpose)
        return f.id

    def delete_file(self, file_id: str) -> None:
        """Best-effort removal for files previously uploaded via upload_file."""
        try:
            self.client.files.delete(file_id)
        except Exception as exc:
            # Keep failures non-fatal so cleanup never blocks inference flows.
            print(f"[openai] failed to delete file {file_id}: {exc}")

def _img_to_buf(img):
    if isinstance(img, np.ndarray):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img, mode="RGB")

    image_format = "PNG"
    buf = io.BytesIO()
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
    elif isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img)).convert("RGB")
    else:
        try:
            img = Image.fromarray(np.asarray(img)).convert("RGB")
        except Exception:
            raise ValueError("Unsupported image type for _img_to_buf")
    img.save(buf, format=image_format)
    buf.seek(0)
    buf.name = f"image.{image_format.lower()}"
    return buf