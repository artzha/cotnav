# openaimodel.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union
import os, time, random

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

# ----- Types -----
Part = Dict[str, Any]                 # {"type":"input_text",...} | {"type":"input_image",...} | {"type":"input_file",...}
Prompt = Union[str, Sequence[Part]]   # plain string OR list of parts (matches your ResponsesMessage helpers)

class OpenAIModel:
    """
    Minimal Responses-API wrapper that mirrors your prior pattern:
      response = client.responses.create(**model_args, instructions=..., input=..., service_tier=...)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 180.0,
        service_tier: Optional[str] = None,  # e.g., "flex"
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.timeout = timeout
        self.service_tier = service_tier

    # ---------------- Sync core ----------------

    def call(
        self,
        *,
        model_args: Dict[str, Any],
        instructions: Optional[str],
        input: List[Dict[str, Any]],
        max_retries: int = 5,
        timeout: Optional[float] = None,
        service_tier: Optional[str] = None,
    ):
        """
        Raw robust call. Returns the full Response object.
        `model_args` is passed through verbatim (e.g., {"model":"gpt-5", "reasoning":{...}, "tools":[...], ...}).
        """
        body = dict(model_args)
        body["instructions"] = instructions
        body["input"] = input
        body["service_tier"] = service_tier or self.service_tier

        last_err: Optional[Exception] = None
        for i in range(max_retries):
            try:
                return self.client.with_options(timeout=timeout or self.timeout).responses.create(**body)
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_err = e
                backoff = min(16 * (2 ** i), 60) + random.uniform(0.5, 2.0)
                print(f"[openai] transient error ({i+1}/{max_retries}): {e} — retrying in {backoff:.1f}s")
                time.sleep(backoff)
        raise last_err if last_err else RuntimeError("OpenAI call failed without exception")

    def call_text(
        self,
        *,
        model_args: Dict[str, Any],
        instructions: Optional[str],
        input: List[Dict[str, Any]],
        max_retries: int = 5,
        timeout: Optional[float] = None,
        service_tier: Optional[str] = None,
        auto_bump_tokens: bool = True,
        max_bump_cap: int = 8192,
    ) -> str:
        """
        Convenience: returns `response.output_text`. If the response is incomplete due to
        `max_output_tokens`, optionally retries once with a larger cap.
        """
        # First attempt
        resp = self.call(
            model_args=model_args,
            instructions=instructions,
            input=input,
            max_retries=max_retries,
            timeout=timeout,
            service_tier=service_tier,
        )
        txt = (getattr(resp, "output_text", None) or "").strip()
        status = getattr(resp, "status", None)
        inc = getattr(resp, "incomplete_details", None)
        reason = getattr(inc, "reason", None) if inc else None

        if status == "completed" and txt:
            return txt

        # Auto-bump once if we ran out of output tokens
        if auto_bump_tokens and status == "incomplete" and reason == "max_output_tokens":
            bumped_args = dict(model_args)
            current_max = int(bumped_args.get("max_output_tokens") or bumped_args.get("max_completion_tokens") or 512)
            bumped_args["max_output_tokens"] = min(current_max * 2, max_bump_cap)

            resp2 = self.call(
                model_args=bumped_args,
                instructions=instructions,
                input=input,
                max_retries=max_retries,
                timeout=timeout,
                service_tier=service_tier,
            )
            txt2 = (getattr(resp2, "output_text", None) or "").strip()
            if getattr(resp2, "status", None) == "completed" and txt2:
                return txt2

        # Fall back: return whatever text we have (possibly empty) to avoid hiding signal
        return txt

    # -------------- Small conveniences --------------

    @staticmethod
    def to_messages(prompt: Prompt) -> List[Dict[str, Any]]:
        """
        Optional helper: build Responses-API messages from a plain string or list of parts.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        if isinstance(prompt, (list, tuple)):
            parts: List[Part] = []
            for part in prompt:
                if isinstance(part, str):
                    parts.append({"type": "input_text", "text": part})
                elif isinstance(part, dict):
                    parts.append(part)
                else:
                    raise TypeError(f"Unsupported prompt part: {type(part)}")
            return [{"role": "user", "content": parts}]
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    def upload_file(self, path: str, purpose: str = "user_data") -> str:
        f = self.client.files.create(file=open(path, "rb"), purpose=purpose)
        return f.id
