# cotnav/models/vlms/pivot_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .infer_registry import get as get_infer

from cotnav.geometry.camera import (Calib, project_to_pixel)
from cotnav.geometry.motion import MotionTemplateLibrary
from typing import Iterable, Optional, TypedDict

ImgLike = Union[str, Path, Image.Image, np.ndarray]
Color = Tuple[int, int, int]

def create_pivot(**kwargs: Any) -> Any:
    return PIVOT(**kwargs)

""" BEGIN HELPER TYPEDEFS"""
# Enable later once we add more apis
# @dataclass
# class PivotVQAResult:
#     text: str
#     info: Dict[str, Any]

"""END HELPER TYPEDEFS"""

# Helper predicate
def is_send_token(msg: Dict[str, Any]) -> bool:
    """
    Return True if the provided message content indicates a send token.
    Accepts either a raw content dict or a ChatMessage wrapper.
    """
    if not isinstance(msg, dict):
        return False
    # If it's a full ChatMessage, inspect its first content element if present
    if "content" in msg and isinstance(msg["content"], list) and msg["content"]:
        c0 = msg["content"][0]
        if isinstance(c0, dict) and c0.get("type") == SEND_TOKEN:
            return True
    # Otherwise, treat as a direct token dict
    if msg.get("type") == SEND_TOKEN:
        return True
    return False

class PIVOT:
    """Minimal wrapper that forwards VQA calls to the configured VLM."""

    def __init__(self, *, vlm: Dict[str, Any], motion_parameters: Dict[str, Any]):
        self.vlm = self._init_vlm(vlm)

        generate_defaults = dict(vlm.get("generate_defaults", {}))
        self._default_instructions: Optional[str] = generate_defaults.pop("instructions", None)
        self._base_model_args: Dict[str, Any] = dict(vlm.get("model_args", {}))
        extra_model_args = generate_defaults.pop("model_args", {})
        if extra_model_args:
            self._base_model_args.update(extra_model_args)
        self._call_defaults: Dict[str, Any] = generate_defaults

        self._default_service_tier: Optional[str] = vlm.get("service_tier")
        self._default_timeout: Optional[float] = vlm.get("timeout")
        self._default_max_retries: int = int(vlm.get("max_retries", 5))

        # Keep a copy of the motion configuration for downstream helpers.
        self.mp_cfg = copy.deepcopy(motion_parameters or {})
        self._initial_motion_cfg = copy.deepcopy(self.mp_cfg)

    def motion_templates(self):
        mp_cfg = copy.deepcopy(self.mp_cfg)
        motion_bank = MotionTemplateLibrary(
            max_curvature=float(mp_cfg['max_curvature']),
            max_path_len=float(mp_cfg['max_free_path_length']),
            num_options=int(mp_cfg['num_options'])
        )
        return motion_bank.arcs()

    # -------------------- Public API --------------------

    def vqa(self, system_prompt: str, prompts: ChatThread, resume: bool = False, **call_kwargs: Any) -> PivotVQAResult:
        """Upload the image and run a single VLM call with the provided question."""  
        inputs = self.vlm.compile_prompt(prompts)
        response = self.vlm.generate_response(
            system_prompt,
            inputs,
            **call_kwargs
        )
        return response

    def annotate_goal_heading(
        self,
        image: ImgLike,
        heading_angle: float,
        # TODO write more args as needed
    ):
        """Annotates the image with a goal arrow on the top center that points in the heading angle direction to the goal"""
        # TODO Implement this.

    def annotate_constant_curvature(
        self,
        image: ImgLike,
        *,
        points_uv: Optional[Sequence[Tuple[float, float]]] = None, # [u,v] in pixels 
        arcs: Optional[Sequence[Any]] = None, # ConstantCurvatureArc objects
        calib: Optional[Calib] = None, # if arcs given, must provide calib
        # Style / options:
        selected_idx: Optional[int] = None,
        colors: Optional[Sequence[Color]] = None,
        thickness: int = 4,
        endpoint_radius: int = 20,
        overlay_alpha: float = 0.9,
        samples_per_meter: int = 10,
        # Constant height of the template in base frame (meters). Match your Convoi default.
        z_base: float = -0.4,
        label_text: bool = True,
        label_font_size: Optional[int] = 40,
        label_font_color: Optional[Color] = None,
        border_padding: int = 0,
    ) -> Tuple[Image.Image, List[Tuple[int, int]]]:
        """
        Draw numbered endpoint circles (like Convoi's annotate_image) and, if `arcs` are
        provided, draw the projected arc polylines from the robot to each endpoint.

        - If `arcs` is None: only numbered circles from `points_uv` are drawn.
        - If `arcs` is provided: arcs are sampled in base frame, projected using
          `project_xyz_to_uv(...)`, then rasterized as polylines on the image.

        Returns:
            annotated_image (PIL.Image RGB),
            endpoints_px    list of (u, v) ints (the circle centers actually drawn)
        """
        # --- Prepare base image + overlay ---
        base = self._to_pil(image).convert("RGB")
        W, H = base.size
        under = base.copy()
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        endpoints_px: List[Tuple[int, int]] = []
        font = self._get_label_font(label_font_size) if label_text else None
        if label_font_color:
            rgb_vals = tuple(int(c) for c in label_font_color)
            font_rgb = rgb_vals[:3] if len(rgb_vals) >= 3 else (255, 255, 255)
        else:
            font_rgb = (255, 255, 255)

        # --- If arcs are given, draw them (requires projection) ---
        if arcs is not None and len(arcs) > 0:
            if calib is None:
                raise ValueError("If arcs are given, must also provide calib.")

            # Prepare colors
            cols = list(colors) if colors else None

            for i, arc in enumerate(arcs):
                # sample along arc in base frame (meters)
                n = max(2, int(samples_per_meter * float(arc.length)))
                s_vals = np.linspace(0.0, float(arc.length), n)
                xy_local = np.array([arc.xy_at_s(float(s)) for s in s_vals], dtype=np.float32)  # (N,2)
                xyz = np.concatenate([xy_local, np.full((n, 1), z_base, np.float32)], axis=1)   # (N,3)

                # project to image
                uv_all, vis_mask = project_to_pixel(xyz, calib)

                # draw line segments only where both endpoints are visible & inside image
                color_i = cols[i % len(cols)] if cols else (0, 255, 0)
                for j in range(len(uv_all) - 1):
                    if vis_mask[j] and vis_mask[j + 1]:
                        u0, v0 = uv_all[j]
                        u1, v1 = uv_all[j + 1]
                        draw.line([(u0, v0), (u1, v1)], fill=(*color_i, 255), width=thickness)

                # remember the final visible endpoint (fallback to last sample)
                # if caller also supplies points_uv, we'll use those for circle centers instead
                valid_idx = np.flatnonzero(vis_mask)
                if valid_idx.size > 0:
                    u_end, v_end = uv_all[valid_idx[-1]]
                else:
                    u_end, v_end = uv_all[-1]
                endpoints_px.append((int(round(u_end)), int(round(v_end))))

        # --- Determine circle centers: prefer provided points_uv if given ---
        if points_uv is not None and len(points_uv) > 0:
            circle_centers = [(int(round(u)), int(round(v))) for (u, v) in points_uv]
        else:
            circle_centers = endpoints_px

        # --- Draw numbered circles (Convoi style) ---
        circle_color_default = (255, 255, 255)
        circle_color_selected = (0, 255, 0)
        pad = max(0, int(border_padding))
        min_x = pad + endpoint_radius
        min_y = pad + endpoint_radius
        max_x = max(min_x, W - 1 - pad - endpoint_radius)
        max_y = max(min_y, H - 1 - pad - endpoint_radius)

        adjusted_centers: List[Tuple[int, int]] = []

        for i, (cx, cy) in enumerate(circle_centers):
            sel = (selected_idx is not None and i == int(selected_idx))
            color = circle_color_selected if sel else circle_color_default

            cx = max(min_x, min(max_x, cx))
            cy = max(min_y, min(max_y, cy))
            adjusted_centers.append((cx, cy))

            # filled circle
            draw.ellipse(
                [cx - endpoint_radius, cy - endpoint_radius,
                 cx + endpoint_radius, cy + endpoint_radius],
                fill=(*color, 255),
                outline=None
            )

            if label_text:
                label = str(i)
                # center text crudely using textbbox
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = (8, 10)
                tx = cx - int(tw / 2)
                ty = cy - int(th / 2)
                # black outline then white text for legibility
                draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 255), font=font)
                draw.text((tx, ty), label, fill=(*font_rgb, 255), font=font)

        # --- Composite like cv2.addWeighted(alpha=0.3) ---
        comp = Image.alpha_composite(under.convert("RGBA"), overlay)
        if 0.0 < overlay_alpha < 1.0:
            comp = Image.blend(under.convert("RGBA"), comp, overlay_alpha)

        # Use whichever centers we actually drew for return value
        ret_centers = adjusted_centers if len(adjusted_centers) > 0 else endpoints_px
        return comp.convert("RGB"), ret_centers

    # -------------------- Context management --------------------

    def get_context(self) -> Dict[str, Any]:
        """Return the current motion parameter configuration."""
        return {"motion_parameters": copy.deepcopy(self.mp_cfg)}

    def reset_context(self) -> None:
        """Restore the motion parameters to their initial state."""
        self.mp_cfg = copy.deepcopy(self._initial_motion_cfg)

    def restore_context(self, context: Dict[str, Any]) -> None:
        """Restore motion parameters from a previously captured context."""
        if not isinstance(context, dict):
            raise TypeError("context must be a dict produced by get_context().")
        motion_cfg = context.get("motion_parameters")
        if motion_cfg is not None:
            self.mp_cfg = copy.deepcopy(motion_cfg)

    # -------------------- Internals --------------------

    def _init_vlm(self, cfg: Dict[str, Any]) -> Any:
        """
        Accepts:
          - {"instance": <vlm>}
          - {"factory": callable, "kwargs": {...}}
          - {"name": "openai:gpt5", "provider_kwargs": {...}}  # if infer_registry available
        """
        if "instance" in cfg and cfg["instance"] is not None:
            return cfg["instance"]

        if "factory" in cfg and callable(cfg["factory"]):
            return cfg["factory"](**dict(cfg.get("kwargs", {})))

        if "name" in cfg and get_infer is not None:
            return get_infer(cfg["name"], **dict(cfg.get("provider_kwargs", {})))

        raise ValueError(
            "vlm config must provide one of: "
            "'instance', 'factory'+optional 'kwargs', or 'name'+optional 'provider_kwargs'."
        )

    @staticmethod
    def _to_pil(image: ImgLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, Path)):
            p = str(image)
            if p.startswith("http://") or p.startswith("https://"):
                raise ValueError("Cannot convert URL images to PIL directly; download first.")
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _get_label_font(self, size: Optional[int]) -> Optional[ImageFont.ImageFont]:
        if size is None:
            try:
                return ImageFont.load_default()
            except Exception:
                return None
        try:
            return ImageFont.truetype("DejaVuSans.ttf", int(size))
        except Exception:
            try:
                return ImageFont.load_default()
            except Exception:
                return None
