# cotnav/models/vlms/pivot_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy
from pathlib import Path
import tempfile
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .infer_registry import get as get_infer

from cotnav.models.vlms.interface import (ChatQuery)
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

def convert_response_to_unified_format(
    response_json: Any,
    dataset: str="",
    mission: str="",
    start_frame: int=0,
    end_frame: int=0,
    conditional_enabled: bool=False,
) -> Dict[str, Any]:
    """Convert VLM output(s) to the unified format."""

    # --- small coercers ---
    def coerce(obj: Any) -> Any:
        # pydantic -> dict
        try:
            from pydantic import BaseModel  # type: ignore
            if isinstance(obj, BaseModel):
                return obj.model_dump()
        except Exception:
            pass
        # str -> json
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                return obj
        return obj

    def as_decisions(x: Any) -> List[Dict[str, Any]]:
        x = coerce(x)
        if isinstance(x, dict) and "decisions" in x:
            return list(coerce(x["decisions"]))
        if isinstance(x, list):
            return list(x)
        raise ValueError("Expected a list or a dict with key 'decisions'.")

    def to_choice_reason(d: Any) -> Dict[str, Any]:
        d = coerce(d) or {}
        # print("D" , d)
        # accept {choice, reason} or {final_choice, final_reason}
        choice = d.get("choice", d.get("final_choice", None))
        reason = d.get("reason", d.get("final_reason", ""))
        # normalize choice to int when possible
        try:
            choice = int(choice)
        except Exception:
            pass
        return {"choice": choice, "reason": str(reason)}

    data = coerce(response_json)

    out = {
        "dataset": dataset,
        "mission": mission,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "intermediate_responses": [],
        "final_response": {"stage": 0, "choice": None, "reason": ""},
    }

    if conditional_enabled:
        # Expect [decisions_stage0, final_dict]
        if not (isinstance(data, list) and len(data) >= 2):
            raise ValueError("Conditional mode expects [decisions_list, final_dict].")
        stage0 = as_decisions(data[0])
        final_dict = to_choice_reason(as_decisions(data[-1])[0])
        out["intermediate_responses"] = [
            {"stage": 0, **to_choice_reason(d)} for d in stage0
        ]
        out["final_response"] = {"stage": 0, **final_dict}
    else:
        # Expect a single list (or dict with 'decisions'); last element is final
        decs = [to_choice_reason(d) for d in as_decisions(data[0])]
        if not decs:
            raise ValueError("Non-conditional mode requires a non-empty decisions list.")
        inter, final = decs[:-1], decs[-1]
        out["intermediate_responses"] = [{"stage": 0, **d} for d in inter]
        out["final_response"] = {"stage": 0, **final}

    return out

def convert_response_to_actions(
    response: Dict[str, Any],
    motion_arcs: Sequence[Any],
    num_actions: int,
    action_dim: int
):
    """Convert unified response to action list."""
    choice = response.get("final_response", {}).get("choice", None)
    assert choice is not None, "No final choice found in response."

    selected_arc = motion_arcs[choice]
    # Convert to xyz actions sampled along arc
    arc_xy = selected_arc.sample_along_arc(num_samples=num_actions)  # (N, 2)

    # Add constant z height
    if action_dim == 3:
        arc_xy = np.hstack((arc_xy, np.full((arc_xy.shape[0], 1), -0.4)))  # (N, 3)
    
    return arc_xy

class PIVOT:
    """Minimal wrapper that forwards VQA calls to the configured VLM."""

    def __init__(
        self, *, 
        vlm: Dict[str, Any], 
        motion_parameters: Dict[str, Any], 
        annotation: Dict[str, Any], 
        **kwargs
    ) -> None:
        """
        Args:
            vlm: configuration dict for the VLM instance. See _init_vlm().
            motion_parameters: dict of motion parameter settings for motion templates.
            annotation: (optional) dict of annotation settings (not used yet).
        """
        self.vlm = self._init_vlm(vlm)
        self._ann_cfg = annotation
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
        self._num_actions = vlm.get("num_actions", 20)
        self._action_dim = vlm.get("action_dim", 3)  # e.g., (x,y,z)    

    def motion_templates(self):
        mp_cfg = copy.deepcopy(self.mp_cfg)
        motion_bank = MotionTemplateLibrary(
            max_curvature=float(mp_cfg['max_curvature']),
            max_path_len=float(mp_cfg['max_free_path_length']),
            num_options=int(mp_cfg['num_options'])
        )
        return motion_bank.arcs()

    # -------------------- Public API --------------------

    def preprocess_image(self, image: ImgLike, calib: Calib) -> Image.Image:
        """Convert input image to PIL.Image RGB."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)

        # Convert to list of lists of PIL images
        if image.ndim == 5: # [B, T, C, H, W]
            image = image.transpose(0, 1, 3, 4, 2) # [B, T, H, W, C]
            images = [ 
                [self._to_pil(image[b, t]) for t in range(image.shape[1])] 
                for b in range(image.shape[0])
            ]
        elif image.ndim == 4: # [T, C, H, W]
            image = image.transpose(0, 2, 3, 1) # [T, H, W, C]
            images = [[self._to_pil(image[t]) for t in range(image.shape[0])]]
        else:
            raise ValueError("Unsupported image shape for preprocess_image.")

        # Annotate last image with motion templates if available
        image_msgs = []
        motion_arcs = self.motion_templates()
        for b in range(len(images)):
            if len(images[b]) == 0:
                continue

            last_img = images[b][-1]
            annotated, _ = self.annotate_constant_curvature(
                last_img,
                arcs=motion_arcs,
                calib=calib,
                **self._ann_cfg
            )
            images[b][-1] = annotated
            image_msgs.append([ChatQuery("image", "user", img) for img in images[b]])
  
        return image_msgs

    def __call__(self, rgb_image: ImgLike, system_prompt: str, intermediate_prompts: List[str], calib: Calib, **call_kwargs: Any) -> Dict[str, Any]:
        """Upload the image and run a VLM call with a prompt is given"""
        image_ctx_b = self.preprocess_image(rgb_image, calib)

        motion_arcs = self.motion_templates()
        intermediate_responses_b = []
        intermediate_costs_b = []
        unified_actions_b = torch.empty((0, self._num_actions, self._action_dim)).float()

        for b in range(len(image_ctx_b)):
            image_ctx = image_ctx_b[b]
            messages = []
            messages.extend(image_ctx)
            # TODO: Annotate the last image with motion templates if available

            intermediate_responses = []
            intermediate_costs = {}
            for prompt in intermediate_prompts:
                messages.append(ChatQuery("text", "user", prompt))
                response = self.vqa(system_prompt, messages)
                response_output_str = json.dumps(response.output_parsed.model_dump(), ensure_ascii=False, indent=2)
                stage_response = ChatQuery("text", "assistant", response_output_str)
                messages.append(stage_response)
                intermediate_responses.append(response_output_str)
                cost, cost_breakdown = self.vlm.get_cost(
                    self.vlm.get_model_name(),
                    response.usage.input_tokens - response.usage.input_tokens_details.cached_tokens,
                    response.usage.input_tokens_details.cached_tokens,
                    response.usage.output_tokens
                )
                intermediate_costs[f"step_{len(intermediate_responses)}"] = {
                    "cost": cost, "breakdown": cost_breakdown
                }

            unified_response = convert_response_to_unified_format(
                intermediate_responses,
                conditional_enabled=len(intermediate_responses) > 1
            )
            intermediate_responses_b.append(unified_response)
            intermediate_costs_b.append(intermediate_costs)

            # Convert to unified action format
            unified_actions = convert_response_to_actions(
                unified_response, motion_arcs,  self._num_actions, self._action_dim
            )
            unified_actions_b = torch.cat(
                (unified_actions_b, torch.from_numpy(unified_actions).unsqueeze(0)), dim=0)

        B = len(intermediate_responses_b)
        motion_arcs_xyz = torch.empty((len(motion_arcs), self._num_actions, self._action_dim)).float()
        for k, arc in enumerate(motion_arcs):
            arc_xy = torch.from_numpy(arc.sample_along_arc(num_samples=self._num_actions))  # (N, 2)
            motion_arcs_xyz[k] = torch.cat((arc_xy, torch.full((arc_xy.shape[0], 1), -0.4)), dim=1)  # (N, 3)
        motion_arcs_xyz_b = torch.tile(motion_arcs_xyz.unsqueeze(0), (B, 1, 1, 1))

        return {
            "responses": intermediate_responses_b,
            "costs": intermediate_costs_b,
            "action_preds": unified_actions_b,        # [B x N x 3]
            "motion_arcs": motion_arcs_xyz_b          # [B x K x N x 3]
        }

    def vqa(self, system_prompt: str, prompts: list(ChatQuery), resume: bool = False, **call_kwargs: Any) -> PivotVQAResult:
        """Upload the image and run a single VLM call with the provided question."""  
        inputs = self.vlm.compile_prompt(prompts)
        response = self.vlm.generate_response(
            system_prompt,
            inputs,
            **call_kwargs
        )
        return response
    
    def batch_vqa(self, system_prompt: str, batch_prompts: Iterable[ChatThread], **call_kwargs: Any) -> List[PivotVQAResult]:
        """Upload the image and run a batch of VLM calls with the provided questions."""  
        batch_inputs = [self.vlm.compile_prompt(prompts) for prompts in batch_prompts]
        responses = self.vlm.generate_batch_response(
            system_prompt,
            batch_inputs,
            **call_kwargs
        )
        return responses

    def annotate_goal_heading(
        self,
        image: ImgLike,
        heading_angle: float,
        *,
        center: Optional[Tuple[int, int]] = None,
        color: Color = (51, 255, 255),
        thickness: int = 20,
        style: str = "triangle",          # "triangle" | "line" | "chevron"
        length_ratio: float = 0.15,       # fraction of image min(W,H) for shaft length
        head_len_ratio: float = 0.07,     # fraction for arrowhead length
        head_wid_ratio: float = 0.05,     # fraction for arrowhead width
        degrees: bool = True,             # if True, heading_angle is in degrees
        overlay_alpha: float = 0.9        # blend strength over the original image
    ) -> Image.Image:
        """
        Draw a goal-direction arrow near the top of the image.

        Convention (image-centric):
          - 0° (or 0 rad) points UP (toward smaller v / y).
          - Positive angles rotate clockwise (to the RIGHT on the image).
          - Set `degrees=False` if `heading_angle` is already in radians.

        Args:
            image: PIL.Image | np.ndarray | str/Path (local file). URLs not supported here.
            heading_angle: heading of goal relative to robot forward.
            center: (u, v) pixels of the arrow base. Defaults to (W/2, 10%*H).
            color: (R,G,B)
            thickness: shaft thickness (pixels)
            style: "triangle" (default), "line", or "chevron"
            length_ratio: shaft length relative to min(W,H)
            head_len_ratio: head length relative to min(W,H)
            head_wid_ratio: head width relative to min(W,H)
            degrees: interpret `heading_angle` as degrees if True
            overlay_alpha: 0..1 for blending drawn overlay

        Returns:
            PIL.Image.Image with the arrow rendered.
        """
        base = self._to_pil(image).convert("RGB")
        W, H = base.size
        under = base.copy()

        # Create transparent overlay to draw vector graphics cleanly
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Default center near top
        if center is None:
            c = (int(W * 0.5), int(H * 0.1))
        else:
            c = (int(center[0]), int(center[1]))

        # Scale geometry from image size
        s = float(min(W, H))
        shaft_len = max(8.0, s * float(length_ratio))
        head_len  = max(6.0, s * float(head_len_ratio))
        head_wid  = max(4.0, s * float(head_wid_ratio))

        # Convert heading to radians, image-centric:
        # 0 -> up, +cw, y points down in image coords.
        import math
        ang = math.radians(heading_angle) if degrees else float(heading_angle)

        # Unit vector: 0 => (0,-1) up; +ang rotates clockwise
        ux = -math.sin(ang)
        uy = -math.cos(ang)

        # Base and tip of shaft
        x0, y0 = c
        x1 = x0 + ux * shaft_len
        y1 = y0 + uy * shaft_len

        def as_xy(pt):
            return (int(round(pt[0])), int(round(pt[1])))

        # Draw styles
        if style.lower() == "line":
            draw.line([as_xy((x0, y0)), as_xy((x1, y1))], fill=(*color, 255), width=int(thickness))

        elif style.lower() == "chevron":
            # Shaft
            draw.line([as_xy((x0, y0)), as_xy((x1, y1))], fill=(*color, 255), width=int(thickness))
            # Two small fletches at tip
            # Perp vector
            px, py = -uy, ux
            f = head_len * 0.6
            tip = (x1, y1)
            left  = (x1 - ux * f + px * (head_wid * 0.6), y1 - uy * f + py * (head_wid * 0.6))
            right = (x1 - ux * f - px * (head_wid * 0.6), y1 - uy * f - py * (head_wid * 0.6))
            draw.line([as_xy(tip), as_xy(left)],  fill=(*color, 255), width=int(max(2, thickness - 2)))
            draw.line([as_xy(tip), as_xy(right)], fill=(*color, 255), width=int(max(2, thickness - 2)))

        else:  # "triangle" (default) — filled triangular arrowhead + shaft
            # Shaft: stop a bit before the tip to tuck under the head
            shaft_end = (x1 - ux * (head_len * 0.6), y1 - uy * (head_len * 0.6))
            draw.line([as_xy((x0, y0)), as_xy(shaft_end)], fill=(*color, 255), width=int(thickness))

            # Triangle head at the tip
            # Perp vector for width
            px, py = -uy, ux
            tip    = (x1, y1)
            base_c = (x1 - ux * head_len, y1 - uy * head_len)
            left   = (base_c[0] + px * (head_wid * 0.5), base_c[1] + py * (head_wid * 0.5))
            right  = (base_c[0] - px * (head_wid * 0.5), base_c[1] - py * (head_wid * 0.5))

            draw.polygon([as_xy(tip), as_xy(left), as_xy(right)], fill=(*color, 255))
        # Composite overlay
        comp = Image.alpha_composite(under.convert("RGBA"), overlay)
        if 0.0 < overlay_alpha < 1.0:
            comp = Image.blend(under.convert("RGBA"), comp, overlay_alpha)

        return comp.convert("RGB")

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
