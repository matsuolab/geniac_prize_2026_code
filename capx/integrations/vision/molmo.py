from __future__ import annotations

import base64
import io
import os
import re
import time
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import PIL
from PIL import Image

from capx.utils import unix_http, unix_http_client

_PROC: Any | None = None  # kept for backward compatibility; unused with vLLM HTTP API
_MODEL: Any | None = None  # kept for backward compatibility; unused with vLLM HTTP API

# SERVICE_URL = "https://openrouter.ai/api/" # OpenRouter
#
# Overridable by environment because this is the *fallback* object localizer: when
# SAM3's language path finds nothing, this is what stands between the agent and
# "object not found". A submission running offline in a container has no separate
# Molmo server, and hard-coding the endpoint means the fallback is simply absent
# there — measured, 2933 refused requests across 115 of 137 tasks in one run.
# Pointing it at a vision-language model the container is already serving costs
# nothing and gives the fallback back.
SERVICE_URL = os.environ.get("CAPX_MOLMO_URL", "http+unix://molmo.sock/v1")
DEFAULT_MODEL = os.environ.get("CAPX_MOLMO_MODEL", "allenai/Molmo2-8B")



def _scaled(points: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], float]:
    """Decide which normalization the coordinates are on, and drop the impossible ones.

    Molmo1 answers on 0-100 and Molmo2 on 0-1000; a general vision-language model
    standing in for either (Qwen3-VL, say) answers on 0-1000 in whatever tag shape
    it feels like. Only the Molmo2 tag states its scale, so for every other format
    the magnitudes are the evidence there is: a coordinate above 100 cannot be on
    the 0-100 scale. Measured against the container's own server, this is what
    stood between the fallback localizer and a point -- 815 of 820 replies parsed
    to nothing because 641 was "out of range" rather than "0.641 of the way across".
    """
    if not points:
        return [], 100.0
    scale = 100.0 if max(max(abs(x), abs(y)) for x, y in points) <= 100.0 else 1000.0
    return [(x, y) for x, y in points if 0.0 <= x <= scale and 0.0 <= y <= scale], scale


def _boxes_to_centres(nums: list[float]) -> list[tuple[float, float]]:
    """Read a flat number list as (x1, y1, x2, y2) boxes and return their centres."""
    return [
        ((nums[i] + nums[i + 2]) / 2.0, (nums[i + 1] + nums[i + 3]) / 2.0)
        for i in range(0, len(nums) - 3, 4)
    ]


def _parse_points(text: str) -> tuple[list[tuple[float, float]], float]:
    """Parse point coordinates from model text output.

    Supports multiple formats:
    - Molmo2: <points coords="obj_idx x y ...">label</points> (normalized 0-1000)
    - Molmo1: <point x="X" y="Y">
    - Legacy: <points x1=".." y1="..">
    - JSON: {"point_2d": [x, y]} or {"bbox_2d": [x1, y1, x2, y2]}
    - Fallback: plain "x, y" pairs, or a bare list of four numbers read as a box

    Every format except Molmo2's carries no scale, so it is inferred per `_scaled`.
    The JSON and bare-box forms are here because a model asked to point at
    something answers with a bounding box about a third of the time, and a box's
    centre is a better point than its top-left corner.

    Args:
        text: Generated text potentially containing point tags.

    Returns:
        Tuple of (points, norm_scale) where points is a list of (x, y) tuples
        and norm_scale is the normalization scale (100.0 or 1000.0).
    """
    # 1) Molmo2 format: <points coords="type obj_idx x y ...">label</points>
    # Format: type indicator, then triplets of (obj_idx, x, y) where x,y are in [0, 1000]
    coords_match = re.search(r'<points\s+coords\s*=\s*["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
    if coords_match:
        try:
            nums = [float(n) for n in coords_match.group(1).split()]
        except ValueError:
            return [], 1000.0
        points: list[tuple[float, float]] = []
        # Skip first number (type indicator), then parse complete triplets: (obj_idx, x, y)
        i = 1
        while i + 2 < len(nums):
            x, y = nums[i + 1], nums[i + 2]
            if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
                points.append((x, y))
            i += 3
        return points, 1000.0  # Molmo2 uses 0-1000 normalization

    # 2) Parse one or more <point x=".." y=".."> tags (Molmo1 format)
    point_tags = re.findall(r"<point\b[^>]*>", text, flags=re.IGNORECASE)
    points = []
    for tag in point_tags:
        mx = re.search(r"\bx\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, flags=re.IGNORECASE)
        my = re.search(r"\by\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, flags=re.IGNORECASE)
        if mx and my:
            points.append((float(mx.group(1)), float(my.group(1))))
    if points:
        return _scaled(points)

    # 3) Legacy <points x1=".." y1=".." ...> format
    tag_match = re.search(r"<points\b[^>]*>", text, flags=re.IGNORECASE)
    if tag_match:
        source = tag_match.group(0)
        xs = {
            int(i): float(v)
            for i, v in re.findall(r"x(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source)
        }
        ys = {
            int(i): float(v)
            for i, v in re.findall(r"y(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source)
        }
        idxs = sorted(set(xs) & set(ys))
        points = [(xs[i], ys[i]) for i in idxs]
        if points:
            return _scaled(points)

    # 4) JSON-ish replies: {"point_2d": [x, y]} and {"bbox_2d": [x1, y1, x2, y2]}
    json_points = [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(
            r'"point_2d"\s*:\s*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)', text
        )
    ]
    if json_points:
        return _scaled(json_points)
    json_boxes = [
        _boxes_to_centres([float(n) for n in m.group(1).split(",")])[0]
        for m in re.finditer(
            r'"(?:bbox_2d|box_2d|bbox)"\s*:\s*\[\s*((?:[0-9]*\.?[0-9]+\s*,\s*){3}[0-9]*\.?[0-9]+)',
            text,
        )
    ]
    if json_boxes:
        return _scaled(json_boxes)

    # 5) Fallback: bracketed groups of four numbers are boxes, wherever in the prose
    # they sit ("One moka pot is located at [459, 587, 593, 781]").
    quads = [
        _boxes_to_centres([float(n) for n in m.group(1).split(",")])[0]
        for m in re.finditer(
            r"\[\s*((?:[0-9]*\.?[0-9]+\s*,\s*){3}[0-9]*\.?[0-9]+)\s*\]", text
        )
    ]
    if quads:
        return _scaled(quads)
    # Signed, and never starting mid-number, so "-5, 3" is not read as (5, 3).
    pairs = re.findall(r"(?<![\d.-])(-?[0-9]*\.?[0-9]+)\s*,\s*(-?[0-9]*\.?[0-9]+)", text)
    return _scaled([(float(x), float(y)) for x, y in pairs])


def _overlay_and_save(
    image: Image.Image, points: list[tuple[float, float]], save_path: str, *, norm_scale: float = 100.0
) -> None:
    """Overlay points on image and save figure.

    Args:
        image: PIL image the points refer to.
        points: Normalized coordinates ordered top-down, left-to-right.
        save_path: Path to save the visualization.
        norm_scale: Normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    if not points:
        return
    width, height = image.size
    xs = [x / norm_scale * width for x, _ in points]
    ys = [y / norm_scale * height for _, y in points]

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(xs, ys, s=60, c="red", edgecolors="white", linewidths=1.5, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x + 3, y + 3, str(i), color="yellow", fontsize=10, weight="bold", zorder=4)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def convert_to_pixel_coordinates(
    points: list[tuple[float, float]], image: PIL.Image.Image, norm_scale: float = 100.0
) -> list[tuple[int, int]]:
    """Convert normalized point coordinates to pixel coordinates.
    
    Args:
        points: List of (x, y) tuples in normalized coordinates.
        image: PIL image to get dimensions from.
        norm_scale: Normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    width, height = image.size
    xs = [int(x / norm_scale * width) for x, _ in points]
    ys = [int(y / norm_scale * height) for _, y in points]
    return list(zip(xs, ys, strict=True))


def _image_to_data_url(image: PIL.Image.Image) -> str:
    """Encode a PIL image as a PNG data URL suitable for OpenAI-compatible APIs."""
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def init_molmo(
    # model_name: str = "allenai/moldmo-2-8b:free", # OpenRouter
    # model_name: str = "allenai/Molmo2-O-7B",
    model_name: str = DEFAULT_MODEL,
    base_url: str = SERVICE_URL,
    api_key: str | None = None,
) -> Callable[[PIL.Image.Image, list[str] | None], dict[str, tuple[int | None, int | None]]]:
    """Initialize a detector that queries a vLLM OpenAI-compatible server.

    Args:
        model_name: The served model name (as registered in vLLM).
        base_url: Base URL of the OpenAI-compatible endpoint (no trailing slash beyond /v1).
        api_key: Optional API key; omit if the server does not require auth.

    Returns:
        A callable that takes an image and a list of object strings, and returns a
        mapping from object name to a pixel coordinate tuple (x, y). If a point
        could not be parsed, the value is (None, None).
    """
    # chat_url = f"{base_url.rstrip('/')}/v1/chat/completions" # SGLANG 
    chat_url = f"{base_url.rstrip('/')}/chat/completions" # vLLM
    unix_http.check_server_url(chat_url)
    session = unix_http_client.new_session()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def det_fn(
        image: PIL.Image.Image, objects: list[str] | None = None
    ) -> dict[str, tuple[int | None, int | None]]:
        """
        Args:
            image: PIL.Image.Image: The RGB image to process.
            objects: list[str]: The list of object queries to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """

        if not objects:
            return {}

        img_url = _image_to_data_url(image)
        all_points: dict[str, tuple[int | None, int | None]] = {}
        for obj in objects:
            prompt = (
                f"Point at {obj}"
            )
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_url}},
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stop": ["<|endoftext|>"],
                # Pointing at a named object is a look-up, not a reasoning task, and
                # a thinking model spends the whole budget before saying anything:
                # measured against a Qwen3 server, 795 of 795 replies came back with
                # an empty `content`. Servers that do not know this key ignore it.
                "chat_template_kwargs": {"enable_thinking": False},
            }

            max_retries, backoff = 3, 1.0
            for attempt in range(max_retries):
                try:
                    resp = session.post(chat_url, json=payload, headers=headers, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    message = data.get("choices", [{}])[0].get("message", {}) or {}
                    # A server running a reasoning parser splits the reply in two,
                    # and everything lands in reasoning_content when the model was
                    # still thinking at the token limit. The point is in there.
                    generated_text = message.get("content") or message.get(
                        "reasoning_content"
                    ) or ""
                    print(f"Generated text for '{obj}': {generated_text}")
                    break
                except TimeoutError:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"Request failed for '{obj}' (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(backoff * (2 ** attempt))
                    else:
                        generated_text = ""

            points, norm_scale = _parse_points(generated_text or "")
            if points:
                abs_coords = convert_to_pixel_coordinates(points, image, norm_scale)[0]
                # save_path = f"outputs/visualization/{obj}_points_{time.time()}.png"
                # _overlay_and_save(image, points, save_path, norm_scale=norm_scale)
                # print(f"Saved visualization to: {save_path}")
            else:
                abs_coords = (None, None)
                print("No points parsed from model output; skipping visualization.")
            all_points[obj] = abs_coords

        return all_points

    # register_object_detector(det_fn)
    return det_fn