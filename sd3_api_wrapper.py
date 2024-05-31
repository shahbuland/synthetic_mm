import getpass
import io
import logging
import os
import shutil
import sys
import time
import uuid
from google.protobuf.struct_pb2 import Struct
from pathlib import Path

from stability_sdk.api import Context, Endpoint, generation

from secret import STABILITY_KEY, STABILITY_HOST

context = Context(STABILITY_HOST, STABILITY_KEY)
(balance, pfp) = context.get_user_info()
print(f"Logged in org:{context._user_organization_id} with balance:{balance}")

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional, Union

import requests
from google.protobuf.struct_pb2 import Struct
from PIL import Image
from stability_sdk.client import StabilityInference, generation

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_PROTOBUF = "application/x-protobuf"

MIME_TYPE_MAP = {
    "image/gif": "gif",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "video/mp4": "mp4",
}

PROTOBUF_SAMPLER_FROM_STR = {
    "DDIM": generation.SAMPLER_DDIM,
    "DDPM": generation.SAMPLER_DDPM,
    "K_DPM_2_ANCESTRAL": generation.SAMPLER_K_DPM_2_ANCESTRAL,
    "K_DPM_2": generation.SAMPLER_K_DPM_2,
    "K_DPMPP_2M": generation.SAMPLER_K_DPMPP_2M,
    "K_DPMPP_2S_ANCESTRAL": generation.SAMPLER_K_DPMPP_2S_ANCESTRAL,
    "K_DPMPP_SDE": generation.SAMPLER_K_DPMPP_SDE,
    "K_EULER_ANCESTRAL": generation.SAMPLER_K_EULER_ANCESTRAL,
    "K_EULER": generation.SAMPLER_K_EULER,
    "K_HEUN": generation.SAMPLER_K_HEUN,
    "K_LMS": generation.SAMPLER_K_LMS,
}

STYLE_PRESETS = [
  'None', '3d-model', 'analog-film', 'anime', 'cinematic', 'comic-book', 'digital-art',
  'enhance', 'fantasy-art', 'isometric', 'line-art', 'low-poly', 'modeling-compound',
  'neon-punk', 'origami', 'photographic', 'pixel-art',
]

@dataclass
class Response:
    finish_reasons: List[int] = field(default_factory=list)
    images: List[bytes] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    video: Optional[bytes] = None
    profile: Optional[dict] = None
    round_trip_time: float = 0.0

def _response_from_protobuf(response_bytes: bytes) -> Response:
    response = Response()
    answer_batch = generation.AnswerBatch()
    answer_batch.ParseFromString(response_bytes)
    for answer in answer_batch.answers:
        for artifact in answer.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                response.finish_reasons.append(artifact.finish_reason)
                response.images.append(artifact.binary)
                response.seeds.append(artifact.seed)
            elif artifact.type == generation.ARTIFACT_VIDEO:
                response.finish_reasons.append(artifact.finish_reason)
                response.seeds.append(artifact.seed)
                response.video = artifact.binary
            elif (
                artifact.type == generation.ARTIFACT_TEXT
                and artifact.mime == "text/plain"
                and artifact.finish_reason != generation.FinishReason.NULL
            ):
                response.finish_reasons.append(artifact.finish_reason)
            elif (
                artifact.type == generation.ARTIFACT_TEXT
                and artifact.mime == CONTENT_TYPE_JSON
            ):
                response.profile = json.loads(artifact.text)
    return response

def _serialize(request: Union[dict, generation.Request]) -> bytes:
    if isinstance(request, dict):
        return json.dumps(request).encode("utf-8")
    else:
        return request.SerializeToString()

def generation_request_grpc(
    prompt: str,
    negative_prompt: str,
    sampler: str = "K_DPMPP_2M",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 32,
    cfg_scale: float = 7.5,
    image_bytes: Optional[bytes] = None,
    mask_bytes: Optional[bytes] = None,
    samples: int = 1,
    denoise_strength: float = 1,
    style_preset: str = "",
    mime_type: str = "image/png",
    init_image_mime_type: Optional[str] = None,
    mask_image_mime_type: Optional[str] = None,
    control_strength: float = 0.0,
    video_frames: int = 25,
    video_interp: bool = False,
    video_loras: List[dict] = [],
    video_motion_id: int = 40,
    workflow: str = "",
    workflow_params: dict = {}
):
    pb_sampler = PROTOBUF_SAMPLER_FROM_STR.get(sampler)
    if pb_sampler is None:
        raise ValueError(
            f"Unknown sampler {sampler}, valid values are: {PROTOBUF_SAMPLER_FROM_STR.keys()}"
        )

    image_parameters = generation.ImageParameters(
        transform=generation.TransformType(diffusion=pb_sampler),
        width=width,
        height=height,
        samples=samples,
        seed=[seed],
        steps=steps,
        parameters=[
            generation.StepParameter(
                sampler=generation.SamplerParameters(cfg_scale=cfg_scale),
                schedule=generation.ScheduleParameters(start=denoise_strength),
            )
        ],
    )

    extras = Struct()
    extras.update(
        {
            "$IPC": {"preset": style_preset},
            "mime_type": mime_type,
            "control_strength": control_strength,
            "video_frames": video_frames,
            "video_interp": video_interp,
            "video_loras": video_loras,
            "video_motion_id": video_motion_id,
            "workflow": workflow,
            **workflow_params
        }
    )

    prompts = [
        generation.Prompt(
            text=prompt, parameters=generation.PromptParameters(weight=1.0)
        ),
        generation.Prompt(
            text=negative_prompt,
            parameters=generation.PromptParameters(weight=-1.0),
        ),
    ]
    if image_bytes is not None:
        prompts.append(
            generation.Prompt(
                artifact=generation.Artifact(
                    type=generation.ARTIFACT_IMAGE,
                    binary=image_bytes,
                    mime=init_image_mime_type,
                )
            )
        )
    if mask_bytes is not None:
        prompts.append(
            generation.Prompt(
                artifact=generation.Artifact(
                    type=generation.ARTIFACT_MASK,
                    binary=mask_bytes,
                    mime=mask_image_mime_type,
                )
            )
        )

    request = generation.Request(prompt=prompts, image=image_parameters, extras=extras)
    return request

def run_request_ssc(
    params: generation.Request,
    endpoint: str = "stable-diffusion-v3-0",
    environment: str = "dev"
) -> Response:
    endpoint_url = f"https://{environment}.api.stability.ai/v1/generation/{endpoint}/"
    print(f"Sending protobuf request to {endpoint_url}...")
    response = requests.post(
        endpoint_url,
        headers={
            "Accept": CONTENT_TYPE_PROTOBUF,
            "Authorization": f"Bearer {STABILITY_KEY}",
            "Content-Type": CONTENT_TYPE_PROTOBUF,
        },
        data=_serialize(params),
    )
    if not response.ok:
        if response.status_code == 404 or (response.status_code == 400 and response.text.startswith("Unsupported")):
            print(f"{endpoint} endpoint appears to be offline.")
        print(response)
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    return _response_from_protobuf(response.content)

def generate_image(prompt, path):
    request = generation_request_grpc(
        prompt=prompt,
        negative_prompt="",
        width=1024,
        height=1024,
        seed=0,
        style_preset="None",
        workflow="image_core_plus_v1.0_trt"
    )
    try:
        response = run_request_ssc(request)
        with open(path, 'wb') as f:
            f.write(response.images[0])
        return True
    except:
        return False

if __name__ == "__main__":
    # Try generating an image of cat
    res = False
    while not res:
        res = generate_image("A photo of a cute cat", "test_image.png")
        print("Tried once")
    print("Success")