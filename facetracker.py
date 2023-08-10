import asyncio
import os
import time
import numpy as np
from http import HTTPStatus
from typing import Any

import aiohttp
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerResult, FaceLandmarkerOptions, RunningMode

DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DIR, "models")
FACE_TASK_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
MJPG_FOURCC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')


async def upload_results(result: Any):
    try:
        response = await client.put("/v1/faces", json=result)
        if response.status != HTTPStatus.OK:
            text = await response.text()
            print(f"failed to upload faces {response.status}: {text}")
    except aiohttp.ClientOSError:
        pass


def print_result(loop: asyncio.BaseEventLoop, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    faces = []

    for i in range(0, len(result.facial_transformation_matrixes)):
        matrix = result.facial_transformation_matrixes[i]
        matrix[0][3] -= 0.5
        matrix[1][3] -= 0.5
        matrix[0][3] *= -1
        matrix[1][3] *= -1
        matrix[2][3] *= -1
        matrix = np.matmul([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], matrix)

        landmarks = [{
            "position": [(landmark.x - 0.5) * x_aspect, -(landmark.y - 0.5) * y_aspect, landmark.z],
            "presence": landmark.presence,
            "visibility": landmark.visibility,
        } for landmark in result.face_landmarks[i]]
        blend_shapes = {s.category_name: s.score for s in result.face_blendshapes[i]}
        blend_shapes["mouthPucker"] = 1 - blend_shapes["mouthPucker"]
        faces.append({
            "transform": matrix.transpose().ravel().tolist(),
            "landmarks": landmarks,
            "blendShapes": blend_shapes,
        })
    packet = {
        "type": "faces",
        "faces": faces,
    }
    loop.create_task(upload_results(packet))


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


async def upload_frame(width: int, height: int, data: bytes) -> bool:
    try:
        response = await client.put("/v1/camera",
                                    headers={"Width": str(width), "Height": str(height)},
                                    data=data)
        if response.status != HTTPStatus.OK:
            text = await response.text()
            print(f"failed to upload camera {response.status}: {text}")
            return False
    except aiohttp.ClientOSError:
        print(f"failed to connect to API, retrying")
        await asyncio.sleep(5)
        return False


async def main():
    global client
    client = aiohttp.ClientSession("http://localhost:8888")

    loop = asyncio.get_event_loop()
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_TASK_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        result_callback=lambda *a, **kw: print_result(loop, *a, **kw))

    with FaceLandmarker.create_from_options(face_options) as landmarker:
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break

            # cv2.imshow("Camera", frame)

            now = time.monotonic()
            frame_timestamp_ms = int((now - epoch) * 1e3)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.flip(frame, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame)
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            frame_bytes = frame.tobytes()
            expected_frame_bytes = 4 * width * height
            if len(frame_bytes) != expected_frame_bytes:
                raise RuntimeError(f"Invalid frame size byte_len={len(frame_bytes)} expected {expected_frame_bytes}")

            await upload_frame(width, height, frame_bytes)

            deadline = now + interval
            sleep = deadline - time.monotonic()
            await asyncio.sleep(max(sleep, 0))


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam.set(cv2.CAP_PROP_FPS, 60)

fourcc = cam.get(cv2.CAP_PROP_FOURCC)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)

if width > height:
    x_aspect = width / height
    y_aspect = 1
else:
    x_aspect = 1
    y_aspect = height / width

interval = 1 / fps

print(f"fourcc={decode_fourcc(fourcc)} width={width} height={height} fps={fps}")

epoch = time.monotonic()
last = epoch
client: aiohttp.ClientSession

asyncio.run(main())
