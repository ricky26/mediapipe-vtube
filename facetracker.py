import asyncio
import os
import time
from typing import Any

import cv2
import mediapipe as mp
import msgpack
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerResult, FaceLandmarkerOptions, RunningMode

DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DIR, "models")
FACE_TASK_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
MJPG_FOURCC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')


def print_result(loop: asyncio.BaseEventLoop, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    faces = []

    for i in range(0, len(result.facial_transformation_matrixes)):
        landmarks = [{
            "position": [landmark.x, landmark.y, landmark.z],
            "presence": landmark.presence,
            "visibility": landmark.visibility,
        } for landmark in result.face_landmarks[i]]

        # blend_shapes = [{
        #     "index": s.index,
        #     "score": s.score,
        #     "displayName": s.display_name,
        #     "categoryName": s.category_name,
        # } for s in result.face_blendshapes[i]]
        blend_shapes = {s.category_name: s.score for s in result.face_blendshapes[i]}
        faces.append({
            "transform": result.facial_transformation_matrixes[i].ravel().tolist(),
            "landmarks": landmarks,
            "blendShapes": blend_shapes,
        })

    packet = {
        "type": "faces",
        "faces": faces,
    }
    broadcast(packet)


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


async def track():
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

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            frame_bytes = frame.tobytes()
            expected_frame_bytes = 3 * width * height

            if len(frame_bytes) == expected_frame_bytes:
                packet = {"type": "camera", "width": width, "height": height, "bytes": frame_bytes}
                broadcast(packet)
            else:
                print(f"Invalid frame size byte_len={len(frame_bytes)} expected {expected_frame_bytes}")

            deadline = now + interval
            sleep = deadline - time.monotonic()
            await asyncio.sleep(max(sleep, 0))


def broadcast(packet: Any):
    packet_bytes = msgpack.dumps(packet)
    client_writer.write(packet_bytes)
    # for writer in client_writers:
    #     writer.write(packet_bytes)


# async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
#     client_writers.append(writer)
#     print("New connection")
#     try:
#         while True:
#             size = int.from_bytes(await reader.readexactly(4))
#             packet_bytes = await reader.readexactly(size)
#             packet = msgpack.loads(packet_bytes)
#             print(f"Packet from client (currently unsupported): {packet}")
#     except asyncio.IncompleteReadError:
#         pass
#     except BrokenPipeError:
#         pass
#     finally:
#         print("Connection lost")
#         client_writers.remove(writer)
#
#
# async def listen():
#     server = await asyncio.start_server(handle_client, "localhost", 8888)
#     async with server:
#         await server.serve_forever()

async def recv():
    print("Connected")
    try:
        while True:
            size = int.from_bytes(await client_reader.readexactly(4))
            packet_bytes = await client_reader.readexactly(size)
            packet = msgpack.loads(packet_bytes)
            print(f"Packet from client (currently unsupported): {packet}")
    except asyncio.IncompleteReadError:
        pass
    except BrokenPipeError:
        pass
    finally:
        print("Connection lost")


async def main():
    global client_reader
    global client_writer
    (client_reader, client_writer) = await asyncio.open_connection("localhost", 8888)

    tracking = track()
    receiving = recv()
    await asyncio.gather(tracking, receiving)


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam.set(cv2.CAP_PROP_FPS, 60)

fourcc = cam.get(cv2.CAP_PROP_FOURCC)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)

interval = 1 / fps

print(f"fourcc={decode_fourcc(fourcc)} width={width} height={height} fps={fps}")

epoch = time.monotonic()
last = epoch
client_reader: asyncio.StreamReader
client_writer: asyncio.StreamWriter

asyncio.run(main())
