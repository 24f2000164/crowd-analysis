import asyncio
import websockets
import json
import cv2
import numpy as np

STREAM_ID = input("Enter stream ID to view (e.g. 'test_stream'): ")

async def run():

    uri = f"ws://127.0.0.1:8000/ws/stream/{STREAM_ID}"

    async with websockets.connect(uri) as ws:

        while True:

            message = await ws.recv()

            # metadata
            if isinstance(message, str):

                meta = json.loads(message)

                if meta["type"] == "status":
                    print("Status:", meta)

                elif meta["type"] == "frame":
                    print(
                        "Frame:",
                        meta["frame_index"],
                        "people:",
                        meta["track_count"],
                        "behavior:",
                        meta["behavior"]
                    )

            # frame image
            else:

                img = np.frombuffer(message, dtype=np.uint8)
                frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

                cv2.imshow("Crowd Analysis", frame)

                if cv2.waitKey(1) == 27:
                    break

asyncio.run(run())