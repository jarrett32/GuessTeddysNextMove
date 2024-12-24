import asyncio
import os
import time

from blinkpy.auth import Auth
from blinkpy.blinkpy import Blink
from blinkpy.helpers.util import json_load
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv('BLINK_USERNAME')
PASSWORD = os.getenv('BLINK_PASSWORD')
OUTPUT_DATA_PATH = 'data/{camera_name}_{timestamp}.jpg'

# Initialize the Blink object
async def initialize_blink(username, password):
    from aiohttp import ClientSession
    blink = Blink(session=ClientSession())
    if os.path.exists("src/data_collection/blink_auth.json"):
        auth = Auth(await json_load("src/data_collection/blink_auth.json"),
                     no_prompt=True)
        blink.auth = auth
        await blink.start()
    else:
        auth = Auth({"username": username, "password": password}, no_prompt=True)
        await blink.start()
        print("Please enter the auth key:")
        auth_key = input()
        await auth.send_auth_key(blink, auth_key)
        await blink.save("src/data_collection/blink_auth.json")

    blink.auth = auth

    await blink.setup_post_verify()
    return blink

def print_camera_info(blink):
    for name, camera in blink.cameras.items():
        print(name)
        print(camera.attributes)

def get_cameras(blink):
    cameras = []
    for name, _ in blink.cameras.items():
        cameras.append(blink.cameras[name])
    return cameras

# Screenshot loop
async def take_screenshots(blink, cameras, output_path):
    while True:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        for camera in cameras:
            current_output_path = output_path.format(timestamp=timestamp,
                                                     camera_name=camera.name)
            await camera.snap_picture()
            await blink.refresh(force=True)
            await camera.image_to_file(current_output_path)
            print(f"Screenshot taken: {current_output_path}")

async def main():
    blink = await initialize_blink(USERNAME, PASSWORD)
    print_camera_info(blink)
    cameras = get_cameras(blink)
    await take_screenshots(blink, cameras, OUTPUT_DATA_PATH)      # Take a new

if __name__ == "__main__":
    print(os.getenv('BLINK_USERNAME'))
    asyncio.run(main())
