 
"""
Download required AI models for the Crowd Behavior Analysis system.
"""

import os
import urllib.request

MODELS = {
    "osnet_x0_25_msmt17.pt":
    "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt"
}

MODEL_DIR = "models"


def download(url, path):
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)
    print(f"Saved → {path}")


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, url in MODELS.items():

        path = os.path.join(MODEL_DIR, name)

        if os.path.exists(path):
            print(f"{name} already exists")
            continue

        download(url, path)


if __name__ == "__main__":
    main()

