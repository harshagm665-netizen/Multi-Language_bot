import os
import requests
import json

# Configuration
MODELS_DIR = "piper/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Language -> Model Filenames mapping (Synchronized with backend.py)
MODELS = {
    "English (US)": "en_US-amy-low.onnx",
    "Hindi": "hi_IN-pratham-medium.onnx",
    "Tamil": "ta_IN-tamil_female-medium.onnx",
    "Kannada": "kn_IN-kannada_male-medium.onnx",
    "Malayalam": "ml_IN-arjun-medium.onnx"
}

# Base URL for rhasspy piper voices
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

# Direct URLs or relative paths in rhasspy repo
LANG_PATHS = {
    "en_US-amy-low.onnx": "en/en_US/amy/low/en_US-amy-low.onnx",
    "hi_IN-pratham-medium.onnx": "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
    "ta_IN-tamil_female-medium.onnx": "COMMUNITY:ezhilkumaran/piper-tamil/resolve/main/ta_IN-roja-medium.onnx", # Redirected to Roja
    "kn_IN-kannada_male-medium.onnx": "COMMUNITY:braille-projects/piper-voices-kannada/resolve/main/kn_IN-kannada_male-medium.onnx",
    "ml_IN-arjun-medium.onnx": "ml/ml_IN/arjun/medium/ml_IN-arjun-medium.onnx",
}

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"Skipping {dest}, already exists.")
        return
    print(f"Downloading {url} to {dest}...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Failed to download {url} (Status: {r.status_code})")

def main():
    for name, filename in MODELS.items():
        if filename in LANG_PATHS:
            path = LANG_PATHS[filename]
            
            # Check for community URL
            if path.startswith("COMMUNITY:"):
                onnx_url = path.replace("COMMUNITY:", "https://huggingface.co/")
            else:
                onnx_url = f"{BASE_URL}/{path}"
            
            # Download .onnx
            download_file(onnx_url, os.path.join(MODELS_DIR, filename))
            # Download .json config
            json_url = f"{onnx_url}.json"
            download_file(json_url, os.path.join(MODELS_DIR, f"{filename}.json"))

    # Special cases for community models
    print("\nNote: Kannada (kn_IN) and Tamil (ta_IN) are missing from the official Piper repository.")
    print("Suggested sources:")
    print(" - Kannada: https://github.com/braille-projects/piper-voices-kannada")
    print(" - Tamil: https://huggingface.co/simrat39/tamil-piper-model OR ezhilkumaran/piper-tamil")
    print("Download these manually into the piper/models/ folder and name them according to backend.py")

if __name__ == "__main__":
    main()
