import os
import requests
import json

# Configuration
MODELS_DIR = "piper/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Language -> Model Filenames mapping
MODELS = {
    "English (US)": "en_US-amy-low.onnx",
    "English (India)": "en_GB-southern_english_female-low.onnx",
    "Hindi": "hi_IN-pratham-medium.onnx",
    "Tamil": "ta_IN-tamil_female-medium.onnx",  # Changed to working model
    "Malayalam": "ml_IN-arjun-medium.onnx",
    "Kannada": "kn_IN-kannada_male-medium.onnx",  # Added Kannada
    "French": "fr_FR-siwis-low.onnx",
    "Spanish": "es_ES-carlfm-x_low.onnx"
}

# Direct download URLs for each model
MODEL_URLS = {
    # Official Piper voices
    "en_US-amy-low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx",
    "en_GB-southern_english_female-low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/southern_english_female/low/en_GB-southern_english_female-low.onnx",
    "hi_IN-pratham-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
    "ml_IN-arjun-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ml/ml_IN/arjun/medium/ml_IN-arjun-medium.onnx",
    "fr_FR-siwis-low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx",
    "es_ES-carlfm-x_low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx",
   
    # Community Tamil model (simrat39)
    "ta_IN-tamil_female-medium.onnx": "https://huggingface.co/simrat39/tamil-piper-model/resolve/main/ta_IN-tamil_female-medium.onnx",
   
    # Community Kannada model
    "kn_IN-kannada_male-medium.onnx": "https://huggingface.co/Bheemappa/Piper_Kannada/resolve/main/kn_IN-kannada_male-medium.onnx",
}

# Alternative URLs if primary fails
ALTERNATIVE_URLS = {
    "ta_IN-tamil_female-medium.onnx": [
        "https://huggingface.co/ezhilkumaran/piper-tamil/resolve/main/ta_IN-roja-medium.onnx",
        "https://github.com/simrat39/piper-tamil/releases/download/v1.0.0/ta_IN-tamil_female-medium.onnx"
    ],
    "kn_IN-kannada_male-medium.onnx": [
        "https://huggingface.co/steja/piper-kannada/resolve/main/kn_IN-kannada-medium.onnx",
    ]
}

def download_file(url, dest, timeout=60):
    """Download file with proper error handling"""
    if os.path.exists(dest):
        print(f"✓ Skipping {os.path.basename(dest)}, already exists.")
        return True
   
    print(f"⬇ Downloading {os.path.basename(dest)}...")
    print(f"  URL: {url}")
   
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
       
        if r.status_code == 200:
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
           
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
           
            print(f"\n  ✓ Successfully downloaded {os.path.basename(dest)}")
            return True
        else:
            print(f"  ✗ Failed (HTTP {r.status_code})")
            return False
           
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")
        return False

def download_with_fallback(filename, primary_url, alternatives=None):
    """Try primary URL first, then alternatives"""
    dest = os.path.join(MODELS_DIR, filename)
   
    # Try primary URL
    if download_file(primary_url, dest):
        return True
   
    # Try alternative URLs
    if alternatives:
        print(f"  Trying alternative URLs for {filename}...")
        for alt_url in alternatives:
            # For alternative URLs, might need different filename
            if download_file(alt_url, dest):
                return True
   
    return False

def create_json_config(model_filename, language_code, sample_rate=22050):
    """Create a basic JSON config if download fails"""
    config = {
        "audio": {
            "sample_rate": sample_rate
        },
        "espeak": {
            "voice": language_code
        },
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8
        },
        "phoneme_type": "espeak",
        "phoneme_map": {},
        "phoneme_id_map": {}
    }
   
    json_path = os.path.join(MODELS_DIR, f"{model_filename}.json")
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Created basic config: {model_filename}.json")

def main():
    print("=" * 60)
    print("Piper TTS Model Downloader")
    print("=" * 60)
    print(f"Download directory: {os.path.abspath(MODELS_DIR)}\n")
   
    success_count = 0
    failed_models = []
   
    for language, filename in MODELS.items():
        print(f"\n[{language}]")
        print("-" * 40)
       
        if filename not in MODEL_URLS:
            print(f"  ✗ No URL configured for {filename}")
            failed_models.append((language, filename))
            continue
       
        primary_url = MODEL_URLS[filename]
        alternatives = ALTERNATIVE_URLS.get(filename, [])
       
        # Download ONNX model
        if download_with_fallback(filename, primary_url, alternatives):
            success_count += 1
           
            # Download JSON config
            json_filename = f"{filename}.json"
            json_url = f"{primary_url}.json"
            json_dest = os.path.join(MODELS_DIR, json_filename)
           
            if not download_file(json_url, json_dest):
                # Try to create basic config
                lang_code = filename.split("-")[0].replace("_", "-")
                create_json_config(filename, lang_code)
        else:
            failed_models.append((language, filename))
   
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✓ Successful: {success_count}/{len(MODELS)}")
   
    if failed_models:
        print(f"✗ Failed: {len(failed_models)}")
        print("\nFailed models:")
        for lang, fname in failed_models:
            print(f"  - {lang}: {fname}")
       
        print("\n" + "-" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("-" * 60)
        print("\nFor Tamil:")
        print("  1. Visit: https://huggingface.co/simrat39/tamil-piper-model")
        print("  2. Download the .onnx and .onnx.json files")
        print(f"  3. Place them in: {os.path.abspath(MODELS_DIR)}")
       
        print("\nFor Kannada:")
        print("  1. Visit: https://huggingface.co/Bheemappa/Piper_Kannada")
        print("     OR: https://github.com/braille-projects/piper-voices-kannada")
        print("  2. Download the .onnx and .onnx.json files")
        print(f"  3. Place them in: {os.path.abspath(MODELS_DIR)}")
       
        print("\nAlternative - Use gdown for Google Drive hosted models:")
        print("  pip install gdown")
        print("  gdown <google_drive_file_id>")
   
    # List downloaded files
    print("\n" + "-" * 60)
    print("Files in models directory:")
    print("-" * 60)
    if os.path.exists(MODELS_DIR):
        files = os.listdir(MODELS_DIR)
        if files:
            for f in sorted(files):
                size = os.path.getsize(os.path.join(MODELS_DIR, f))
                size_mb = size / (1024 * 1024)
                print(f"  {f} ({size_mb:.2f} MB)")
        else:
            print("  (empty)")

if __name__ == "__main__":
    main()
