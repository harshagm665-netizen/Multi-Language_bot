import os
import zipfile
import sys

def create_advanced_zip(output_filename, source_dir):
    # Exclusion patterns
    EXCLUDE_DIRS = {'venv', '.git', '__pycache__', 'models', '.gemini', '.pytest_cache', 'piper', 'model'}
    EXCLUDE_FILES = {'.env', 'create_zip.py', '.DS_Store', 'Novabot_Source_Code.zip'}
    
    print(f"📦 Starting 'Advanced' compression for: {source_dir}")
    print(f"🚫 Excluding: {', '.join(EXCLUDE_DIRS)} and {', '.join(EXCLUDE_FILES)}")

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Prune excluded directories in-place to prevent walking into them
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                if file in EXCLUDE_FILES:
                    continue
                
                # Check for specific path exclusions (like piper/models)
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, source_dir)
                
                # Double check models exclusion if it's deeper
                if 'piper' in rel_path and 'models' in rel_path:
                    continue
                    
                zipf.write(file_path, rel_path)
                # print(f"  + Added: {rel_path}")

    size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"\n✅ Created: {output_filename}")
    print(f"📊 Final Size: {size_mb:.2f} MB")
    
    if size_mb < 25:
        print("✨ Target achieved! The file is under the 25MB limit.")
    else:
        print("⚠️ Warning: File exceeds 25MB. Consider further exclusions.")

if __name__ == "__main__":
    output = "Novabot_Source_Code.zip"
    source = os.getcwd()
    create_advanced_zip(output, source)
