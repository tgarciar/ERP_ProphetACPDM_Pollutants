import os
import requests


files_to_download = {
    "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_1": "path/to/directory1/filename1.ext",
    "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_2": "path/to/directory2/filename2.ext",
}

def download_file(url, dest_path):
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Stream the download to avoid loading the entire file into memory
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Write the content to a file
        with open(dest_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}. Reason: {str(e)}")

if __name__ == "__main__":
    for url, path in files_to_download.items():
        download_file(url, path)
