import os
import requests


files_to_download = {
    "https://drive.google.com/file/d/1jPFcPwXmypFKZcvZL001lUJp1Aq3OwzY/view?usp=sharing": "experiment1/saved_results/datasets/forecasts_experiment1.csv",
    "https://drive.google.com/file/d/1ysRn-yefeKuFlF2zytWQSNlXxfAxkgVo/view?usp=sharing": "experiment2/saved_results/datasets/forecasts_experiment2.csv",
    "https://drive.google.com/file/d/16mIGSUHTMzEAlnOPvgzPEBYqyTv47SmD/view?usp=sharing": "experiment3/saved_results/datasets/forecasts_experiment3.csv",
    "https://drive.google.com/file/d/1R_ctPkUFmN1AjUkrJZ7yAZGyvYwKE5fp/view?usp=sharing": "weather_normalisation/saved_results/WETNOR_timeseries/ERP_WETNOR_combined_ts.csv",
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
