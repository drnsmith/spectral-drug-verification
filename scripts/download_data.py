from pathlib import Path
import requests

FIGSHARE_FILE_URL = "https://figshare.com/ndownloader/files/51679751"
OUTPUT_PATH = Path("data/raw/raman_spectra_api_compounds.csv")


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def main():
    if OUTPUT_PATH.exists():
        print(f"File already exists at {OUTPUT_PATH}")
        return

    print(f"Downloading dataset to {OUTPUT_PATH} ...")
    download_file(FIGSHARE_FILE_URL, OUTPUT_PATH)
    print("Download complete.")


if __name__ == "__main__":
    main()