# This file is responsible for downloading the HVFHV Trip data from the internet. There are two urls defined in this code one is responsible for download of all the zone ids of NYC and another one is responsible for downloading heavy dataset of trips for the months of Feb, March and April 2026. The dataset is downloaded in smaller chunks to reduce load on the RAM and an update mechanism is being used such that the user gets an update about ho much data has been downloaded and what is the total size of the data. If the system already contians the data then code skips the download and informs the user about the same.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import config


ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
HVFHV_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to local path."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return

    print(f"  Downloading {dest.name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (100 * 1024 * 1024) < len(chunk):
                    print(f"    {downloaded / (1024 * 1024):.0f} MB / {total / (1024 * 1024):.0f} MB")

    print(f"  Saved: {dest} ({dest.stat().st_size:,} bytes)")


def main():
    print("=" * 60)
    print("UrbanFlow — Data Downloader v2.0")
    print("  Dataset: NYC TLC HVFHV (Uber/Lyft)")
    print("=" * 60)

    print("\n[1/2] Taxi Zone Lookup CSV")
    download_file(ZONE_LOOKUP_URL, config.ZONE_LOOKUP_PATH)

    print("\n[2/2] HVFHV trip data files")
    found = 0
    missing = 0
    for path in config.RAW_DATA_FILES:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] Found: {path.name} ({size_mb:.0f} MB)")
            found += 1
        else:
            url = f"{HVFHV_BASE_URL}/{path.name}"
            try:
                download_file(url, path)
                found += 1
            except requests.RequestException as exc:
                print(f"  [!!] Could not download {path.name}: {exc}")
                missing += 1

    if missing > 0:
        print(f"\n  WARNING: {missing} file(s) missing.")
        print(f"  Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")
        print(f"  Look for 'High Volume FHV' trip data files.")
        if found == 0:
            sys.exit(1)

    print(f"\n[OK] {found} HVFHV data file(s) ready for preprocessing!")
    print(f"  Next step: python data/preprocess.py")


if __name__ == "__main__":
    main()
