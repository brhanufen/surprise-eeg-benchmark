#!/usr/bin/env python3
"""Download ERP CORE BIDS data (MMN and P3 paradigms) from OSF.

The OSF structure is:
  ERP_CORE_BIDS_Raw_Files/
    sub-XXX/
      ses-MMN/
        eeg/
          sub-XXX_ses-MMN_task-MMN_eeg.set
          sub-XXX_ses-MMN_task-MMN_eeg.fdt
          sub-XXX_ses-MMN_task-MMN_events.tsv
          sub-XXX_ses-MMN_task-MMN_eeg.json
          sub-XXX_ses-MMN_task-MMN_channels.tsv
      ses-P3/
        eeg/
          (similar structure)
"""

import os
import json
import urllib.request
import ssl
import time

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
OSF_NODE = "9f5w7"
BIDS_FOLDER_ID = "5f762ca7e64e7e0116aa6d38"
CTX = ssl.create_default_context()

SESSIONS = ["ses-MMN", "ses-P3"]  # MMN (primary) and P3 (internal replication)


def osf_api_get(url):
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "Python-urllib/3.9 ERP-CORE-Downloader")
    with urllib.request.urlopen(req, timeout=120, context=CTX) as resp:
        return json.loads(resp.read().decode())


def list_folder_all(folder_id):
    url = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/{folder_id}/"
    items = []
    while url:
        data = osf_api_get(url)
        items.extend(data.get("data", []))
        url = data.get("links", {}).get("next")
    return items


def get_folder_id(item):
    return item["links"]["upload"].split("/")[-2]


def download_file(download_url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return False
    req = urllib.request.Request(download_url)
    req.add_header("User-Agent", "Python-urllib/3.9 ERP-CORE-Downloader")
    with urllib.request.urlopen(req, timeout=300, context=CTX) as resp:
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
    return True


def download_folder_recursive(folder_id, local_path):
    """Download all files in a folder recursively."""
    items = list_folder_all(folder_id)
    count = 0
    for item in items:
        name = item["attributes"]["name"]
        kind = item["attributes"]["kind"]
        if kind == "file":
            dl_url = item["links"].get("download", "")
            if dl_url:
                dest = os.path.join(local_path, name)
                if download_file(dl_url, dest):
                    size = item["attributes"].get("size", 0)
                    print(f"    Downloaded: {name} ({size} bytes)")
                    count += 1
                else:
                    print(f"    Exists: {name}")
        elif kind == "folder":
            fid = get_folder_id(item)
            count += download_folder_recursive(fid, os.path.join(local_path, name))
    return count


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Download top-level BIDS files
    print("Downloading top-level BIDS files...")
    top_items = list_folder_all(BIDS_FOLDER_ID)
    for item in top_items:
        if item["attributes"]["kind"] == "file":
            name = item["attributes"]["name"]
            dl_url = item["links"].get("download", "")
            if dl_url:
                dest = os.path.join(RAW_DIR, name)
                if download_file(dl_url, dest):
                    print(f"  Downloaded: {name}")

    # Get subject folders
    subject_folders = sorted(
        [i for i in top_items
         if i["attributes"]["kind"] == "folder" and i["attributes"]["name"].startswith("sub-")],
        key=lambda x: x["attributes"]["name"]
    )
    print(f"\nFound {len(subject_folders)} subject folders")

    total = 0
    for i, sf in enumerate(subject_folders):
        subject_name = sf["attributes"]["name"]
        sub_folder_id = get_folder_id(sf)
        print(f"\n[{i+1}/{len(subject_folders)}] {subject_name}")

        try:
            # List session folders inside this subject
            sub_items = list_folder_all(sub_folder_id)
            for ses_item in sub_items:
                ses_name = ses_item["attributes"]["name"]
                if ses_item["attributes"]["kind"] == "folder" and ses_name in SESSIONS:
                    ses_folder_id = get_folder_id(ses_item)
                    local_ses_path = os.path.join(RAW_DIR, subject_name, ses_name)
                    print(f"  Session: {ses_name}")
                    n = download_folder_recursive(ses_folder_id, local_ses_path)
                    total += n
            time.sleep(0.3)
        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(2)

    print(f"\n{'='*50}")
    print(f"Done! Downloaded {total} new files total.")


if __name__ == "__main__":
    main()
