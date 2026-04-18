import os
import requests
import zipfile
import gdown

def download_and_extract(url: str, dest_dir: str, zip_filename: str):
    """
    Downloads a zip file from a Google Drive URL and extracts it into the target directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, zip_filename)
    
    # Simple redirect handling for Google Drive links
    print(f"Downloading from {url}...")
    try:
        gdown.download(url, zip_path, quiet=False)
    except ImportError:
        print("gdown not installed")
        r = requests.get(url)
        assert r.status_code == 200, f"Failed to download. Status code: {r.status_code}"
        with open(zip_path, "wb") as f:
            f.write(r.content)
            
    # Extract the zip file
    print(f"Extracting {zip_filename} into {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Successfully extracted to {dest_dir}\n")

if __name__ == "__main__":
    target_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"--- Downloading Oklahoma97 Dataset ---")
    okla_url = "https://drive.google.com/u/0/uc?id=1tNcxgtEQX3dtDKqwDMswJEvxpKBpov75&export=download"
    download_and_extract(okla_url, os.path.join(target_dir, "oklahoma"), "oklahoma.zip")

    print(f"--- Downloading UNC28 Dataset ---")
    unc_url = "https://drive.google.com/u/0/uc?id=17Tqxi3BD1u5OPximt_Kb_ZcX-0DyuRfl&export=download"
    download_and_extract(unc_url, os.path.join(target_dir, "unc"), "unc28.zip")
    
    print("Download and extraction complete.")
    print(f"You can find the .pkl files in:")
    print(f"  {target_dir}/oklahoma")
    print(f"  {target_dir}/unc")
