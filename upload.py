from huggingface_hub import HfApi
import os
from pathlib import Path

def upload_zst_to_hf(
    repo_id: str,
    zst_file_path: str,
    token: str = None,
    repo_type: str = "dataset"
):
    # Initialize the Hugging Face API client
    api = HfApi()
    
    # If token is not provided, try to get it from environment variable
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError("Please provide a token or set the HF_TOKEN environment variable")
    
    # Upload the file
    print(f"Uploading {zst_file_path} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=zst_file_path,
        path_in_repo=os.path.basename(zst_file_path),
        repo_id=repo_id,
        token=token,
        repo_type=repo_type
    )
    print("Upload completed successfully!")

if __name__ == "__main__":
    for p in Path("data/Slimpajama").iterdir():
        # Example usage
        upload_zst_to_hf(
            repo_id="LiqunMa/temp",
            zst_file_path=p,
            token=""  # Or set HF_TOKEN environment variable
        )