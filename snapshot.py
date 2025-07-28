import sys
import pandas
from huggingface_hub import login, snapshot_download

# Login to Hugging Face
huggingface_token = ""  # Replace with your actual token
login(token=huggingface_token)

model_name = 'MBZUAI/Llama-3-Nanda-10B-Chat'

local_dir = "/mnt/storage/deeksha/models/MBZUAI/Llama-3-Nanda-10B-Chat"

snapshot_download(model_name, local_dir=local_dir, allow_patterns=["*"])
print(f"Model downloaded to {local_dir}")