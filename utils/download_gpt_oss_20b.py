'''Download the ggml-org/gpt-oss-20b-GGUF model from HuggingFace.

Downloads into models/hugging_face (respects $HF_HOME if set).
Only downloads GGUF files.

Usage:
    python utils/download_gpt_oss_20b.py
'''

import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

snapshot_download(
    repo_id='ggml-org/gpt-oss-20b-GGUF',
    allow_patterns=['*.gguf'],
    token=os.environ.get('HF_TOKEN'),
)
