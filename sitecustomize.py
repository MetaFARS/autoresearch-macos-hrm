import os
import sys


if os.environ.get("AUTORESEARCH_PREPARE_PATCH") == "1":
    try:
        import requests
        import torch
        import pyarrow.parquet
        import rustbpe
        import tiktoken
    except Exception:
        pass
    sys.platform = "darwin"
    try:
        import torch

        if hasattr(torch.backends, "mps"):
            torch.backends.mps.is_available = lambda: True
    except Exception:
        pass
