"""JSON receipts and content hashes shared by data and services."""
import hashlib
import json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
