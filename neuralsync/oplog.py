import json, os
from typing import Dict, Any

class OpLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            open(path,'w').close()

    def append(self, op: Dict[str,Any]) -> int:
        with open(self.path,'a') as f:
            pos = f.tell()
            f.write(json.dumps(op, ensure_ascii=False)+'\n')
            return pos

    def read_since(self, offset: int = 0):
        with open(self.path,'r') as f:
            f.seek(offset)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line: break
                yield pos, json.loads(line)
