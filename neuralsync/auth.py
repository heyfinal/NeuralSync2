from fastapi import Request, HTTPException
from .config import load_config

cfg = load_config()

async def bearer_guard(request: Request):
    if not cfg.token:
        return
    hdr = request.headers.get('Authorization','')
    if not hdr.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing Bearer token')
    token = hdr.split(' ',1)[1]
    if token != cfg.token:
        raise HTTPException(status_code=403, detail='Invalid token')
