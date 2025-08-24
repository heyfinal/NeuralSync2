import os, numpy as np, hashlib, requests
from .utils import env_bool

DIM = int(os.environ.get('NS_VECTOR_DIM','512'))

def _hash_tokens(text:str, dim:int)->np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.blake2b(tok.encode('utf-8'), digest_size=8).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec) + 1e-8
    return (vec/n).astype('float32')

def _openai_embed(text:str)->np.ndarray:
    key = os.environ.get('OPENAI_API_KEY')
    if not key: raise RuntimeError('OPENAI_API_KEY not set')
    model = os.environ.get('NS_OPENAI_EMBED_MODEL','text-embedding-3-small')
    url = 'https://api.openai.com/v1/embeddings'
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    data = {'model': model, 'input': text}
    r = requests.post(url, json=data, headers=headers, timeout=15)
    r.raise_for_status()
    v = np.array(r.json()['data'][0]['embedding'], dtype='float32')
    n = np.linalg.norm(v) + 1e-8
    return (v/n).astype('float32')

def embed(text:str)->np.ndarray:
    if env_bool('NS_USE_OPENAI', False):
        try:
            return _openai_embed(text or '')
        except Exception:
            pass
    return _hash_tokens(text or '', DIM)
