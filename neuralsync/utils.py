import time, re, hashlib, os

def now_ms():
    return int(time.time()*1000)

SECRET_PAT = re.compile(r'(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-\_]{20,}|ghp_[A-Za-z0-9]{20,})')

def redact(text: str) -> str:
    return SECRET_PAT.sub('[REDACTED]', text or '')

def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def env_bool(name:str, default:bool=False)->bool:
    v=os.environ.get(name); return default if v is None else str(v).lower() in ('1','true','yes','on')
