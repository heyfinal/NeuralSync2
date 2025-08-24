import pathlib, uuid, yaml
from dataclasses import dataclass

DEFAULT_HOME = pathlib.Path.home()/'.neuralsync'
DEFAULT_HOME.mkdir(parents=True, exist_ok=True)

@dataclass
class NSConfig:
    site_id: str
    db_path: str
    oplog_path: str
    vector_dim: int = 512
    bind_host: str = '127.0.0.1'
    bind_port: int = 8373
    token: str = ''

def load_config() -> 'NSConfig':
    cfg_path = DEFAULT_HOME / 'config.yaml'
    if not cfg_path.exists():
        cfg = NSConfig(site_id=str(uuid.uuid4()), db_path=str(DEFAULT_HOME/'memory.db'), oplog_path=str(DEFAULT_HOME/'oplog.jsonl'))
        save_config(cfg)
        return cfg
    with open(cfg_path,'r') as f:
        raw = yaml.safe_load(f) or {}
    return NSConfig(site_id=raw.get('site_id'), db_path=raw.get('db_path', str(DEFAULT_HOME/'memory.db')),
                    oplog_path=raw.get('oplog_path', str(DEFAULT_HOME/'oplog.jsonl')),
                    vector_dim=int(raw.get('vector_dim',512)), bind_host=raw.get('bind_host','127.0.0.1'),
                    bind_port=int(raw.get('bind_port',8373)), token=raw.get('token',''))

def save_config(cfg:'NSConfig'):
    with open(DEFAULT_HOME/'config.yaml','w') as f: yaml.safe_dump(cfg.__dict__, f)
