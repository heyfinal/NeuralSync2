from fastapi import FastAPI, Depends
from .config import load_config
from .storage import connect, upsert_item, recall, put_persona, get_persona
from .api import MemoryIn, RecallIn, PersonaIn
from .utils import now_ms, redact
from .crdt import Version
from .oplog import OpLog
from .auth import bearer_guard
import uuid, json

cfg = load_config()
con = connect(cfg.db_path)
oplog = OpLog(cfg.oplog_path)
app = FastAPI(title='NeuralSync')

LAMPORT = 0

def tick():
    global LAMPORT
    LAMPORT += 1
    return LAMPORT

@app.get('/health')
async def health():
    return {'ok': True, 'site_id': cfg.site_id}

@app.get('/persona', dependencies=[Depends(bearer_guard)])
async def persona_get():
    return get_persona(con)

@app.post('/persona', dependencies=[Depends(bearer_guard)])
async def persona_set(body: PersonaIn):
    ver = Version(lamport=tick(), site_id=cfg.site_id)
    put_persona(con, body.text, ver.site_id, ver.lamport)
    oplog.append({'op':'persona_set','lamport':ver.lamport,'site_id':ver.site_id,'clock_ts':now_ms(),'payload':{'text':body.text}})
    return {'ok': True}

@app.post('/remember', dependencies=[Depends(bearer_guard)])
async def remember_post(body: MemoryIn):
    now = now_ms()
    item = {
        'id': str(uuid.uuid4()),
        'kind': body.kind,
        'text': redact(body.text),
        'scope': body.scope,
        'tool': body.tool,
        'tags': json.dumps(body.tags, ensure_ascii=False),
        'confidence': body.confidence,
        'benefit': body.benefit,
        'consistency': body.consistency,
        'vector': None,
        'created_at': now,
        'updated_at': now,
        'ttl_ms': body.ttl_ms,
        'expires_at': (now + body.ttl_ms) if body.ttl_ms else None,
        'tombstone': 0,
        'site_id': cfg.site_id,
        'lamport': tick(),
        'source': body.source or 'api',
        'meta': json.dumps(body.meta or {}, ensure_ascii=False),
    }
    out = upsert_item(con, item)
    
    # Create JSON-safe payload for oplog (exclude binary data)
    payload = {k: v for k, v in out.items() if k != 'vector'}
    oplog.append({'op':'add','id':out['id'],'lamport':out['lamport'],'site_id':out['site_id'],'clock_ts':now_ms(),'payload':payload})
    
    # Return JSON-safe response
    response = {k: v for k, v in out.items() if k != 'vector'}
    return response

@app.post('/recall', dependencies=[Depends(bearer_guard)])
async def recall_post(body: RecallIn):
    raw_items = recall(con, body.query, body.top_k, body.scope, body.tool)
    
    # Filter out binary data for JSON serialization
    items = []
    for item in raw_items:
        clean_item = {k: v for k, v in item.items() if k != 'vector'}
        items.append(clean_item)
    
    persona = get_persona(con).get('text','')
    pre = ''
    if persona: pre += f"Persona: {persona}\n\n"
    for i,it in enumerate(items,1):
        pre += f"[M{i}] ({it['kind']}, {it['scope']}, conf={it.get('confidence','')}) {it['text']}\n"
    return {'items': items, 'preamble': pre}

@app.post('/sync/pull', dependencies=[Depends(bearer_guard)])
async def sync_pull(payload: dict):
    since = int(payload.get('since') or 0)
    ops = []
    cursor = since
    for pos, op in oplog.read_since(since):
        ops.append({'pos': pos, 'op': op})
        cursor = pos
    return {'cursor': cursor, 'ops': ops}

@app.post('/sync/push', dependencies=[Depends(bearer_guard)])
async def sync_push(payload: dict):
    ops = payload.get('ops', [])
    applied = 0
    for rec in ops:
        op = rec.get('op') or rec
        typ = op.get('op')
        if typ == 'persona_set':
            put_persona(con, op['payload']['text'], op['site_id'], op['lamport'])
            applied += 1
        elif typ in ('add','update'):
            upsert_item(con, op.get('payload', {}))
            applied += 1
        elif typ == 'delete':
            from .storage import delete_item
            delete_item(con, op['id'])
            applied += 1
    return {'applied': applied}
