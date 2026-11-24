"""Access to the headline embeddings produced by Codes/1_data/03_embed.py (chunked by hid)."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; FILE_ROWS = 500_000          # headlines per embedding file (emb_NNN.npy)

SPEC = ["model", "revision", "tokenizer", "max_len", "pooling", "hidden_size"]     # how a chunk was embedded: the same for every chunk

def emb_dir(model): return ROOT / "Data/embedding" / model.split("/")[-1]

def headlines():
    """the unique headlines of Codes/1_data/02_headlines.py, row i = hid i, a missing headline as the empty string"""
    import pandas as pd
    H = pd.read_parquet(ROOT / "Data/headlines/headlines.parquet", columns=["hid", "headline"])
    assert (H.hid.values == np.arange(len(H))).all(), "the headline table is not ordered by hid"
    return H.headline.fillna("").tolist()

_verified = set()
def verify(model):
    """once per process: the sha1 of every 500k-headline chunk of Data/headlines/headlines.parquet must equal the sha1 recorded in the
    chunk's embedding manifest, so a rebuilt headline table can never be read against stale vectors, and every chunk must have the
    same encoder SPEC"""
    if model in _verified: return
    import hashlib
    H = headlines()
    npys = {f.stem for f in emb_dir(model).glob("emb_*.npy")}; jsons = {f.stem for f in emb_dir(model).glob("emb_*.json")}
    if npys != jsons: raise RuntimeError(f"{emb_dir(model)}: arrays without manifest {sorted(npys - jsons)} / manifests without array {sorted(jsons - npys)}")
    for mf in sorted(emb_dir(model).glob("emb_*.json")):
        j = json.loads(mf.read_text()); sha = hashlib.sha1("\n".join(H[j["hid_lo"]:j["hid_hi"]]).encode()).hexdigest()
        if sha != j["text_sha1"]: raise RuntimeError(f"{mf}: headline table changed since this chunk was embedded (re-run 03_embed.py)")
    encoder_spec(model); _verified.add(model)

def encoder_spec(model):
    """the encoder SPEC of the headline chunks (model revision, tokenizer, max length, pooling), checked to be one for all chunks"""
    specs = {json.dumps({k: m.get(k) for k in SPEC}, sort_keys=True) for m in manifests(model)}
    if len(specs) != 1: raise RuntimeError(f"{emb_dir(model)}: chunks embedded with {len(specs)} different encoder settings")
    return json.loads(specs.pop())

def load_emb(model, hids, dtype=np.float16):
    """embeddings for an array of hids, in that order (reads the chunk files via mmap; missing chunks raise)"""
    verify(model); hids = np.asarray(hids, dtype=np.int64); out = None; d = emb_dir(model)
    for ch in np.unique(hids // FILE_ROWS):
        X = np.load(d / f"emb_{ch:03d}.npy", mmap_mode="r"); sel = np.where(hids // FILE_ROWS == ch)[0]; rows = hids[sel] - ch * FILE_ROWS; o = np.argsort(rows)
        if out is None: out = np.empty((len(hids), X.shape[1]), dtype=dtype)
        out[sel[o]] = np.asarray(X[rows[o]], dtype=dtype)
    return out

def manifests(model): return [json.loads(f.read_text()) for f in sorted(emb_dir(model).glob("emb_*.json"))]
