"""Embeds the category templates (one fixed sentence per RavenPack category) or a text table the way the headlines were embedded
(same model revision, max length, mean pooling; Codes/1_data/03_embed.py). Run from Codes/3_paper/ on a GPU:
    ../../env/bin/python 2_embed.py --model roberta-large [--texts ../../Data/clean/perturb_texts.parquet]
Output Data/clean/template_emb_<model>.npz + run_template_emb_<model>.json, or Data/clean/<stem>_emb_<model>.npz + run_<stem>_emb_<model>.json"""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from lib import ROOT
from lib.constants import ENCODERS
from lib.ridge import category_index

from common import chronogpt
from common.emb import encoder_spec
from common.runinfo import write_manifest

BATCH_TOKENS = 16384


def template(category):
    return "Company " + category.replace("-", " ")


def embed(texts, model):
    """mean over the non-pad tokens of the last hidden state, with the headline chunks' model revision and max length;
    batches are sorted by length as in 03_embed.py"""
    spec = encoder_spec(model)
    max_len, revision = int(spec["max_len"]), spec.get("revision")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if chronogpt.is_chrono(spec["model"]):                               # its own network and tokenizer, the same pooling
        now = chronogpt.spec(spec["model"])
        assert all(spec[k] == v for k, v in now.items()), f"the headlines were embedded with another ChronoGPT encoder: {spec} vs {now}"
        net = chronogpt.load(spec["model"], dev)
        emb = chronogpt.pool(net, chronogpt.tokenize(texts, max_len), BATCH_TOKENS).astype(np.float16)
        assert np.isfinite(emb).all()
        return emb, dev, revision, max_len
    tok = AutoTokenizer.from_pretrained(spec["model"], revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # padding side as the tokenizer sets it, as 03_embed.py; pad tokens are masked out of the mean
    net = AutoModel.from_pretrained(spec["model"], revision=revision, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    enc = tok(texts, truncation=True, max_length=max_len)["input_ids"]
    lens = np.array([len(x) for x in enc])
    order = np.argsort(lens)
    out = np.zeros((len(texts), net.config.hidden_size), np.float16)
    i = 0
    with torch.inference_mode():
        while i < len(order):
            bs = min(1024, max(1, BATCH_TOKENS // int(lens[order[min(i + 1023, len(order) - 1)]])))    # as 03_embed.py
            idx = order[i:i + bs]
            b = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt").to(dev)
            h = net(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16)
            i += bs
    assert np.isfinite(out).all()
    return out, dev, revision, max_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-large", choices=list(ENCODERS))
    p.add_argument("--texts", default=None, help="parquet with a 'text' column to embed instead of the templates")
    a = p.parse_args()
    t0 = time.time()
    if a.texts:
        T = pd.read_parquet(a.texts)
        texts = T.text.fillna("").tolist()
        emb, dev, revision, max_len = embed(texts, a.model)
        stem = Path(a.texts).stem
        out = ROOT / f"Data/clean/{stem}_emb_{a.model}.npz"
        np.savez(out.with_suffix(".tmp.npz"), emb=emb)
        out.with_suffix(".tmp.npz").replace(out)
        write_manifest(out.with_name(f"run_{stem}_emb_{a.model}.json"), vars(a), rows=len(T), device=dev, revision=revision, max_len=max_len,
                       texts_sha1=hashlib.sha1("\n".join(texts).encode()).hexdigest(), seconds=round(time.time() - t0))
        print(f"{len(T)} texts embedded with {a.model} on {dev} -> {out} [{time.time() - t0:.0f}s]")
        return
    cats = category_index()
    texts = [template(c) for c in cats]
    emb, dev, revision, max_len = embed(texts, a.model)
    out = ROOT / f"Data/clean/template_emb_{a.model}.npz"
    np.savez(out.with_suffix(".tmp.npz"), categories=np.array(cats.tolist()), templates=np.array(texts), emb=emb)
    out.with_suffix(".tmp.npz").replace(out)
    write_manifest(out.with_name(f"run_template_emb_{a.model}.json"), vars(a), categories=len(cats), device=dev, revision=revision, max_len=max_len,
                   seconds=round(time.time() - t0))
    print(f"{len(cats)} templates embedded with {a.model} on {dev} -> {out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
