"""Step 3: frozen-LLM mean-pooled embeddings of the unique headlines (CKX Sec. 2.2: last hidden layer, mean over tokens).
One GPU job per chunk of CHUNK headlines (e.g. a job array). Direct use (from Codes/1_data/):
    ../../env/bin/python 03_embed.py --chunk 0 [--model NousResearch/Llama-2-13b-hf | roberta-large | manelalab/chrono-gpt-v1-19991231]
ChronoGPT (the robustness encoder, common/chronogpt.py) has its own network and tokenizer; the pooling is the same.
In : Data/headlines/headlines.parquet
Out: Data/embedding/<model_tag>/emb_<chunk:03d>.npy   float16 (CHUNK x d), row i = headline hid chunk*CHUNK + i
     Data/embedding/<model_tag>/emb_<chunk:03d>.json  manifest: hid range, model + revision, tokenizer, max_len, pooling,
                                                      sha1 of the headline strings, count of headlines at max_len, runtime
Writes to a temp file and renames, so a file that exists is complete; a chunk is skipped only if its json manifest matches."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Codes"))
from common import chronogpt
from common.emb import FILE_ROWS as CHUNK, headlines
REVISION = {"NousResearch/Llama-2-13b-hf": "4b54e9fdd2ebd8db0901d37b98d3a53bbfaa4503", "roberta-large": "722cf37b1afa9454edce342e7895e588b6ff1d59"}   # the Hugging Face commits embedded
p = argparse.ArgumentParser()
p.add_argument("--chunk", type=int, required=True); p.add_argument("--model", default="NousResearch/Llama-2-13b-hf")
p.add_argument("--max_len", type=int, default=128); p.add_argument("--batch_tokens", type=int, default=16384)
a = p.parse_args(); tag = a.model.split("/")[-1]; out_dir = ROOT / "Data/embedding" / tag; out_dir.mkdir(parents=True, exist_ok=True)
H = headlines()
lo, hi = a.chunk * CHUNK, min((a.chunk + 1) * CHUNK, len(H))
if lo >= len(H): print("chunk beyond the table, nothing to do"); raise SystemExit
texts = H[lo:hi]
sha = hashlib.sha1("\n".join(texts).encode()).hexdigest()
emb_fn, mf_fn = out_dir / f"emb_{a.chunk:03d}.npy", out_dir / f"emb_{a.chunk:03d}.json"
chrono = chronogpt.is_chrono(a.model)
spec = {"hid_lo": lo, "hid_hi": hi, "n": hi - lo, "model": a.model, "max_len": a.max_len, "text_sha1": sha,
        "pooling": "mean over non-pad tokens of the last hidden state, fp32 accumulate, stored fp16"}
spec.update(chronogpt.spec(a.model) if chrono else {"revision": REVISION[a.model]})   # the encoder identity: a chunk of another one is redone
if emb_fn.exists() and mf_fn.exists() and all(json.loads(mf_fn.read_text()).get(k) == v for k, v in spec.items()): print("chunk complete, skipping", emb_fn); raise SystemExit
dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
if chrono:
    net = chronogpt.load(a.model, dev); spec.update({"torch": torch.__version__, "device": torch.cuda.get_device_name() if dev == "cuda" else dev})
    print(f"chunk {a.chunk}: hids {lo}-{hi} ({hi - lo:,} headlines) | {a.model} on {dev}", flush=True)
    ids = chronogpt.tokenize(texts, a.max_len); lens = np.array([len(x) for x in ids])
    out = chronogpt.pool(net, ids, a.batch_tokens, log=lambda i: print(f"{i}/{len(ids)} {time.time() - t0:.0f}s", flush=True)).astype(np.float16)
else:
    tok = AutoTokenizer.from_pretrained(a.model, revision=REVISION[a.model])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.model, revision=REVISION[a.model], torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    spec.update({"hidden_size": model.config.hidden_size, "tokenizer": tok.__class__.__name__, "torch": torch.__version__,
                 "device": torch.cuda.get_device_name() if dev == "cuda" else dev})
    print(f"chunk {a.chunk}: hids {lo}-{hi} ({hi - lo:,} headlines) | {a.model} on {dev}", flush=True)
    enc = tok(texts, truncation=True, max_length=a.max_len); lens = np.array([len(x) for x in enc["input_ids"]])
    order = np.argsort(lens); out = np.zeros((hi - lo, model.config.hidden_size), dtype=np.float16); i = nb = 0
    with torch.inference_mode():
        while i < len(order):
            bs = min(1024, max(1, a.batch_tokens // int(lens[order[min(i + 1023, len(order) - 1)]]))); idx = order[i:i + bs]
            b = tok.pad({"input_ids": [enc["input_ids"][j] for j in idx]}, return_tensors="pt").to(dev)
            h = model(**b).last_hidden_state.float(); m = b["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16); i += bs; nb += 1
            if nb % 100 == 0: print(f"{i}/{len(order)} {time.time() - t0:.0f}s", flush=True)
assert np.isfinite(out).all(), "non-finite embeddings"; assert (np.abs(out).sum(1) > 0).all(), "zero vectors"
spec.update({"truncated": int((lens >= a.max_len).sum()), "len_p99": int(np.percentile(lens, 99)), "secs": round(time.time() - t0), "finished": time.strftime("%Y-%m-%d %H:%M")})
mf_fn.unlink(missing_ok=True)                                        # a vector file without its manifest is refused by common/emb.py
tmp = emb_fn.with_name(emb_fn.name + ".tmp.npy"); np.save(tmp, out); os.replace(tmp, emb_fn)
tmp = mf_fn.with_name(mf_fn.name + ".tmp"); tmp.write_text(json.dumps(spec, indent=1)); os.replace(tmp, mf_fn)
print("done", emb_fn, out.shape, f"{time.time() - t0:.0f}s", flush=True)
