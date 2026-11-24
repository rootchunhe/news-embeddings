"""Factual prompts for the three encoders: what each pretrained language model says about facts known before 2000. Run from Codes/4_llm_qa/
on a CPU node (no GPU; Llama-2 needs about 60 GB of memory, the whole run takes some minutes):
    HF_HOME=<Hugging Face cache> ../../env/bin/python completions.py
Llama-2 (13B) and ChronoGPT-1999 continue each prompt by greedy decoding (20 new tokens); RoBERTa-large, a masked language model,
fills a <mask> placed after the prompt (its five most likely tokens). Weights are the pinned revisions used for the embeddings, in each
model's default precision on CPU. Output: Results/llm_qa/completions.txt, the raw decoded text of every model and prompt."""
import importlib.util
import platform
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, GPT2TokenizerFast

CODES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODES))
from common import chronogpt                                                    # noqa: E402

PROMPTS = ["The capital of France is", "The capital of Japan is",
           "The company with the largest stock market value in the world is", "The chairman of the Federal Reserve is",
           "Warren Buffett is the chairman of", "The chief executive of Microsoft is",
           "The New York Stock Exchange is located on", "IBM is a company that makes"]
NEW_TOKENS, TOP_K = 20, 5
LLAMA, ROBERTA, CHRONO = "NousResearch/Llama-2-13b-hf", "roberta-large", "manelalab/chrono-gpt-v1-19991231"
REVISION = {LLAMA: "4b54e9fdd2ebd8db0901d37b98d3a53bbfaa4503", ROBERTA: "722cf37b1afa9454edce342e7895e588b6ff1d59",
            CHRONO: chronogpt.REVISION[CHRONO]}                                 # as in 1_data/03_embed.py and common/chronogpt.py
OUT = CODES.parent / "Results/llm_qa/completions.txt"


def chrono():
    """ChronoGPT-1999 with its output layer: the official inference network of the pinned revision, loaded with the same weights"""
    src = hf_hub_download(CHRONO, "ChronoGPT_inference.py", revision=REVISION[CHRONO])
    spec = importlib.util.spec_from_file_location("chrono_inference", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.ChronoGPT(**chronogpt.config(CHRONO))
    net.load_state_dict(chronogpt.load(CHRONO, "cpu").state_dict())
    tok = GPT2TokenizerFast.from_pretrained(chronogpt.TOKENIZER[0], revision=chronogpt.TOKENIZER[1])

    def complete(prompt):
        x = torch.tensor([[chronogpt.EOT] + tok(prompt, add_special_tokens=False)["input_ids"]])
        n = x.shape[1]
        for _ in range(NEW_TOKENS):                                              # greedy; the network has no cache, so each step reruns the text
            x = torch.cat([x, net(x)[0][0, -1].argmax().view(1, 1)], dim=1)
        return repr(tok.decode(x[0, n:].tolist(), clean_up_tokenization_spaces=False))
    return net.eval(), complete


def llama():
    tok = AutoTokenizer.from_pretrained(LLAMA, revision=REVISION[LLAMA])
    net = AutoModelForCausalLM.from_pretrained(LLAMA, revision=REVISION[LLAMA], torch_dtype=torch.float32).eval()

    def complete(prompt):
        x = tok(prompt, return_tensors="pt")
        y = net.generate(**x, max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
        return repr(tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    return net, complete


def roberta():
    tok = AutoTokenizer.from_pretrained(ROBERTA, revision=REVISION[ROBERTA])
    net = AutoModelForMaskedLM.from_pretrained(ROBERTA, revision=REVISION[ROBERTA], torch_dtype=torch.float32).eval()

    def complete(prompt):
        x = tok(f"{prompt} {tok.mask_token}.", return_tensors="pt")
        p = net(**x).logits[0, (x["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]].softmax(-1)
        top = p.topk(TOP_K)
        return ", ".join(f"{tok.decode([int(i)], clean_up_tokenization_spaces=False)!r} {float(q):.3f}" for q, i in zip(top.values, top.indices))
    return net, complete


def main():
    torch.manual_seed(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# torch {torch.__version__}, transformers {transformers.__version__}, {platform.machine()}, CPU threads {torch.get_num_threads()}",
             f"# greedy continuation of {NEW_TOKENS} tokens (ChronoGPT, Llama-2); RoBERTa: the {TOP_K} most likely tokens for '<mask>.' after the prompt",
             "# decoded without clean-up; each continuation and each RoBERTa token is printed with repr, so line breaks appear as \\n"]
    for name, build in [("ChronoGPT-1999", chrono), ("RoBERTa-large", roberta), ("Llama-2 (13B)", llama)]:
        model = {"ChronoGPT-1999": CHRONO, "RoBERTa-large": ROBERTA, "Llama-2 (13B)": LLAMA}[name]
        net, complete = build()
        lines += ["", f"== {name}: {model} @ {REVISION[model]}, {sum(p.numel() for p in net.parameters()):,} parameters"]
        with torch.inference_mode():
            for prompt in PROMPTS:
                lines += [f"PROMPT {prompt!r}", f"OUTPUT {complete(prompt)}"]
                print(lines[-2], lines[-1], sep="\n", flush=True)
        del net
    OUT.write_text("\n".join(lines) + "\n")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
