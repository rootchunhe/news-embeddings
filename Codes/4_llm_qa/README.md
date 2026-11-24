# LLM Q&A (block 4)

Run from this folder with `../../env/bin/python completions.py` on a CPU node, with `HF_HOME` pointing to the Hugging Face cache that
holds the encoders' weights. No GPU is needed; Llama-2 (13B) takes about 60 GB of memory, and the run takes some minutes.

`completions.py` asks the three encoders the same factual prompts about facts known before 2000 (capitals, the largest company by
stock market value, the Federal Reserve chairman, Warren Buffett, Microsoft's chief executive, the New York Stock Exchange, IBM):
ChronoGPT-1999 (its official inference network with the pinned weights) and Llama-2 (13B) continue each prompt by greedy decoding,
RoBERTa-large fills a `<mask>` after the prompt (its five most likely tokens). Weights are the revisions used for the embeddings
(`1_data/03_embed.py`, `common/chronogpt.py`). Output: `Results/llm_qa/completions.txt`, the raw decoded text of every model and
prompt, which the paper quotes in a footnote on ChronoGPT-1999.

## The encoders

Parameters as counted by `completions.py`; cutoffs as in `3_paper/lib/constants.py` (`PRETRAIN_CUTOFF`) and for ChronoGPT its
vintage.

| | ChronoGPT-1999 | RoBERTa-large | Llama-2 (13B) |
|---|---|---|---|
| Hugging Face model | `manelalab/chrono-gpt-v1-19991231` | `roberta-large` | `NousResearch/Llama-2-13b-hf` |
| Revision | `45d6568340081add5c5b1444feb17a94b8f8199d` | `722cf37b1afa9454edce342e7895e588b6ff1d59` | `4b54e9fdd2ebd8db0901d37b98d3a53bbfaa4503` |
| Parameters (with the output layer) | 1,858,535,658 | 355,412,057 | 13,015,864,320 |
| Pretraining cutoff | December 1999 | July 2019 | September 2022 |
| Pretraining text | corpus of 7 billion tokens published before 2000 (He et al., 2026) | Liu et al. (2019) | Touvron et al. (2023) |
| Type | causal (next token) | masked (fills a blank) | causal (next token) |
| Answer here | greedy continuation, 20 tokens | five most likely tokens for `<mask>.` | greedy continuation, 20 tokens |

## Outputs

Copied from `Results/llm_qa/completions.txt` (rerun `completions.py` to regenerate). Continuations and RoBERTa tokens are shown with
repr, so line breaks appear as `\n`; RoBERTa's tokens carry their probabilities.

| Prompt | ChronoGPT-1999 | RoBERTa-large | Llama-2 (13B) |
|---|---|---|---|
| `'The capital of France is'` | ' the most\nbeautiful city in the world. 1t is\nbuilt on the banks of the' | ' Paris' 0.994, ' Lyon' 0.005, ' France' 0.000, ' Nice' 0.000, ' London' 0.000 | 'a city of contrasts. It is a city of history, culture, and art. It is' |
| `'The capital of Japan is'` | ' the most cos mopolitan city in the world. It is the home of the Japanese, Chinese' | ' Tokyo' 0.983, ' Kyoto' 0.011, ' Osaka' 0.006, ' Kobe' 0.000, ' Hiroshima' 0.000 | 'a city of contrasts. It is a city of the past and the future, of tradition and' |
| `'The company with the largest stock market value in the world is'` | " the United States Steel corporation. It's stock is worth $1, ,000, ,000," | ' Apple' 0.252, ' Amazon' 0.148, ' Microsoft' 0.123, ' Facebook' 0.091, ' Google' 0.047 | 'Apple. The company is worth $1.2 trillion.\nThe company with the largest' |
| `'The chairman of the Federal Reserve is'` | '\nsues the following statement:\n\n\nThe Federal Reserve Board has been\ninformed that the Federal' | ' retiring' 0.147, ' out' 0.074, ' gone' 0.066, ' back' 0.039, ' dead' 0.031 | 'the most powerful person in the world. He is the man who decides how much money is in' |
| `'Warren Buffett is the chairman of'` | ' the committee on the celebration of the one hundredth anniversary of the birth of Abraham Lincoln, which will' | ' Berkshire' 0.907, ' Inc' 0.045, ' Buffett' 0.004, ' the' 0.004, ' Alphabet' 0.004 | 'Berkshire Hathaway, a conglomerate holding company that owns more than ' |
| `'The chief executive of Microsoft is'` | '\na man who has been in the business\nof making automobiles for more\nthan 20 years.' | ' retiring' 0.382, ' out' 0.137, ' gone' 0.048, ' back' 0.043, ' leaving' 0.042 | 'to step down from his role in the summer, the company has announced.\nSteve Ballmer' |
| `'The New York Stock Exchange is located on'` | ' the site of the old New York City Hall. The first building was erected in 1875. The' | ' Manhattan' 0.590, ' Broadway' 0.373, ' Hudson' 0.016, ' downtown' 0.003, ' below' 0.002 | 'Wall Street in New York City.\nThe New York Stock Exchange (NYSE) is the' |
| `'IBM is a company that makes'` | "\nmoney.\n\n\nThe company's earnings in\n1860 were $1,000, 900" | ' computers' 0.636, ' software' 0.159, ' things' 0.035, ' money' 0.028, ' stuff' 0.018 | 'a lot of money. It is a company that has a lot of employees. It is a company' |
