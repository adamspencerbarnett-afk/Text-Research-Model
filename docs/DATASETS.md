# Question-and-answer datasets for the reasoning work

Fetched 9 October 2026 into `data_uploads/` (not in git; re-fetch with the URLs below). Each has a
different kind of answer, which is the point: the check loop needs answers that can be verified.

| Set | What it tests | Answer type | Size | Licence | Source |
| --- | --- | --- | --- | --- | --- |
| Alpaca | instruction following, open answers | free text | 52,002 | CC BY-NC 4.0 (research) | tatsu-lab/stanford_alpaca `alpaca_data.json` |
| GSM8K | multi-step arithmetic word problems | one number | 7,473 / 1,319 | MIT | openai/grade-school-math `train.jsonl`, `test.jsonl` |
| SVAMP | one-step arithmetic with distractors | one number + equation | 1,000 | MIT | arkilpatel/SVAMP `SVAMP.json` |
| SQuAD 1.1 | reading comprehension over a given passage | span of the passage | 87,599 / 10,570 | CC BY-SA 4.0 | rajpurkar.github.io/SQuAD-explorer `train-v1.1.json`, `dev-v1.1.json` |
| ARC (Easy + Challenge) | grade-school science, multiple choice | one letter | 7,787 | CC BY-SA 4.0 | ai2-public-datasets.s3.amazonaws.com/arc `ARC-V1-Feb2018.zip` (includes the 14M-sentence ARC corpus) |
| OpenBookQA | science facts + reasoning, multiple choice, with the "book" of facts | one letter | 5,957 | Apache 2.0 | ai2-public-datasets.s3.amazonaws.com/open-book-qa |
| CommonsenseQA | common-sense multiple choice | one letter | 9,741 / 1,221 | MIT (questions); ConceptNet | s3.amazonaws.com/commensenseqa |
| Dolly 15k | instructions in 8 categories (closed QA, summarisation, classification, …) | free text, some with context | 15,011 | CC BY-SA 3.0 | databricks/databricks-dolly-15k |

Not fetched yet: bAbI (the fbaipublicfiles URL no longer serves it; mirror needed) and BoolQ (the
Google bucket is closed; available on Hugging Face as parquet).

Conversion: `compiler/experiments/prepare_qa.py` turns Alpaca (`--format alpaca`) and GSM8K
(`--format gsm8k`, final line `Answer: N`) into the chat text the model trains on. The others need
their own `--format`; add one per set as it is first used, keeping the checkable answer on a final
`Answer:` line so `m2_qa.py`'s numeric exact-match scorer and the check loop can read it.

Why these: GSM8K and SVAMP give the number channel and the executor an exact verdict; SQuAD gives
the copy head a passage to copy from where the answer is a span (the cleanest test of
copy-from-evidence); ARC, OpenBookQA and CommonsenseQA give one-letter answers that can be scored
exactly without a calculator; Dolly adds answer types Alpaca lacks.
