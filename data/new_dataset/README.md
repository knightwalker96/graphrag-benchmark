# Hackathon Resources

| Path | Contents |
|---|---|
| `corpus/corpus.jsonl` | 2,951 documents, ~5,471,565 tokens |
| `questions/eval_public.jsonl` | 100 questions, with answers |
| `questions/eval_hidden.jsonl` | 50 questions, without answers |

All three files are JSONL — one JSON object per line. Open them and look.

## The corpus is the only source of truth

Answers are defined over these documents — not over the real world, and not
over what a model remembers. If Wikipedia today disagrees with
`corpus.jsonl`, the corpus wins.

Documents are English Wikipedia articles converted to plain text. Coverage
spans 1987–2023 and the subject matter is narrow, so read some documents
before you design anything.

## Attribution

Corpus text is derived from English Wikipedia and is licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each document
carries its source URL.
