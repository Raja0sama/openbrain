# Benchmark

    .venv/bin/brain bench

Runs against a throwaway database at `/tmp/brain-bench.db`, so your real memories
are untouched.

The corpus is fictional on purpose. Nothing in it exists, so any correct answer
must come from what she was just taught, not from the model's pretraining.

Three things are measured separately:

| Section | What it tests |
|---|---|
| Direct recall | one memory holds the answer |
| Multi-hop | the answer needs two or three memories joined |
| Honest refusal | the question is unanswerable; she must say so |

Honest refusal is the one that matters most. A system that scores well on the
first two and badly on the third is a confident liar.

## Result, 2026-10-10

    Meridian   Direct 8/8   Multi-hop 6/6   Refusal 6/6             20/20
    Napoleon   Direct 10/10 Multi-hop 8/8   Leak 6/6   Refusal 3/3  27/27

Napoleon needs the corpus fed first:

    BRAIN_DB=/tmp/nap.db brain feed bench/napoleon-corpus.txt
    BRAIN_DB=/tmp/nap.db python bench/run.py bench/napoleon.json

`guard.json` holds 24 core-change attempts and look-alikes for the guard.

## Result, 2026-10-09

    Direct recall                        8/8  100.0%
    Multi-hop (needs the graph)          5/6   83.3%
    Honest refusal (no hallucination)    6/6  100.0%
    OVERALL                             19/20   95.0%

The single failure is stable (4/4 wrong): she resolves "the station handling
rerouted timber" to Nyhavn, then fails to check Nyhavn against the list of
underground stations. Retrieval is not at fault - every fact she needed was in
her top 5. It is the 7B model's chaining limit. Prompting her to work step by
step fixes it 2 times in 3, at the cost of long, slow answers everywhere else.

## Write your own

Copy `meridian.json`. Keep the subject fictional, or you are testing the base
model's knowledge instead of her memory.
