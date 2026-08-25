# Test fixtures

## `mix_2spk.wav`, `ref_spkA.wav`, `ref_spkB.wav`

A two-speaker overlapping mixture and its two reference sources, used by the
Phase 0 pretrained-model smoke test (`scripts/smoke_pretrained.py`).

Built from two utterances by different speakers in LibriSpeech `dev-clean`:

| File | Source utterance |
|---|---|
| `ref_spkA.wav` | `1988/147956/1988-147956-0018` |
| `ref_spkB.wav` | `2035/147960/2035-147960-0013` |

`mix_2spk.wav` sums them at 0.9 and 0.7 gain with speaker B delayed 2000 ms,
so the clip contains both single-talker and overlapped regions — the two cases
the pipeline has to tell apart (ADR-0010).

16 kHz, mono, float PCM.

**Licence:** derived from LibriSpeech, CC BY 4.0 (Panayotov et al., 2015).
Redistribution is permitted with attribution; see `docs/06-datasets.md` §2 and
[ADR-0013](../../docs/adr/0013-dataset-licensing.md).
