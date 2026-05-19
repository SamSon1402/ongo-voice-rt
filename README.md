# OngoVoice-RT

> Real-time, edge-first conversational pipeline for the Ongo companion robot.
> Built for InteractionLabs by Sameer M. — Paris, May 2026.

<img width="1356" height="908" alt="image" src="https://github.com/user-attachments/assets/3ffa55c0-99e2-4fce-9420-556240b4eaaa" />
<img width="690" height="338" alt="image" src="https://github.com/user-attachments/assets/3d411c94-cde0-4944-a36e-35ec3ae20c79" />
<img width="869" height="856" alt="image" src="https://github.com/user-attachments/assets/7111296d-0533-4e7b-93ad-3300b831cfd7" />


A streaming voice loop where **wake-word, VAD, and short-utterance ASR
run on-device**, and a cloud LLM is only woken when intent confidence
drops below threshold. End-to-end time-to-first-token is the contract.

```
  mic ─► VAD ─► wake-word ─► ASR ─► Router ─► [edge LLM | cloud LLM] ─► TTS ─► speaker
                                       │
                                  conf gate
                                  budget gate
                                  privacy gate
```

The router is the heart of the system. Most of Ongo's turns ("set a
5-min timer", "next song", "what time is it") never leave the device.
Open questions and ambiguous intents go to the cloud. We measure the
edge-resolved ratio and dollar cost per device-month live.

---

## Why this exists

> *"Edge ML for Robotics: fine-tuning and deploying CV and audio
> models on-device for real-world robotics use cases. Infrastructure
> & Benchmarking: implementing and maintaining integrations with LLM
> and audio model providers."*
> — InteractionLabs, Founding ML Engineer JD

Always-cloud is too slow and too expensive for an always-on companion.
Always-edge is too dumb. The interesting work is in the routing layer
and the streaming primitives that make TTFT feel instant.

---

## Quick start

```bash
make install      # poetry / pip install -e ".[dev]"
make test         # pytest -q  → 38 tests
make api          # uvicorn ongovoice.api:app  (port 8001)
make demo         # send a turn through the pipeline from the CLI
```

A single turn from the CLI:

```bash
ongovoice say "Hey Ongo, set a five minute timer."
# → router=edge  ttft=87ms  cost=$0.000  "Timer set — five minutes."

ongovoice say "Hey Ongo, what's on my calendar?"
# → router=cloud(anthropic/haiku-4.5)  ttft=418ms  cost=$0.0014
```

---

## Layout

```
src/ongovoice/
  core/        # PCM frames, transcripts, intents, errors, protocols
  audio/       # VAD, wake-word, ring buffer
  asr/         # whisper-tiny edge + cloud fallback
  router/      # confidence + budget + privacy gates
  llm/         # Anthropic / OpenAI / Mistral / local providers
  tts/         # streaming TTS surface
  pipeline/    # the conversation manager that wires it all together
  api/         # FastAPI + WebSocket bidi audio
  telemetry/   # turn-level metrics + cost tracking
```

---

## Design notes

**Async all the way down.** Every step is an `async def` returning either
a value or an `AsyncIterator`. The TTFT contract is meaningless if the
pipeline blocks anywhere.

**Streaming is the default.** LLM providers return `AsyncIterator[Token]`,
TTS consumes `AsyncIterator[str]`. We never accumulate-then-flush — first
token out of the LLM is the first audio frame queued for TTS.

**Provider abstraction is real.** `LLMProvider` is a Protocol with a
single `stream()` method. Adding a new provider is a 60-line class.
The A/B harness measures TTFT and cost per provider in production.

**Routing is data, not branches.** A `RoutingDecision` is a value
object carrying the gate that fired and why. Every decision is logged
so you can audit *why* a turn went to cloud six weeks later.

**Barge-in is first-class.** If the user starts talking while Ongo
is speaking, we cancel the TTS stream and the LLM stream upstream.
The pipeline manager owns those `asyncio.Task` handles for exactly
this reason.

---

## Status

| Area | State |
|---|---|
| Domain types & protocols | ✅ done |
| Async ring buffer + VAD glue | ✅ done |
| Mock ASR + edge whisper-tiny adapter | ✅ done / 🟡 scaffolded |
| Router (confidence + budget + privacy) | ✅ done |
| Anthropic / OpenAI / Mock LLM providers | ✅ done |
| Mistral / local llama providers | 🟡 scaffolded |
| Streaming TTS (mock + Piper) | ✅ done / 🟡 scaffolded |
| Pipeline manager + barge-in | ✅ done |
| FastAPI HTTP surface | ✅ done |
| WebSocket bidi audio | ✅ done |
| Turn-level cost & TTFT tracking | ✅ done |

🟡 items are vendor-specific and need the real SDK / weights to fill in.
The Protocol seams are correct.

---

— Sameer M. · samson1402.github.io · sameer@…
