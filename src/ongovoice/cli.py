"""Command-line interface."""

from __future__ import annotations

import asyncio

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ongovoice.asr.mock_asr import MockASR
from ongovoice.core import AudioFrame
from ongovoice.llm.mock_provider import MockProvider
from ongovoice.pipeline import ConversationManager
from ongovoice.router import IntentClassifier, RouterPolicy
from ongovoice.tts.mock_tts import MockTTS

app = typer.Typer(name="ongovoice", no_args_is_help=True)
console = Console()


@app.command()
def say(
    text: str = typer.Argument(..., help="Pretend the user said this"),
    confidence: float = typer.Option(0.92, "--confidence", "-c"),
) -> None:
    """Drive one turn through the pipeline from a transcript."""
    asyncio.run(_run_one(text, confidence))


async def _run_one(text: str, confidence: float) -> None:
    asr = MockASR(final_text=text, final_confidence=confidence)
    edge_llm = MockProvider(canned_response="Done.")
    cloud_llm = MockProvider(
        canned_response="Let me think — I'll get you an answer shortly.",
        first_token_ms=380,
    )
    tts = MockTTS()

    manager = ConversationManager(
        asr=asr,
        edge_llm=edge_llm,
        cloud_llm=cloud_llm,
        tts=tts,
        router=RouterPolicy(),
        classifier=IntentClassifier(),
    )

    audio = _silent_audio(600.0)
    turn = await manager.handle_turn(audio)

    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="cyan")
    t.add_row("turn_id", turn.turn_id)
    t.add_row("user", turn.user_text)
    t.add_row("ongo", turn.assistant_text)
    t.add_row("routing", f"{turn.metrics.routing.target.value}  ({turn.metrics.routing.intent.value})")
    t.add_row("reason", turn.metrics.routing.reason)
    t.add_row("ttft", f"{turn.metrics.ttft_ms:.1f} ms")
    t.add_row("asr", f"{turn.metrics.asr_ms:.1f} ms")
    t.add_row("llm 1st tok", f"{turn.metrics.llm_first_token_ms:.1f} ms")
    t.add_row("cost", f"${turn.metrics.cost_usd:.5f}")
    t.add_row("edge resolved", "✓" if turn.metrics.edge_resolved else "✗")

    label = "[green]EDGE[/green]" if turn.metrics.edge_resolved else "[blue]CLOUD[/blue]"
    console.print(Panel(t, title=f"OngoVoice-RT · {label}", border_style="dim"))


async def _silent_audio(duration_ms: float):
    frame_ms = 30.0
    samples = int(16000 * frame_ms / 1000)
    n = int(duration_ms / frame_ms)
    for _ in range(n):
        yield AudioFrame(sample_rate=16000, pcm=np.zeros(samples, dtype=np.int16))
        await asyncio.sleep(0)


if __name__ == "__main__":
    app()
