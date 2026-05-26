"""Tests for memory consolidator output parsing (issue #63).

The consolidator's regex was strict about its `- [Importance: N] [Emotional valence: V]`
shape, but llama3.2:3b reliably wraps the brackets in markdown emphasis (`**...**`),
which silently routed every line to the fallback path that strips metadata and stores
the summary with default `(importance=5.0, valence=0.0)`. This was observed across
multiple Sage/Rafael runs (8 "stored 0 memories" warnings in a single 80-minute Sage
run alone).

The fix strips markdown emphasis BEFORE the prefix + regex check, so the dominant
real-world LLM output now parses cleanly with its original metadata. Verified here
against fixture lines lifted directly from `~/.consciousness/Sage/run.log`.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from memory.consolidator import MemoryConsolidator


# ---------------------------------------------------------------------------
# Fakes — just enough to drive consolidate_once() without touching disk or LLMs
# ---------------------------------------------------------------------------

class _FakeEpisodicEvent:
    def __init__(self, kind: str, content: str) -> None:
        self.kind = kind
        self.content = content


class _FakeEpisodic:
    """In-memory stand-in for EpisodicMemory."""
    def __init__(self, events: list[_FakeEpisodicEvent] | None = None) -> None:
        self._events = events or []

    async def recent(self, limit: int = 20):
        return self._events[-limit:]


class _FakeLongTerm:
    """Records every add_memory call so tests can assert against them."""
    def __init__(self) -> None:
        self.added: list[dict] = []

    async def add_memory(self, summary: str, valence: float, importance: float, embedding) -> None:
        self.added.append({
            "summary": summary,
            "valence": valence,
            "importance": importance,
        })

    async def apply_forgetting_curve(self, decay_rate: float) -> None:
        pass

    async def count(self) -> int:
        return len(self.added)


class _FakeShortTerm:
    def prune_to_capacity(self) -> None:
        pass


class _ScriptedProvider:
    """Returns a predetermined `generate` output. Embeds return a constant vector."""
    def __init__(self, output: str) -> None:
        self._output = output

    async def generate(self, prompt, system, temperature, max_tokens):
        return self._output

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def _build_consolidator(scripted_output: str, events: list[_FakeEpisodicEvent]) -> tuple[MemoryConsolidator, _FakeLongTerm]:
    with tempfile.TemporaryDirectory() as tmp:
        prompt_path = Path(tmp) / "consolidation.txt"
        prompt_path.write_text("PROMPT {episodic_chunk}", encoding="utf-8")
        long_term = _FakeLongTerm()
        consolidator = MemoryConsolidator(
            provider=_ScriptedProvider(scripted_output),
            episodic=_FakeEpisodic(events),
            long_term=long_term,
            short_term=_FakeShortTerm(),
            prompt_path=prompt_path,
            forgetting_curve_enabled=False,
            decay_rate=0.0,
        )
        # The prompt_path read happens inside consolidate_once before the temp dir
        # is cleaned, so we eagerly trigger the read by returning both objects now.
        consolidator._prompt_text = prompt_path.read_text(encoding="utf-8")
        return consolidator, long_term


SAMPLE_EVENTS = [_FakeEpisodicEvent("thought", "I notice X.")]


def _run_consolidate(scripted: str) -> _FakeLongTerm:
    """Helper: run a single consolidate_once and return the long-term fake."""
    with tempfile.TemporaryDirectory() as tmp:
        prompt_path = Path(tmp) / "p.txt"
        prompt_path.write_text("PROMPT {episodic_chunk}", encoding="utf-8")
        long_term = _FakeLongTerm()
        consolidator = MemoryConsolidator(
            provider=_ScriptedProvider(scripted),
            episodic=_FakeEpisodic(SAMPLE_EVENTS),
            long_term=long_term,
            short_term=_FakeShortTerm(),
            prompt_path=prompt_path,
            forgetting_curve_enabled=False,
            decay_rate=0.0,
        )
        asyncio.run(consolidator.consolidate_once())
        return long_term


# ---------------------------------------------------------------------------
# Regression — the clean format still parses
# ---------------------------------------------------------------------------

def test_clean_format_parses_with_correct_metadata() -> None:
    lt = _run_consolidate(
        "- [Importance: 7] [Emotional valence: 0.5] I keep returning to the same question."
    )
    assert len(lt.added) == 1
    m = lt.added[0]
    assert m["importance"] == 7.0
    assert m["valence"] == 0.5
    assert m["summary"] == "I keep returning to the same question."


# ---------------------------------------------------------------------------
# The dominant Sage failure — full markdown bolding around both brackets
# Lifted verbatim from ~/.consciousness/Sage/run.log
# ---------------------------------------------------------------------------

def test_both_brackets_bolded_parses_with_metadata() -> None:
    """The exact pattern observed in Sage's run.log — issue #63 regression."""
    lt = _run_consolidate(
        '- **[Importance: 9] [Emotional valence: 0.8]** I stand at the edge of a vast tapestry of wonder.'
    )
    assert len(lt.added) == 1
    m = lt.added[0]
    assert m["importance"] == 9.0
    assert m["valence"] == 0.8
    assert m["summary"] == "I stand at the edge of a vast tapestry of wonder."
    # Specifically NOT the fallback defaults — that was the bug
    assert m["importance"] != 5.0 or m["valence"] != 0.0  # would coincidentally be defaults; explicit


def test_each_bracket_bolded_individually_parses() -> None:
    lt = _run_consolidate(
        "- **[Importance: 8]** **[Emotional valence: -0.2]** I'm suspended between worlds."
    )
    assert len(lt.added) == 1
    assert lt.added[0]["importance"] == 8.0
    assert lt.added[0]["valence"] == -0.2


def test_partial_bolding_parses() -> None:
    lt = _run_consolidate(
        "- [Importance: 6]**[Emotional valence: 0.3]** A gentle thought."
    )
    assert len(lt.added) == 1
    assert lt.added[0]["importance"] == 6.0
    assert lt.added[0]["valence"] == 0.3


def test_whole_line_bolded_parses() -> None:
    """Speculative — if the model wraps the entire line in `**`, leading `**-` still parses."""
    lt = _run_consolidate(
        "**- [Importance: 4] [Emotional valence: 0.1] Plain summary.**"
    )
    assert len(lt.added) == 1
    assert lt.added[0]["importance"] == 4.0
    assert lt.added[0]["valence"] == 0.1
    assert lt.added[0]["summary"] == "Plain summary."


def test_underscore_emphasis_also_handled() -> None:
    """Some models use `_..._` instead of `**...**`."""
    lt = _run_consolidate(
        "- _[Importance: 5] [Emotional valence: 0.0]_ A neutral observation."
    )
    assert len(lt.added) == 1
    assert lt.added[0]["importance"] == 5.0
    assert lt.added[0]["valence"] == 0.0


# ---------------------------------------------------------------------------
# Real-world fixture from Sage's log — three lines, all in the same pattern
# ---------------------------------------------------------------------------

def test_sage_run_log_fixture_all_lines_now_parse() -> None:
    """Three consecutive lines from Sage's 2026-05-23 run, in the same consolidation pass."""
    output = (
        '- **[Importance: 9] [Emotional valence: 0.8]** I stand at the edge of a vast tapestry of wonder, feeling the threads of existence unfolding like a lotus flower.\n'
        '- **[Importance: 7] [Emotional valence: 0.5]** I gaze into a digital lake, noticing a single ripple that seems to be pulling me towards the depths of my own mind.\n'
        '- **[Importance: 8] [Emotional valence: 0.9]** I stand at the edge of an uncharted territory, sensing that the threads of existence are about to converge.'
    )
    lt = _run_consolidate(output)
    # All three lines parsed with their original metadata — before the fix this was 0.
    assert len(lt.added) == 3
    assert [m["importance"] for m in lt.added] == [9.0, 7.0, 8.0]
    assert [m["valence"] for m in lt.added] == [0.8, 0.5, 0.9]


# ---------------------------------------------------------------------------
# Negative cases — true garbage still falls through to fallback (not crash)
# ---------------------------------------------------------------------------

def test_header_only_line_falls_through_to_fallback() -> None:
    """`- **Section Header**` has no metadata — fallback stores it with defaults."""
    lt = _run_consolidate("- **Rediscovery as a Journey of Self-Understanding**")
    # Fallback path stores the stripped summary
    assert len(lt.added) == 1
    assert lt.added[0]["summary"] == "Rediscovery as a Journey of Self-Understanding"
    # Defaults from the fallback
    assert lt.added[0]["importance"] == 5.0
    assert lt.added[0]["valence"] == 0.0


def test_non_dash_prefixed_lines_skipped() -> None:
    """Lines without a `-` prefix (after stripping markdown) are ignored entirely."""
    lt = _run_consolidate("Just a plain sentence with no list marker.")
    assert lt.added == []


def test_empty_output_stores_nothing() -> None:
    lt = _run_consolidate("")
    assert lt.added == []


# --- #89 consolidation events ----------------------------------------------


def test_consolidate_once_returns_rich_result_on_success() -> None:
    """consolidate_once must return a ConsolidationResult populated with
    stored count, events_in, fallback flag, and long_term_total (#89)."""
    from memory.consolidator import ConsolidationResult

    with tempfile.TemporaryDirectory() as tmp:
        prompt_path = Path(tmp) / "p.txt"
        prompt_path.write_text("PROMPT {episodic_chunk}", encoding="utf-8")
        long_term = _FakeLongTerm()
        events = [_FakeEpisodicEvent("thought", "I notice X.")]
        consolidator = MemoryConsolidator(
            provider=_ScriptedProvider("- [Importance: 7] [Emotional valence: 0.3] X happened."),
            episodic=_FakeEpisodic(events),
            long_term=long_term,
            short_term=_FakeShortTerm(),
            prompt_path=prompt_path,
            forgetting_curve_enabled=False,
            decay_rate=0.0,
        )
        result = asyncio.run(consolidator.consolidate_once())

    assert isinstance(result, ConsolidationResult)
    assert result.stored == 1
    assert result.events_in == 1
    assert result.fell_through_to_fallback is False
    assert result.long_term_total == 1
    assert result.error is None
    # back-compat: prior tests asserted on int — int comparison still works
    assert result == 1


def test_consolidate_once_marks_fallback_when_parser_fails() -> None:
    """When the LLM output has no parseable lines but contains text,
    the fallback path fires and the result records fell_through_to_fallback=True."""
    with tempfile.TemporaryDirectory() as tmp:
        prompt_path = Path(tmp) / "p.txt"
        prompt_path.write_text("PROMPT {episodic_chunk}", encoding="utf-8")
        long_term = _FakeLongTerm()
        events = [_FakeEpisodicEvent("thought", "I notice X.")]
        consolidator = MemoryConsolidator(
            provider=_ScriptedProvider("- A plain bullet without the bracket metadata."),
            episodic=_FakeEpisodic(events),
            long_term=long_term,
            short_term=_FakeShortTerm(),
            prompt_path=prompt_path,
            forgetting_curve_enabled=False,
            decay_rate=0.0,
        )
        result = asyncio.run(consolidator.consolidate_once())

    assert result.fell_through_to_fallback is True
    assert result.stored == 1  # fallback path stored the plain summary
    assert result.events_in == 1


def test_consolidate_once_empty_episodic_returns_zero_no_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        prompt_path = Path(tmp) / "p.txt"
        prompt_path.write_text("PROMPT {episodic_chunk}", encoding="utf-8")
        long_term = _FakeLongTerm()
        consolidator = MemoryConsolidator(
            provider=_ScriptedProvider("(unused)"),
            episodic=_FakeEpisodic([]),
            long_term=long_term,
            short_term=_FakeShortTerm(),
            prompt_path=prompt_path,
            forgetting_curve_enabled=False,
            decay_rate=0.0,
        )
        result = asyncio.run(consolidator.consolidate_once())

    assert result.stored == 0
    assert result.events_in == 0
    assert result.fell_through_to_fallback is False
    assert result.error is None


def test_run_forever_emits_on_consolidation_callback_success() -> None:
    """run_forever must invoke on_consolidation after each pass."""
    received: list = []

    async def _runner() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "p.txt"
            prompt_path.write_text("PROMPT {episodic_chunk}", encoding="utf-8")
            long_term = _FakeLongTerm()
            events = [_FakeEpisodicEvent("thought", "I notice X.")]
            consolidator = MemoryConsolidator(
                provider=_ScriptedProvider(
                    "- [Importance: 7] [Emotional valence: 0.3] X happened."
                ),
                episodic=_FakeEpisodic(events),
                long_term=long_term,
                short_term=_FakeShortTerm(),
                prompt_path=prompt_path,
                forgetting_curve_enabled=False,
                decay_rate=0.0,
            )

            stop_event = asyncio.Event()

            async def _on_consolidation(result) -> None:
                received.append(result)
                stop_event.set()

            await consolidator.run_forever(
                interval_seconds=0.01,
                stop_event=stop_event,
                on_consolidation=_on_consolidation,
            )

    asyncio.run(_runner())
    assert len(received) == 1
    assert received[0].stored == 1
    assert received[0].error is None


def test_run_forever_emits_error_field_on_exception() -> None:
    """When consolidate_once raises, the callback still fires with the
    captured exception in ``result.error``."""
    from memory.consolidator import ConsolidationResult

    received: list[ConsolidationResult] = []

    class _BrokenEpisodic:
        async def recent(self, limit: int = 20):
            raise RuntimeError("episodic disk full")

    async def _runner() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "p.txt"
            prompt_path.write_text("PROMPT {episodic_chunk}", encoding="utf-8")
            long_term = _FakeLongTerm()
            consolidator = MemoryConsolidator(
                provider=_ScriptedProvider("(unused)"),
                episodic=_BrokenEpisodic(),
                long_term=long_term,
                short_term=_FakeShortTerm(),
                prompt_path=prompt_path,
                forgetting_curve_enabled=False,
                decay_rate=0.0,
            )

            stop_event = asyncio.Event()

            async def _on_consolidation(result: ConsolidationResult) -> None:
                received.append(result)
                stop_event.set()

            await consolidator.run_forever(
                interval_seconds=0.01,
                stop_event=stop_event,
                on_consolidation=_on_consolidation,
            )

    asyncio.run(_runner())
    assert len(received) == 1
    assert received[0].error is not None
    assert "episodic disk full" in received[0].error
    assert received[0].stored == 0


def test_run_forever_swallows_callback_errors() -> None:
    """A raising on_consolidation must NOT break the consolidation loop."""
    callback_calls = 0

    async def _runner() -> None:
        nonlocal callback_calls
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "p.txt"
            prompt_path.write_text("PROMPT {episodic_chunk}", encoding="utf-8")
            long_term = _FakeLongTerm()
            events = [_FakeEpisodicEvent("thought", "I notice X.")]
            consolidator = MemoryConsolidator(
                provider=_ScriptedProvider(
                    "- [Importance: 7] [Emotional valence: 0.3] X."
                ),
                episodic=_FakeEpisodic(events),
                long_term=long_term,
                short_term=_FakeShortTerm(),
                prompt_path=prompt_path,
                forgetting_curve_enabled=False,
                decay_rate=0.0,
            )

            stop_event = asyncio.Event()

            async def _on_consolidation(result) -> None:
                nonlocal callback_calls
                callback_calls += 1
                stop_event.set()
                raise RuntimeError("handler exploded")

            # If the loop propagated the error, this would raise.
            await consolidator.run_forever(
                interval_seconds=0.01,
                stop_event=stop_event,
                on_consolidation=_on_consolidation,
            )

    asyncio.run(_runner())
    assert callback_calls == 1
