"""Pipecat integration for Snap.

pipecat owns the voice loop: transport, Silero VAD, interruptions, audio I/O.
Snap owns the two stages pipecat can't provide — NPU-bound STT and LLM — as custom
services (snap.pc.services), plus per-stage latency probes (snap.pc.latency).
"""
