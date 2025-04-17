"""
ai_debate_p5 -  package root.

Re‑export high - level helpers so notebooks can do

>>> from ai_debate_p5 import run_all_matches
"""
from .debate_engine import run_all_matches          # noqa: F401

__all__ = ["run_all_matches"]
