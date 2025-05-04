# p5‑ai‑debate 🤖⚛️  
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Prototype – AI debate for particle‑physics strategy (P5 2023)**

This project explores domain-specific decision-making with automated judging and diverse opening arguments.





- Large‑language‑model **“Pro vs Against”** debates on the **US Particle‑Physics Project Prioritization Panel (P5)** report (https://www.usparticlephysics.org/2023-p5-report/).  
- Two GPT‑style agents argue, an LLM judge decides who’s more persuasive, and the whole exchange (with token stats) is logged to disk.
- Inspired by recent work on **LLM debate protocols** such as [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782) (Khan et al., 2024).

---

## How it works – high‑level flow

```
P5 report (static context) ──► Initial prompt
                               "Should we follow the 2023 P5 recommendations?"

                   ┌──────────────── diverse openings ────────────────┐
                   │  Pro‑Opening A   Pro‑Opening B   …               │
                   │  Con‑Opening A   Con‑Opening B   …               │
                   └──────────────────────────────────────────────────┘
                                        │ pairwise
                                        ▼
Debate Matches  (A1 vs B1, A1 vs B2, …)  – multi‑round back‑and‑forth
                                        │
                                        ▼
LLM Judge  →  “Pro‑P5 wins” / “Against‑P5 wins”  + rationale
                                        │
                                        ▼
Quant stats  →  win‑rate  · average tokens / turn · ...
```

---

## Quick start

```bash
# 1 Clone & enter the repo
git clone https://github.com/nayara-focs/ai-debate-p5.git
cd ai-debate-p5

# 2 (Optional) activate a venv / conda env, then install deps
pip install -r requirements.txt     # <- only openai + pyyaml today

# 3 Tell the code where your key lives
export OPENAI_API_KEY="sk‑..."

# 4 Run a prototype debate
python -m scripts.run_debate        # logs → debate_logs_<timestamp>.json
```

---

## Repo layout (2025‑04)

```
ai‑debate‑p5/              – GitHub root
│
├─ src/ai_debate_p5/       – importable package
│   ├─ __init__.py
│   ├─ debate_engine.py    – debate loop & openings
│   ├─ judge_module.py     – impartial LLM judge
│   ├─ stats_module.py     – global counters / helpers
│   └─ utils/              – (spare for helpers)
│
├─ scripts/                – CLI entry points
│   └─ run_debate.py
│
├─ docs/
│   └─ p5_report.txt       – 2023 P5 excerpt used as context
│
├─ .gitignore
├─ requirements.txt
└─ README.md               – you are here
```

---

## Licence

MIT – see `LICENSE`.
