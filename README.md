# p5‑ai‑debate 🤖⚛️
Prototype – AI debate for particle‑physics strategy (P5 2023)

Large‑language‑model “Pro vs Against” debates on the 2023 **Particle‑Physics Project Prioritization Panel (P5)** report (https://www.usparticlephysics.org/2023-p5-report/), with an automated LLM judge and detailed token‑usage stats.

---

## 📦 Quick start

```bash
# 1) Clone the repo and enter it
git clone https://github.com/nayara-focs/ai-debate-p5.git
cd ai-debate-p5

# 2) Create / activate a virtual‑ or conda‑env, then install deps
pip install -r requirements.txt

# 3) Add your OpenAI key (bash / zsh example)
export OPENAI_API_KEY="sk‑..."

# 4) Run a prototype debate
python -m scripts.run_debate
