# Market Research Assistant

A Streamlit app that compares "my product" against one or more competitors
across a fixed set of categories, with every competitor claim grounded in a
verbatim quote from a real fetched page.

> **Disclosure:** "my product" in this demo is Narsi, a fictional AI coding
> assistant described in `product_narsi.md`, used only to demonstrate the
> pipeline. It is not a real product. The app shows this banner too.

## Features

- Interactive comparison table across a fixed 10-category list
- Every competitor claim is grounded in a verbatim quote from a page the
  pipeline actually fetched; unsupported claims are dropped, not shown
- Output-contract guardrails: the comparison stage retries on a malformed or
  incomplete reply and falls back to a clearly marked partial result instead
  of crashing
- Per-stage cost/latency/token trace, CSV and Markdown-with-sources downloads
- Pluggable LLM provider (OpenAI or DigitalOcean serverless inference) and
  pluggable "my product" facts source (a local markdown file by default, or
  an experimental, not-actually-knowledge-base-backed alternative; see
  "FACTS_SOURCE=do_knowledge_base is experimental" below)
- Offline eval set (`evals/`) that scores category coverage, grounding rate,
  fact recall and no-evidence precision against hand-authored fixtures

## Installation

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/bnarasimha/market-research-assistant.git
   cd market-research-assistant
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies (add `-r requirements-dev.txt` instead if you also
   want to run tests/lint):

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `LLM_PROVIDER` plus that provider's
   variables. The simplest path is `LLM_PROVIDER=openai` with `OPENAI_API_KEY`
   set; see `market_research/llm.py` for the `do_serverless` alternative.

4. Run the app:

   ```bash
   streamlit run app.py
   ```

   Open `http://localhost:8501`.

### FACTS_SOURCE=do_knowledge_base is experimental, not a real knowledge base

By default (`FACTS_SOURCE=local_markdown`) the product-facts stage reads
`product_narsi.md` directly with BM25 ranking, offline, no model call.

`FACTS_SOURCE=do_knowledge_base` does **not** call a real hosted DigitalOcean
knowledge base or GenAI agent, despite the name. It sends a plain prompt to
whatever generic LLM is configured, with no document or retrieval index
actually attached, and takes whatever citation the model fabricates in
response at face value. Setting up a DigitalOcean knowledge base/GenAI agent
would not change this: nothing in the code calls one. See
`market_research/facts.py`'s `DOKnowledgeBaseSource` docstring. Wiring this
to an actual DO knowledge-base endpoint is unfinished work; until that is
done, treat this path's output as unverified, not as a grounded fact.

### Optional: real competitor web search

Without `SERPER_API_KEY` set, the pipeline falls back to an empty search
backend and fetches zero competitor pages (every competitor cell comes back
`no_evidence`). Set `SERPER_API_KEY` to enable real search.

## Evals

`python -m evals.run --llm fake` runs the offline eval set end-to-end with no
network access and no credentials, scoring six fixture trios on four metrics.
Results are committed under `evals/results/`:

- [`2026-11-12-fake.md`](evals/results/2026-11-12-fake.md): normal run,
  checks the scoring/plumbing are correct (not a real model's quality).
- [`2026-11-12-fake-corrupted-demo.md`](evals/results/2026-11-12-fake-corrupted-demo.md):
  the same run with one fixture's quote deliberately mangled, demonstrating
  that the grounding-rate metric actually detects a broken fixture.

`--llm do_serverless` runs the same trios against a real provider for real
quality numbers; this code path exists and is correct but was never run with
real credentials in this repository's history, so there is no committed
real-model report yet.

## Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -m "not slow"
```

Tests are split into fast unit tests and `integration`-marked tests that
drive the real CrewAI Flow/Crew/Task machinery with a scripted fake LLM (no
network). CI (`.github/workflows/ci.yml`) runs both plus the fake-mode eval
set on Python 3.11 and 3.13.

## Project Structure

- `app.py`, `crew_flow.py`: thin UI/CLI callers of
  `market_research.pipeline.run_comparison`.
- `market_research/`: the typed pipeline (models, LLM provider layer,
  product-facts sources, competitor page fetching + grounding, comparison
  guardrails, disk cache, per-stage trace, pricing estimates).
- `evals/`: the offline eval set (fixtures, golden data, runner, results).
- `tests/`: unit tests plus `integration`-marked end-to-end tests.
- `product_narsi.md`: the fictional product used for this demo.
- `pricing.yaml`: per-million-token price estimates used only for the UI's
  cost display.
