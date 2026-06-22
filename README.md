# RAG App — FastAPI + LangGraph + Pinecone + Groq

A retrieval-augmented generation API. Ask a question at `/query`; the app embeds
it, retrieves the most relevant chunks from Pinecone, and has a Groq-hosted LLM
answer using only that context. Deployed free on **Render** with separate
**SIT** and **PROD** environments. Tests run in **GitHub Actions**; deploys are
handled automatically by **Render** (no GitHub secrets, no keys in the repo).

```
Request → FastAPI (/query) → LangGraph: retrieve (Pinecone) → generate (Groq) → answer
```

## Project layout
- `app/main.py` — FastAPI app, `/health` and `/query` endpoints.
- `app/chains/graph.py` — the LangGraph pipeline (retrieve → generate).
- `app/chains/embeddings.py` — local sentence-transformers embeddings (free, CPU).
- `app/config.py` — reads settings from environment variables.
- `scripts/ingest.py` — one-time script to create the index and load your docs.
- `render.yaml` — defines the two Render web services (SIT + PROD).
- `.github/workflows/ci.yml` — run tests, then trigger the Render deploy.
- `.env.example` — copy to `.env` for local runs (never commit `.env`).

---

## What you need (all have free tiers)
1. **Pinecone** account → free vector database. https://www.pinecone.io
2. **Groq** API key → for the LLM (free). https://console.groq.com
3. **GitHub** account → to host the repo and run CI/CD.
4. **Render** account → free hosting. https://render.com

> Cost note: Pinecone, Render, and Groq all have free tiers, and the embedding
> model runs locally for free — so this whole stack can run at $0.

---

## Step 1 — Run it locally first (recommended)
```bash
# from the project folder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env with your real keys
```
Edit `.env`:
```
APP_ENV=sit
PINECONE_API_KEY=your-real-pinecone-key
PINECONE_INDEX_NAME=rag-sit
GROQ_API_KEY=your-real-groq-key
```
Create the index and load sample docs, then start the API:
```bash
python -m scripts.ingest          # creates the index + upserts docs
uvicorn app.main:app --reload
```
Test it:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangGraph?"}'
```

## Step 2 — Push to GitHub
Create two branches: `develop` (→ SIT) and `main` (→ PROD).
```bash
git init
git add .
git commit -m "RAG app"
git branch -M main
git checkout -b develop
git remote add origin https://github.com/<you>/my-rag-app.git
git push -u origin develop
git push origin main
```

## Step 3 — Create Pinecone indexes
In the Pinecone console, create two serverless indexes (or let `ingest.py`
create them), **dimension 384, metric cosine**:
- `rag-sit`  (for the SIT environment)
- `rag-prod` (for PROD)

Run `python -m scripts.ingest` once per environment, with the matching
`PINECONE_INDEX_NAME` in your `.env`, to load your documents.

## Step 4 — Deploy on Render with the Blueprint
1. Go to https://dashboard.render.com → **New** → **Blueprint**.
2. Connect your GitHub repo. Render reads `render.yaml` and proposes two
   services: `rag-app-sit` (from `develop`) and `rag-app-prod` (from `main`).
3. Click **Apply**. Each service builds with `pip install -r requirements.txt`
   and starts with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. For **each** service, open it → **Environment** → add secret env vars
   (these are NOT in `render.yaml` on purpose — keys must stay secret):
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME` → `rag-sit` for SIT, `rag-prod` for PROD
   - `GROQ_API_KEY`
5. Save. Render redeploys. Your URLs look like
   `https://rag-app-sit.onrender.com` and `https://rag-app-prod.onrender.com`.

Test: `curl https://rag-app-sit.onrender.com/health`

## Step 5 — Auto-deploy (no GitHub secrets needed)
Render deploys on its own — no deploy hooks, no GitHub secrets, no keys in code.
1. When you connected the repo via Blueprint, **Auto-Deploy** is on by default.
   Confirm in each service → **Settings** → **Build & Deploy** → Auto-Deploy = Yes.
2. That's it: pushing to `develop` redeploys SIT, pushing to `main` redeploys PROD.
3. Your API keys live only in each service's **Environment** tab (Step 4) —
   Render's own encrypted secret store, never in the repo or in GitHub.

The GitHub Actions workflow (`.github/workflows/ci.yml`) now only runs the
tests (which are mocked, so they need no keys). Deployment is fully Render's job.

Workflow: build a feature → push to `develop` → verify on SIT →
merge `develop` into `main` → PROD deploys automatically.

> Why not put keys in the code? Anything committed to git is permanent and
> visible to anyone with repo access; leaked API keys get abused within minutes.
> Keeping them in Render's Environment tab is both safer *and* less work.

---

## Free-tier gotchas
- **Render free services sleep after ~15 min idle** and take ~30–60s to wake on
  the next request. You also get 750 free instance-hours/month per workspace.
- **Cold start downloads the embedding model** (~90 MB) the first time, so the
  first request after a deploy is slower.
- **Free Pinecone** allows a limited number of indexes; two small ones is fine.

## Fixes already applied to this project
- `config.py` exported `PINECONE_INDEX`, but the code imported
  `PINECONE_INDEX_NAME` → renamed so imports resolve (app now starts).
- Pinecone/LLM clients were created at import time → made lazy, so the app
  and tests load without keys and only need them to actually answer.
- `requirements.txt` used the deprecated `pinecone-client` → switched to
  `pinecone`, and pinned versions.
- `ci.yml` had the `deploy` job dedented to the top level (invalid workflow) →
  re-indented under `jobs:`.
- Query test hit live services → now mocks the graph so CI is free and reliable.

## Troubleshooting
- **App won't start / ImportError** → confirm env var names match exactly.
- **`PineconeApiException` / index not found** → run `scripts.ingest` for that
  environment, and check `PINECONE_INDEX_NAME` + dimension 384.
- **401 / auth error from Groq** → check `GROQ_API_KEY` in Render env vars.
- **Model decommissioned error** → set `GROQ_MODEL` to a current model from
  https://console.groq.com/docs/models
- **CI deploy step skipped** → it only runs on `push`, not pull requests.
