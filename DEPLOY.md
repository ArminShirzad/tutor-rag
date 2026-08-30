# Deploying

The app ships two profiles. Pick by how much memory the host gives you.

| Host | Profile | Card required | Notes |
|---|---|---|---|
| **Vercel** | API | **no** | Free hobby tier. Serverless, so `/tmp` is the only writable path. |
| Render free | API | yes | Free tier, but the account needs a card on file. |
| Fly.io / Cloud Run | offline | yes | Local models, no per-query rerank cost. |
| Docker anywhere | offline | — | `docker build -t tutor-rag . && docker run -p 7860:7860 tutor-rag` |
| Hugging Face Spaces | offline | yes | Docker Spaces require a PRO subscription; free tier is static-only. |

## Vercel (free, no card)

```bash
npx vercel            # first run links the project and prompts for login
npx vercel --prod
```

Then set the one secret, either in the dashboard under **Settings →
Environment Variables**, or:

```bash
npx vercel env add GEMINI_API_KEY production
```

`api/index.py` is the entrypoint and `vercel.json` routes all traffic to it.
Two serverless constraints are handled there:

- **Read-only filesystem.** Only `/tmp` is writable, so `INDEX_DIR` points
  there. The index is rebuilt on a cold start — about 2 seconds and one batched
  embedding call for this corpus — then reused while the instance stays warm.
- **250 MB bundle limit.** torch does not fit, so the deployment uses hosted
  embeddings and the listwise LLM reranker. Same pipeline, different providers.

## Render (needs a card on the account)

The repo contains `render.yaml`, so Render configures everything itself.

1. Sign in at [render.com](https://render.com) with GitHub.
2. **New → Blueprint**, pick this repository, **Apply**.
3. Render reads `render.yaml` and creates the service. It will prompt for the
   one secret marked `sync: false`:
   - `GEMINI_API_KEY` → your key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
4. First build takes ~3 minutes. The service is then at
   `https://<name>.onrender.com`.

On boot the app builds its index from `data/corpus/` (about 3 seconds for the
sample corpus), so there is no build step and no pre-baked artefact to keep in
sync with the code.

### What to expect on the free tier

- **Cold start ~50 s.** Render suspends free services after 15 minutes idle.
  Hit the URL once before demoing it to anyone.
- **~2 questions per minute.** The Gemini free tier allows 5 requests/minute and
  the slim profile spends two per question (rerank + answer). The rate limiter
  spaces requests rather than failing them, so a burst gets slow, not broken.
  Raise `LLM_RPM` on a paid tier.
- **`RERANK_PROVIDER=lexical`** halves the LLM calls if you would rather have
  throughput than rerank quality.

## Docker (full profile)

```bash
docker build -t tutor-rag .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key tutor-rag
```

The image bakes the embedding model, the cross-encoder, and the built index at
build time, so the container starts ready to serve rather than downloading
~180 MB of weights on every cold start. Build takes ~8 minutes; the image is
roughly 2 GB, almost all of it torch.

Without `GEMINI_API_KEY` the container still runs, in extractive mode.

## Configuration

Everything is environment variables — see [.env.example](.env.example). The
ones that matter for a deployment:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Generation. Absent → extractive mode. |
| `EMBED_PROVIDER` | `local` | `local` \| `gemini` |
| `RERANK_PROVIDER` | `auto` | `auto` \| `cross-encoder` \| `llm` \| `lexical` |
| `LLM_RPM` | `5` | Request spacing. `0` disables. |
| `CANDIDATE_K` | `20` | Retrieved before reranking. Lower it when reranking costs an API call. |
| `FINAL_K` | `5` | Kept after reranking. |
| `MIN_SCORE` | `-5.0` | Refusal threshold. Re-calibrate if you change the reranker. |

**Changing `EMBED_PROVIDER` requires re-indexing.** The app detects this on
startup and rebuilds automatically — querying a 384-d index with a 768-d model
is a silent correctness bug, not an error.
