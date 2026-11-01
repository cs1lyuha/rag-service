# 📚 rag-service

Un serviciu RAG gata de deploy, construit pe concluziile din [rag-eval-lab](https://github.com/cs1lyuha/rag-eval-lab)
(chunking pe secțiuni, căutare hibridă, k=3):

- **ingestie incrementală:** `PUT /documents/{nume}` cu Markdown; un document neschimbat nu se reindexează (hash de conținut);
- **căutare hibridă:** BM25 + n-grame de caractere, fuzionate cu Reciprocal Rank Fusion;
- **răspunsuri cu citări verificate în cod:** o citare către o sursă netrimisă modelului e ștearsă;
- **refuz explicit**, când documentația nu conține răspunsul;
- **streaming SSE:** text pe măsură ce vine, apoi un eveniment final cu citările;
- **chei API** (utilizator / admin), validare de input;
- **gate de calitate în CI:** 65 de întrebări cu praguri; o regresie pică build-ul;
- **Docker + Caddy:** HTTPS automat, buffering oprit pentru SSE, ghid de rollback pe etichete de imagine.

> Pasul 6 din planul meu de studiu: „deploy real”, „CI: dacă scorul de evaluare scade, build-ul pică”, „rollback testat”.

![CI](https://github.com/cs1lyuha/rag-service/actions/workflows/ci.yml/badge.svg)

## API

```bash
export RAG_API_KEYS=k1 RAG_ADMIN_KEY=adm RAG_KB_DIR=kb
uvicorn ragsvc.api:create_app --factory --port 8000

curl -X POST localhost:8000/ask -H "x-api-key: k1" -H "content-type: application/json" \
     -d '{"question": "Cât costă livrarea standard?"}'
```
```json
{"answer": "Livrarea standard costă 50 de lei [1].", "abstained": false,
 "citations": [{"id": "livrare#costul-livrarii", "document": "Livrare", "section": "Costul livrării", "snippet": "..."}],
 "latency_ms": 2.1}
```

```bash
curl -N -X POST localhost:8000/ask/stream -H "x-api-key: k1" -H "content-type: application/json" \
     -d '{"question": "În câte zile pot returna?"}'
# event: delta  data: {"text": "Clienții "}  …  event: done  data: {"answer": "...", "citations": [...]}

curl -X PUT localhost:8000/documents/program -H "x-api-key: adm" --data-binary @program.md   # reindexare imediată
```

| endpoint | cheie | |
|---|---|---|
| `GET /health` | — | documente și secțiuni indexate |
| `GET /documents` | utilizator | nume, număr de secțiuni, hash |
| `PUT /documents/{nume}` | admin | `created` / `updated` / `unchanged` |
| `DELETE /documents/{nume}` | admin | |
| `POST /ask` | utilizator | `{question, k≤8}` |
| `POST /ask/stream` | utilizator | SSE: `delta`, `done`, `error` |

## Calitate (gate în CI)

`python -m ragsvc.evaluate`, pe cele 65 de întrebări din rag-eval-lab, cu generatorul extractiv (offline):

| metrică | valoare | prag în CI |
|---|---:|---:|
| retrieval hit@3 | 0.933 | ≥ 0.90 |
| citare corectă (citează fragmentul cu dovada) | 0.717 | ≥ 0.70 |
| answer recall | 0.614 | ≥ 0.58 |
| refuz greșit (la întrebări cu răspuns) | 0.167 | ≤ 0.20 |
| refuz corect (la întrebări fără răspuns) | 0.400 | ≥ 0.40 |

Pragurile sunt **podele puțin sub valorile actuale**: build-ul pică la o regresie reală, nu la zgomot (`test_gate_catches_a_regression`).

**Limitări, spuse direct:**
- Pragul de refuz al generatorului extractiv (0,3) e ales pe **același** set de întrebări. Nu există un set separat, deci cifrele sunt optimiste.
- Refuzul lexical rămâne slab (2 din 5). Este aceeași concluzie ca în rag-eval-lab: decizia „nu știu” cere un model. `ClaudeAnswerer` (streaming, marcaje [n], `NU_STIU`) e pregătit pentru asta.
- Generatorul extractiv alege propoziția cu cea mai bună acoperire a întrebării. La „În câte zile pot returna?” poate alege întâi regula pentru firme. Un LLM ar formula răspunsul pentru persoane fizice.

## Deploy

```bash
cd deploy
DOMAIN=rag.exemplu.md RAG_ADMIN_KEY=... RAG_API_KEYS=k1,k2 IMAGE_TAG=$(git rev-parse --short HEAD) docker compose up -d --build
```

- **Caddy** obține certificatul automat. `flush_interval -1` face ca SSE să nu fie ținut în buffer de proxy.
- Containerul rulează ca utilizator neprivilegiat, cu sistem de fișiere read-only și health check.
- **Rollback:** imaginile sunt etichetate cu SHA-ul commit-ului; vezi [`deploy/ROLLBACK.md`](deploy/ROLLBACK.md).
- CI-ul are trei porți: **teste → gate de calitate RAG → build Docker + test de fum** (`/health` și o întrebare reală în container).

## Structura

| Fișier | Rol |
|---|---|
| `ragsvc/index.py` | secțiuni, BM25, n-grame de caractere, RRF, upsert / delete incremental, thread-safe |
| `ragsvc/answer.py` | `ExtractiveAnswerer`, `ClaudeAnswerer` (streaming), `clean_citations` |
| `ragsvc/api.py` | FastAPI, chei, SSE |
| `ragsvc/evaluate.py` | metrici + praguri (exit 1 la regresie) |
| `deploy/` | docker-compose cu Caddy, Caddyfile, ghid de rollback |

**28 de teste:** `pytest -q`.

## Ce urmează

- [ ] PostgreSQL + pgvector pentru index persistent și embeddings neurale
- [ ] Gate-ul de calitate rulat și pe `ClaudeAnswerer` (cu casete, ca în [llm-eval-gate](https://github.com/cs1lyuha/llm-eval-gate))
- [ ] Trace-uri per întrebare (Langfuse / OpenTelemetry)
- [ ] Deploy real pe un VPS cu domeniu
