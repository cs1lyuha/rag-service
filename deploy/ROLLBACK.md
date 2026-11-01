# Deploy și rollback

Fiecare imagine e etichetată cu SHA-ul commit-ului, deci un rollback înseamnă doar să pornești eticheta veche.

```bash
# deploy
export IMAGE_TAG=$(git rev-parse --short HEAD)
docker build -t ghcr.io/cs1lyuha/rag-service:$IMAGE_TAG .. && docker compose up -d
curl -fsS https://$DOMAIN/health            # verificare după deploy

# rollback la versiunea anterioară
export IMAGE_TAG=<sha-ul vechi>
docker compose up -d app                     # doar serviciul; Caddy și certificatele rămân
curl -fsS https://$DOMAIN/health
```

Reguli:
- deploy doar din `main`, după ce CI-ul (teste + gate de calitate RAG + build Docker) e verde;
- indexul se reconstruiește din `kb/` la pornire, deci nu există o bază de date de migrat la rollback;
- documentele adăugate prin `PUT /documents` sunt în memorie: după restart, sursa de adevăr rămâne `kb/` din imagine.
