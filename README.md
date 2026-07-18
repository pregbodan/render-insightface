# BEAS InsightFace Render Service

Deploy this folder as a Render Python web service.

Endpoints:

- `GET /health` warms and reports the model/cache status.
- `POST /embed` accepts `{ "probeImage": "<base64>" }` and returns `{ "embedding": [...] }`.
- `POST /sync` accepts enrolled embeddings for optional in-memory caching.
- `POST /match` accepts a probe image or embedding and compares it against embeddings previously loaded through `/sync`.

Environment variables:

- `BEAS_RENDER_API_KEY`: shared secret used by cPanel.
- `INSIGHTFACE_MODEL`: defaults to `buffalo_l`.
- `INSIGHTFACE_MODEL_ROOT`: model cache directory. Use a Render disk path if you want model files to persist across deploys.
- `EMBEDDING_CACHE_MAX_ITEMS`: image embedding cache size.

After deploy, set the cPanel side to:

```php
define('INSIGHTFACE_RENDER_URL', 'https://your-render-service.onrender.com');
define('INSIGHTFACE_RENDER_API_KEY', 'same-secret-as-render');
```

The existing BEAS PHP app only needs `/embed` to remove Python/InsightFace work from shared hosting. `/sync` and `/match` are included for the next step where Render performs the full face lookup from memory.
