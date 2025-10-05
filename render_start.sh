#!/usr/bin/env bash
set -e

# Hent eller generér embeddings, hvis de ikke findes på Render-drevet endnu
if [ ! -f kb_embeddings.jsonl ]; then
  if [ -n "$EMB_URL" ]; then
    echo "Downloading embeddings from $EMB_URL …"
    curl -L "$EMB_URL" -o kb_embeddings.jsonl
  elif [ -f "./embed_my_folder.py" ]; then
    echo "Generating embeddings locally …"
    python embed_my_folder.py
  else
    echo "WARNING: No embeddings present (set EMB_URL or add embed_my_folder.py)"
  fi
fi

# Start FastAPI (skift 'search_api' til dit filnavn uden .py, hvis nødvendigt)
exec uvicorn search_api:app --host 0.0.0.0 --port "$PORT"