# ragdoll
rag with perks

```console
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  --restart unless-stopped \
  qdrant/qdrant:latest
➜  ~ docker exec -it ollama ollama pull nomic-embed-text
➜  ~ docker run -d \
  --name ollama \
  -v ollama_storage:/root/.ollama \
  -p 11434:11434 \
  --restart always \
  ollama/ollama:latest

➜  ~ docker exec -it ollama ollama pull qwen3:1.7b
➜  ~ docker exec -it ollama ollama run qwen3:1.7b
```