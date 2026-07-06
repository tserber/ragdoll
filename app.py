import io
import uuid
import logging
import uvicorn
import docx2txt
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama

# Configure logging to print clearly to your terminal console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RAG_Engine")

app = FastAPI(title="Local Intel Mac RAG Engine")

# Connect directly to your local containers
logger.info("Connecting to Qdrant at http://localhost:6333...")
try:
    qdrant_client = QdrantClient(url="http://localhost:6333")
    COLLECTION_NAME = "mac_knowledge_base"

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' not found. Creating it...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        logger.info(f"Successfully created collection '{COLLECTION_NAME}'.")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' verified and ready.")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client connection: {str(e)}")


class QueryRequest(BaseModel):
    question: str


def chunk_text_fast(text: str, chunk_size_chars: int = 1200, overlap_chars: int = 150):
    """Fast character-based chunking to protect Intel CPU context buffers."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size_chars
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size_chars - overlap_chars
    return chunks


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...), file_id: str = Form(...)):
    """Pipeline A: Extract, chunk, embed, and upsert raw documents."""
    filename = file.filename
    logger.info(f"--- Starting Ingestion Pipeline for file: '{filename}' ---")

    content = await file.read()
    extracted_text = ""

    # --- STEP 1: TEXT EXTRACTION ---
    logger.info(f"[Step 1/4] Extracting text layout from file content buffer...")
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            pages_text = []
            for idx, page in enumerate(reader.pages):
                page_content = page.extract_text()
                if page_content:
                    pages_text.append(page_content)
            extracted_text = " ".join(pages_text)
            logger.info(f"PDF parsed completely. Total pages extracted: {len(reader.pages)}")
        elif filename.endswith(".docx"):
            extracted_text = docx2txt.process(io.BytesIO(content))
            logger.info("Word document (.docx) parsed completely.")
        else:
            logger.warning(f"Unsupported file format rejected: {filename}")
            raise HTTPException(status_code=400, detail="Unsupported file layout. Provide PDF or DOCX.")
    except Exception as e:
        logger.error(f"CRITICAL CRASH in Step 1 (Extraction): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    if not extracted_text.strip():
        logger.warning("Extraction completed but returned a completely empty text body.")
        raise HTTPException(status_code=400, detail="The provided document contains no parseable text.")

    # --- STEP 2: CHUNKING ---
    logger.info(f"[Step 2/4] Slicing text content into processing fragments...")
    text_chunks = chunk_text_fast(extracted_text)
    logger.info(f"Chunking complete. Created {len(text_chunks)} unique text strings.")

    # --- STEP 3: EMBEDDING GENERATION ---
    logger.info(f"[Step 3/4] Sending chunks sequentially to local Ollama API (nomic-embed-text)...")
    points = []

    for idx, chunk in enumerate(text_chunks):
        logger.info(f"  -> Generating vector embedding for chunk {idx + 1}/{len(text_chunks)} ({len(chunk)} chars)...")
        try:
            embed_res = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
            vector = embed_res["embedding"]

            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text_chunk": chunk,
                        "filename": filename,
                        "file_id": file_id,
                        "chunk_index": idx
                    }
                )
            )
        except Exception as e:
            logger.error(f"CRITICAL CRASH in Step 3 (Ollama Embedding) at chunk index {idx}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ollama generation failed at chunk {idx}: {str(e)}")

    # --- STEP 4: VECTOR DB UPSERT ---
    logger.info(f"[Step 4/4] Committing transaction: Loading {len(points)} vectors into Qdrant index...")
    try:
        upsert_result = qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Qdrant transaction response status complete: {upsert_result.status}")
        logger.info(f"--- Ingestion Pipeline Finished Successfully for '{filename}' ---")
        return {"status": "success", "chunks_indexed": len(points), "filename": filename}
    except Exception as e:
        logger.error(f"CRITICAL CRASH in Step 4 (Qdrant Insertion): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Qdrant transaction failed: {str(e)}")


@app.post("/query")
async def query_rag(request: QueryRequest):
    """Pipeline B: Vector search and conditional contextual text generation."""
    logger.info(f"--- Incoming User Query Received: '{request.question}' ---")
    try:
        logger.info("Embedding user query phrase...")
        embed_res = ollama.embeddings(model="nomic-embed-text", prompt=request.question)
        query_vector = embed_res["embedding"]

        logger.info("Querying Qdrant index fields...")
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=2,
            with_payload=True
        )
    except Exception as e:
        logger.error(f"Query pipeline extraction layer failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search engine processing error: {str(e)}")

    contexts = [point.payload["text_chunk"] for point in search_results.points]
    logger.info(f"Found {len(contexts)} matched reference contexts inside database.")

    if not contexts:
        logger.info("Zero context segments matched. Terminating generator.")
        return {"answer": "I couldn't find any relevant data inside my indexed knowledge base to answer that."}

    merged_context = "\n---\n".join(contexts)
    system_prompt = (
        "You are an accurate corporate assistant. Answer the user's question using ONLY the provided document context below. "
        "If the context does not contain the answer, state clearly that you do not know. Do not make up information.\n\n"
        f"CONTEXT:\n{merged_context}"
    )

    logger.info("Routing prompt template sequence to local qwen3:1.7b generation model...")
    try:
        response = ollama.generate(
            model="qwen3:1.7b",
            prompt=f"{system_prompt}\n\nQUESTION: {request.question}\n\nANSWER:",
            options={"num_ctx": 2048, "temperature": 0.0}
        )
        logger.info("Answer generation completed successfully.")
        return {"answer": response["response"]}
    except Exception as e:
        logger.error(f"Generation layer execution crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM generation crashed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)