# Quickstart: Run Ingestion Pipeline

## 1. Configure Qdrant

Set Qdrant connection in your environment (or via config if you wire it):

```bash
set QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
set QDRANT_API_KEY=<your-qdrant-api-key>
set QDRANT_COLLECTION=rag_llm
```

(PowerShell: `$env:QDRANT_URL = "..."`, etc.)

## 2. Prepare source documents

From `rag-api`:

```bash
cd C:\Aeronation\rag\rag-api

mkdir data\aerospace_docs
```

Put your files (PDF, DOCX, TXT, MD, etc.) into:

```text
rag-api/data/aerospace_docs/
```

## 3. Run the ingestion pipeline

From `rag-api`:

```bash
cd C:\Aeronation\rag\rag-api

python -m ingestion.orchestration.system_pipeline ^
  --data-dir ./data/aerospace_docs ^
  --collection rag_llm ^
  --persist-dir ./persist ^
  --max-attempts 3
```

Optional tuning:

```bash
python -m ingestion.orchestration.system_pipeline ^
  --data-dir ./data/aerospace_docs ^
  --collection rag_llm ^
  --persist-dir ./persist ^
  --chunk-size 512 ^
  --chunk-overlap 64 ^
  --max-attempts 3
```

## 4. Verify local persist

After success, you should see:

```bash
dir persist\rag_llm
```

with files like:

```text
docstore.json
index_store.json
graph_store.json
image__vector_store.json
manifest.json
ingestion_state.json
```

Vectors are stored in your Qdrant collection `rag_llm`, metadata in `persist/rag_llm`.

## 5. (Optional) Upload to S3

If you want S3 backup, run:

```bash
python upload_to_s3.py --collection rag_llm
```

(Requires `S3_PERSIST_BUCKET`, `AWS_REGION`, and AWS credentials configured.)
