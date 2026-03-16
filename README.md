# PLagX – Cross-Language Plagiarism Intelligence

A modern, professional web-based cross-language plagiarism detection system for academic authorities. Detects translation-based plagiarism between **Dutch** and **English** research papers using sentence embeddings and cosine similarity.

## Folder structure (clean architecture)

```
PLagX/
├── app/
│   ├── main.py           # FastAPI app, CORS, routes, serve index.html
│   ├── config.py         # Settings (SIMILARITY_THRESHOLD = 0.80, etc.)
│   ├── schemas.py        # Pydantic request/response models
│   ├── store.py          # In-memory Dutch paper embeddings (precomputed)
│   ├── papers.py         # Dutch papers API (upload-dutch, list)
│   ├── plagiarism.py     # Plagiarism analysis API (analyze)
│   ├── text_utils.py     # Sentence splitting, normalization
│   └── services/
│       ├── pdf_service.py         # PDF text extraction
│       ├── embedding_service.py   # Model loaded ONCE globally
│       ├── similarity_service.py # Cosine similarity (sklearn), threshold from config
│       └── plagiarism_service.py# Orchestration: extract → embed → compare
├── templates/
│   └── index.html        # Dashboard (Tailwind, fetch(), loading spinner)
├── static/               # Optional static assets
├── requirements.txt
└── README.md
```

## Run locally

### 1. Create virtual environment

```bash
cd PLagX
python -m venv venv
```

- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Windows (CMD):** `venv\Scripts\activate.bat`
- **macOS/Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will download the sentence-transformers model; this may take a few minutes.

### 3. Run the server

From the **project root** (the folder containing `app/`):

```bash
uvicorn app.main:app --reload
```

Or, if you run from inside `app/`:

```bash
cd app
uvicorn main:app --reload
```

Default: **http://127.0.0.1:8000**

### 4. Verify

- **Dashboard (index.html):** http://127.0.0.1:8000  
- **API docs:** http://127.0.0.1:8000/docs  
- **ReDoc:** http://127.0.0.1:8000/redoc  

### 5. Add Dutch reference papers

Before analyzing English text, add at least one Dutch PDF to the reference corpus:

- Use **http://127.0.0.1:8000/docs** → `POST /api/papers/upload-dutch` (upload a Dutch PDF), or  
- `curl -X POST "http://127.0.0.1:8000/api/papers/upload-dutch" -F "file=@dutch.pdf" -F "title=My Paper"`

Embeddings are **precomputed at upload** and stored in memory; they are **not** recomputed on every request.

### 6. Optional: Preload Dutch papers at startup

Set environment variable `PLAGX_DUTCH_PAPERS_DIR` to a directory path containing Dutch PDFs. On startup, the app will load those PDFs and precompute embeddings into the store.

```bash
set PLAGX_DUTCH_PAPERS_DIR=C:\path\to\dutch_pdfs
uvicorn app.main:app --reload
```

## Configuration (config.py)

- **similarity_threshold** in `config.py` (default `0.80`): Matches above this cosine similarity are considered plagiarism.  
- **PLAGX_SIMILARITY_THRESHOLD**: Environment override.  
- **PLAGX_EMBEDDING_MODEL**: Default `paraphrase-multilingual-MiniLM-L12-v2`.  
- **PLAGX_DUTCH_PAPERS_DIR**: Optional directory of Dutch PDFs to load at startup.

## API summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard (index.html) |
| POST | `/api/plagiarism/analyze` | Analyze text or PDF for plagiarism |
| POST | `/api/papers/upload-dutch` | Upload Dutch PDF (precompute embeddings once) |
| GET | `/api/papers/` | List stored Dutch papers |
| GET | `/health` | Health check |

## Frontend

- **index.html** is served at `/`.  
- **JavaScript** uses `fetch()` to call `POST /api/plagiarism/analyze`.  
- **Loading spinner** is shown while analyzing.  
- **CORS** is enabled so a separate frontend origin can call the API.

## Error handling

- Empty file → 400 with message.  
- Unsupported file format (non-PDF) → 400.  
- PDF extraction failure → 400 with detail.  
- Embedding errors → 503 with message.

Designed for **academic integrity** and college-level use.
