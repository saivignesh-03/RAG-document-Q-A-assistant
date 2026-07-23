# HBO Document Q&A Assistant (RAG Pipeline)

A Retrieval-Augmented Generation (RAG) application that answers natural-language questions about a PDF document, using **hybrid search (semantic + keyword) with reranking**, and generates grounded answers with a **local, free, open-source LLM** (no API costs).

Built as a learning project to gain hands-on, end-to-end experience with modern RAG architecture — from raw PDF to a working chat UI.

---

## What this project does

- Loads a PDF and splits it into searchable chunks
- Finds the most relevant chunks for a question using **two independent search methods** (semantic similarity + keyword matching), combined with **Reciprocal Rank Fusion (RRF)**
- Re-scores the shortlisted chunks with a **cross-encoder reranker** for higher accuracy
- Sends the final, most relevant chunks to a local LLM (**Ollama / Llama 3.1**) to generate a grounded answer
- Refuses to answer when the document doesn't contain the information (no hallucination)
- Exposes everything through a **FastAPI backend** and a **Streamlit chat UI**

---

## Architecture

```
PDF document
     |
     v
Chunking (RecursiveCharacterTextSplitter)
     |
     v
Embeddings (nomic-embed-text) --> stored in ChromaDB
     |
     |-------------------------------|
     v                               v
Semantic search (top 10)     Keyword search / BM25 (top 10)
     |                               |
     |---------------|---------------|
                      v
         RRF fusion (rank-based, top 5)
                      |
                      v
      Cross-encoder reranker (top 3)
                      |
                      v
        LLM (Llama 3.1 via Ollama)
                      |
                      v
              Grounded answer
```

This is exposed via:

- **`main.py`** — FastAPI backend (`POST /ask`)
- **`app.py`** — Streamlit chat interface (calls the FastAPI backend)

---

## Tech stack

| Purpose                 | Technology                                                            |
| ----------------------- | --------------------------------------------------------------------- |
| Orchestration           | LangChain                                                             |
| LLM (answer generation) | Ollama — Llama 3.1 (local, free)                                     |
| Embeddings              | Ollama — nomic-embed-text (local, free)                              |
| Vector database         | ChromaDB                                                              |
| Keyword search          | BM25 (`rank_bm25`)                                                  |
| Result fusion           | Reciprocal Rank Fusion (RRF)                                          |
| Reranking               | Cross-encoder (`sentence-transformers`, `ms-marco-MiniLM-L-6-v2`) |
| Backend API             | FastAPI + Uvicorn                                                     |
| Frontend UI             | Streamlit                                                             |
| PDF parsing             | pypdf                                                                 |

No paid API keys are required anywhere in this project — everything runs locally.

---

## Project structure

```
rag-qa-assistant/
├── data/
│   └── HBO.pdf              # source document (swap for your own PDF)
├── chroma_db/                # generated vector store (created automatically, do not commit)
├── main.py                   # FastAPI backend — the full RAG pipeline
├── app.py                    # Streamlit chat UI
├── requirements.txt
└── README.md
```

---

## Setup instructions

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed on your machine

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-qa-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv rag-qa-assistant
# Windows
rag-qa-assistant\Scripts\activate
# Mac/Linux
source rag-qa-assistant/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the local models via Ollama

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 5. Add your document

Place a PDF file inside the `data/` folder, and update the filename in `main.py` (search for `PyPDFLoader(...)`) if it isn't named `HBO.pdf`.

### 6. Run the backend

```bash
uvicorn main:app
```

Wait for the terminal to print `Pipeline ready!` and `Uvicorn running on http://127.0.0.1:8000`. **Do not use `--reload`** — see the Known Issues section below for why.

Leave this terminal running.

### 7. Run the frontend (in a second, separate terminal)

```bash
# activate the same venv in this new terminal first
streamlit run app.py
```

This opens the chat interface in your browser automatically.

---

## Usage

1. Type a question about your document into the text box
2. Click **Ask**
3. The answer appears, along with expandable **Sources** showing which chunks were used and their relevance scores

If the document doesn't contain the answer, the assistant will say so explicitly rather than guessing.

---

## Development journey: issues encountered and how they were solved

This project was built and debugged step by step. Below is an honest record of the real problems hit along the way — useful if you run into the same ones.

### Environment setup

- **PowerShell blocked venv activation** (`running scripts is disabled on this system`) → fixed with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **`ollama` command not recognized** right after installing → the terminal was opened before Ollama finished registering to PATH. Fixed by closing and reopening the terminal.
- **`rag-qa-assistant\Scripts\activate` failed as "module not found"** → was run one directory too high (parent folder instead of the project folder). Fixed by `cd`-ing into the correct folder first.

### Package/import issues (LangChain restructuring)

- `ModuleNotFoundError: langchain_community` → package wasn't installed separately; fixed with `pip install langchain-community`.
- `ModuleNotFoundError: langchain.text_splitter` → LangChain moved text splitters into their own package; fixed by importing from `langchain_text_splitters` instead.
- `ModuleNotFoundError: langchain.prompts` → similarly moved; fixed by importing `ChatPromptTemplate` from `langchain_core.prompts`.

### File path issues

- `ValueError: File path data/HBO.pdf is not a valid file or url` → the notebook's working directory was the parent folder (`Projectsssss`), not the project folder (`rag-qa-assistant`), so the relative path didn't resolve. Fixed by using the correct relative path (or an absolute path).

### Retrieval logic bugs (the most valuable lessons)

- **Fake "zero-score" matches in keyword search**: the original BM25 implementation always returned the top `k` results regardless of score. Since most chunks don't contain a given query's exact words, they all scored `0` — and irrelevant chunks were returned anyway, purely due to arbitrary sort-order ties. **Fixed** by filtering out any chunk with a score of `0` before ranking.
- **A genuine tie in RRF fusion**: even after the fix above, two different chunks that both legitimately mentioned "Charles Dolan" (the founding story, and a later summary section) ended up with an identical RRF score. This wasn't a bug — RRF only considers rank position, not the actual text, so two chunks that swap 1st/2nd place across the two search methods will always tie. **Solved** by adding a cross-encoder reranker, which reads the question and each candidate chunk's full text together and correctly judged one chunk as more relevant than the other.

### Backend (FastAPI) issues

- `Error loading ASGI app. Attribute "app" not found in module "main"` → the file hadn't been saved yet, or the `app = FastAPI()` line was missing/misnamed.
- **`500 Internal Server Error` on `/ask`** caused by a CUDA crash: `uvicorn --reload` starts two processes (a watcher and the actual server), and both attempted to load the LLM onto the GPU simultaneously, crashing Ollama's underlying server process (`CUDA error: shared object initialization failed`). **Fixed** by running `uvicorn main:app` **without** `--reload`, so only one process loads the model.

### Frontend (Streamlit) issues

- `requests.exceptions.ConnectionError: ... actively refused it` → the FastAPI backend wasn't running when Streamlit tried to call it. Streamlit only calls the backend over HTTP; it does not run the RAG pipeline itself. **Fixed** by keeping both the backend (`uvicorn main:app`) and frontend (`streamlit run app.py`) running simultaneously in two separate terminals.

### Notebook-specific gotchas (relevant if you develop further in Jupyter)

- Jupyter kernels can restart silently (memory limits, IDE updates), which wipes all variables and function definitions in memory even though the code is still visible in the cells. If you hit a sudden `NameError` for something you already defined, use "Run All" to rebuild everything from the top rather than re-running just the newest cell.

---

## Known limitations / possible improvements

- Currently tuned for a single PDF loaded at startup — could be extended to support multiple documents or dynamic uploads
- `k_final` (number of chunks sent to the LLM) is fixed at 3 regardless of how relevant they actually are — could add a minimum relevance-score cutoff so weakly-related chunks are dropped rather than always padding to 3
- Runs entirely on local hardware (Ollama) — response time depends on your CPU/GPU
- No conversation memory — each question is answered independently, with no awareness of previous turns

---

## ⚠️ Important — before running uvicorn + streamlit

Only one process on your machine should be holding the LLM in GPU memory at a time. If any other process is also using Ollama (e.g. a Jupyter notebook with ChatOllama still loaded, or ollama run llama3.1 open in another terminal) while you start the backend, both will try to grab the GPU simultaneously and Ollama's server will crash with:

```bash
ollama._types.ResponseError: llama-server process has terminated: exit status 0xc0000409
CUDA error: shared object initialization failed (status code: 500)
```

**Before starting the app, always:**

1. Close/shut down any Jupyter notebook kernels that reference `ChatOllama` or `OllamaEmbeddings`
2. Make sure no other terminal has `ollama run <model>` open
3. Start the backend **without** `--reload` (`uvicorn main:app`, not `uvicorn main:app --reload`) — `--reload` spawns two processes internally and causes this exact same crash on its own
4. Only then run `streamlit run app.py` in a second terminal

If you ever see this error, it almost always means two processes are competing for the GPU — close one and restart the backend.

## License

Add your preferred license here (e.g. MIT) before publishing.
