# 💬 Local RAG System (Ollama + Qdrant + Streamlit)# 💬 Local RAG System (Ollama + Qdrant + Streamlit)

A **privacy-first Retrieval-Augmented Generation (RAG)** system that runs entirely on your local machine. Upload documents and ask questions—get AI-powered answers with source attribution.A **local Retrieval-Augmented Generation (RAG)** system built using:

Built with:- 🧠 [Ollama](https://ollama.ai) for text generation and embeddings

- 🧠 [Ollama](https://ollama.ai) - Local LLM for generation & embeddings - 💾 [Qdrant](https://qdrant.tech) as a local vector database

- 💾 [Qdrant](https://qdrant.tech) - Vector database for semantic search - 🎨 [Streamlit](https://streamlit.io) for a simple and interactive web UI

- 🎨 [Streamlit](https://streamlit.io) - Interactive web interface

This project runs 100% offline — no external API calls — perfect for privacy-preserving AI assistants and local knowledge bases.

**100% offline** — No cloud APIs, no data leaves your machine.

---

---

## 🚀 Features

## ✨ Features

✅ Upload and process **PDF or TXT** documents

- 📄 **Document Processing** - Upload PDF and TXT files✅ Split documents into **text chunks**

- 🔍 **Smart Chunking** - Overlapping chunks for better context preservation✅ Generate **embeddings locally** using `mxbai-embed-large`

- 🎯 **Semantic Search** - Vector similarity search with Qdrant✅ Store and search vectors using **Qdrant**

- 🤖 **Local LLM** - Powered by Ollama (llama3.2)✅ Answer questions contextually using `llama3.2`

- ⚡ **Streaming Responses** - Real-time answer generation✅ Simple **Streamlit web interface**

- 📊 **Source Attribution** - See which document chunks were used with similarity scores✅ Optional **multi-file upload** and **database reset**

- 📝 **Enhanced Metadata** - Track source files, timestamps, chunk indices

- 🗑️ **Database Management** - Clear and reset your vector database---

- 🐳 **Docker Support** - One-command deployment with docker-compose

## 🧩 Architecture

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Web Interface                │
│         (Upload, Query, Display Results)            │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Document   │  │  Embedding   │  │ Vector Store │
│  Processor   │  │   Service    │  │   (Qdrant)   │
│  (PyPDF2)    │  │  (Ollama)    │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Generation  │
                  │   (Ollama)   │
                  │  llama3.2    │
                  └──────────────┘
```

---

## 📁 Project Structure

```
local-rag/
├── server/
│   ├── app.py                 # Main Streamlit application
│   ├── config.py              # Configuration management (.env support)
│   ├── embedding.py           # Ollama embedding functions
│   ├── vectorstore.py         # Qdrant vector operations
│   ├── document_processor.py  # PDF/TXT processing & chunking
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Environment variables (you create this)
│   └── .env.example           # Example environment configuration
├── docker-compose.yml         # Docker orchestration
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Prerequisites:** Docker and Docker Compose installed

1. **Clone the repository**

   ```bash
   git clone https://github.com/NimanthaSupun/local-RAG-Application.git
   cd local-rag
   ```

2. **Create `.env` file** (optional - uses defaults if omitted)

   ```bash
   cp server/.env.example server/.env
   # Edit server/.env to customize settings
   ```

3. **Start everything with Docker Compose**

   ```bash
   docker-compose up
   ```

4. **Open your browser**
   ```
   http://localhost:8501
   ```

That's it! Ollama, Qdrant, and the Streamlit app will all start automatically.

---

### Option 2: Manual Setup (Windows)

**Prerequisites:**

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- [Qdrant](https://qdrant.tech) running (Docker or native)

#### Step 1: Install Ollama Models

```powershell
ollama pull mxbai-embed-large
ollama pull llama3.2
```

#### Step 2: Start Qdrant (Docker)

```powershell
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

Or use Qdrant Cloud/native installation.

#### Step 3: Setup Python Environment

```powershell
cd server
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Step 4: Configure Environment

Create `server/.env` file (or copy from `.env.example`):

```env
OLLAMA_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=docs
EMBED_MODEL=mxbai-embed-large
GEN_MODEL=llama3.2
EMBED_DIM=1024
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
```

#### Step 5: Run the Application

```powershell
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## 🎮 Usage

### 1. Upload Documents

- Click **"Browse files"** in the sidebar
- Select one or more PDF or TXT files
- Wait for processing (you'll see chunk count)

### 2. Ask Questions

- Type your question in the text input
- Click **"Ask"** button
- View the AI-generated answer with streaming effect
- See source chunks with similarity scores

### 3. Manage Database

- Click **"🗑️ Clear Qdrant Data"** to reset the database
- Useful when starting a new project or removing old documents

---

## ⚙️ Configuration

All configuration is managed via environment variables in `server/.env`:

| Variable            | Default                  | Description                    |
| ------------------- | ------------------------ | ------------------------------ |
| `OLLAMA_URL`        | `http://localhost:11434` | Ollama API endpoint            |
| `QDRANT_URL`        | `http://localhost:6333`  | Qdrant API endpoint            |
| `QDRANT_COLLECTION` | `docs`                   | Collection name in Qdrant      |
| `EMBED_MODEL`       | `mxbai-embed-large`      | Embedding model name (Ollama)  |
| `GEN_MODEL`         | `llama3.2`               | Generation model name (Ollama) |
| `EMBED_DIM`         | `1024`                   | Embedding vector dimension     |
| `CHUNK_SIZE`        | `500`                    | Characters per chunk           |
| `CHUNK_OVERLAP`     | `50`                     | Overlap between chunks         |
| `TOP_K`             | `3`                      | Number of chunks to retrieve   |

---

## 🐳 Docker Deployment

The `docker-compose.yml` includes three services:

1. **Ollama** - LLM service with GPU support (optional)
2. **Qdrant** - Vector database
3. **Streamlit App** - Web interface

### GPU Support (NVIDIA)

If you have an NVIDIA GPU, uncomment the GPU sections in `docker-compose.yml`:

```yaml
# Uncomment for GPU support
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Customize Ports

Edit `docker-compose.yml` to change default ports:

- Streamlit: `8501`
- Qdrant: `6333`, `6334`
- Ollama: `11434`

---

## 🔧 Advanced Features

### Overlapping Chunks

The system uses overlapping chunks to preserve context across boundaries:

```python
# Example: 500 char chunks with 50 char overlap
"...end of chunk 1 overlap..."
"...overlap start of chunk 2..."
```

This ensures important information isn't split awkwardly.

### Enhanced Metadata

Each vector stores rich metadata:

```json
{
  "text": "chunk content...",
  "source_file": "document.pdf",
  "chunk_index": 0,
  "total_chunks": 15,
  "upload_timestamp": "2025-11-04T10:30:00",
  "file_type": "application/pdf"
}
```

Use this to:

- Filter results by source file
- Show document names in UI
- Debug chunking issues
- Track upload history

### Streaming Responses

Answers are streamed token-by-token for a better user experience (like ChatGPT).

---

## 🛠️ Troubleshooting

### Ollama Connection Error

**Error:** `Cannot connect to Ollama`

**Solution:**

1. Check Ollama is running: `ollama list`
2. Start Ollama service: `ollama serve`
3. Verify URL in `.env` matches your Ollama endpoint

### Qdrant Connection Error

**Error:** `Cannot connect to Qdrant`

**Solution:**

1. Check Qdrant is running: `docker ps` (if using Docker)
2. Test connection: `curl http://localhost:6333`
3. Verify URL in `.env`

### Model Not Found

**Error:** `Model 'llama3.2' not found`

**Solution:**

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Memory Issues

If you run out of memory:

- Use a smaller model (e.g., `llama3.2:1b`)
- Reduce `CHUNK_SIZE` and `TOP_K`
- Process fewer documents at once

### Port Already in Use (Windows)

**Error:** `WinError 10013` or port 8501 in use

**Solution:**

```powershell
# Find process using port 8501
netstat -ano | findstr :8501
# Kill the process (replace <PID> with actual PID)
taskkill /PID <PID> /F

# Or use a different port
streamlit run app.py --server.port 8502
```

---

## 🧪 Development

### Project Structure (Refactored)

The codebase is modular for easy maintenance:

- **`config.py`** - Centralized configuration with environment variable loading
- **`embedding.py`** - Ollama embedding generation
- **`vectorstore.py`** - Qdrant client and vector operations
- **`document_processor.py`** - File reading, text extraction, chunking
- **`app.py`** - Streamlit UI and orchestration

### Adding New Features

1. **Custom Chunking Strategy** - Edit `document_processor.py`
2. **Different Embedding Models** - Change `EMBED_MODEL` in `.env`
3. **Alternative LLMs** - Change `GEN_MODEL` in `.env`
4. **New Document Types** - Add parsers to `document_processor.py`

### Testing

Run health checks:

```python
from config import check_services
check_services()  # Verifies Ollama and Qdrant connectivity
```

---

## 📊 Performance Tips

1. **Use GPU** - Enable GPU support in docker-compose for faster inference
2. **Adjust Chunk Size** - Larger chunks = more context but slower search
3. **Tune TOP_K** - More chunks = better context but slower generation
4. **Model Selection** - Smaller models are faster but less capable
5. **Batch Processing** - Upload multiple documents at once

---

## 🤝 Contributing

Contributions welcome! Ideas:

- [ ] Support for more file types (DOCX, HTML, Markdown)
- [ ] Hybrid search (vector + keyword/BM25)
- [ ] Multi-language support
- [ ] Query history and session management
- [ ] Re-ranking with cross-encoder models
- [ ] Export chat history
- [ ] API endpoints for programmatic access

---

## 📝 License

This project is open source. Add a LICENSE file (e.g., MIT) to specify terms.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) - Easy local LLM deployment
- [Qdrant](https://qdrant.tech) - High-performance vector search
- [Streamlit](https://streamlit.io) - Rapid UI development
- [PyPDF2](https://pypdf2.readthedocs.io/) - PDF text extraction

---

## 📧 Support

- **Issues:** [GitHub Issues](https://github.com/NimanthaSupun/local-RAG-Application/issues)
- **Discussions:** [GitHub Discussions](https://github.com/NimanthaSupun/local-RAG-Application/discussions)

---
