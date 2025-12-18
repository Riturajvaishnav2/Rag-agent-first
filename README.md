# 🚀 Text File Based Query Processing AI Agent
### *RAG-Based · Fully Local · Powered by Ollama + Mistral + ChromaDB*

This project is a **production-grade AI Speech processing Agent** that reads long text, chunks it, embeds it, performs semantic retrieval using ChromaDB, and generates accurate query response using **Mistral LLM through Ollama** — all running **locally** and fully containerized with Docker Compose.

---

# 🧱 Project Structure

AmbedkarGPT-Intern-Task/  
├── main.py              → Main RAG pipeline  
├── requirements.txt     → Python dependencies  
├── files/AI-for-Education-RAG.pdf           → Input file for summarization  
├── docker-compose.yml   → Multi-container setup (App + ChromaDB)  
├── Dockerfile           → Python app image  
├── .dockerignore  
└── README.md

---

# 🧠 Key Features

- Reads any `.txt` document or speech  
- Splits text into overlapping semantic chunks  
- Embeds chunks using Sentence-Transformers  
- Stores embeddings in **ChromaDB**  
- Retrieves relevant chunks via vector similarity  
- Processes queries using **Mistral LLM** via Ollama  
- Fully containerized with Docker Compose  
- Works fully offline — full privacy  
- Reproducible setup and single-command runs

---

# 🛠 Tech Stack

- Python  
- LangChain  
- Sentence-Transformers  
- ChromaDB  
- Ollama (Mistral LLM)  
- Docker & Docker Compose  
- Git

---

# ⚠️ Prerequisites

Install the following before running the project:

- Docker: https://www.docker.com/products/docker-desktop/  
- Git: https://git-scm.com/downloads  
- (Optional, for running without Docker) Ollama: https://ollama.com/download

If you're using Docker Compose, Ollama runs as a container and you **do not** need Ollama installed on the host.

---

# 📥 Clone the Repository

    git clone https://github.com/Busted-pinch/AmbedkarGPT-Intern-Task.git
    cd AmbedkarGPT-Intern-Task

---

# ✍️ Prepare Your Input File

Edit or replace the file named `speech.txt` in the repo root.

- Must be plain `.txt`  
- Can be long or short  
- This is the file the agent will generate a response for the query

---

# ⚙️ Configuration (.env)

This project reads configuration from environment variables. Defaults are provided in `.env` (edit it to customize):

- `SPEECH_FILE` (default: `speech.txt`)  
- `PERSIST_DIR` (default: `/data/chroma`)  
- `EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)  
- `CHUNK_SIZE` (default: `1000`)  
- `CHUNK_OVERLAP` (default: `100`)  
- `TOP_K` (default: `3`)  
- `OLLAMA_API_URL` (default: `http://ollama:11434` in Docker, use `http://localhost:11434` for local runs)

---

# ▶️ Execution

## Option A (recommended): Docker Compose

### 1️⃣ Start containers

    sudo docker compose up --build -d

### 2️⃣ Pull the Mistral model (first run only)

    sudo docker compose exec ollama ollama pull mistral

### 3️⃣ Run a query

    sudo docker compose exec app python main.py -q "Summarize the speech"

Optional debug mode (prints retrieved chunks before calling the LLM):

    sudo docker compose exec app python main.py --debug -q "Summarize the speech"

## Option B: Run locally (no Docker)

### 1️⃣ Start Ollama (host)

    ollama serve

### 2️⃣ Pull the Mistral model (host)

    ollama pull mistral

### 3️⃣ Run the script

    OLLAMA_API_URL=http://localhost:11434 PERSIST_DIR=./chroma python main.py -q "Summarize the speech"

---

# 🔍 Internal Workflow (what happens per query)

1. Loads `speech.txt`  
2. Splits into chunks (default chunk size: 1000, overlap: 100)  
3. Generates embeddings with Sentence-Transformers  
4. Persists embeddings to ChromaDB  
5. Performs semantic search for the query  
6. Retrieves the top relevant chunks  
7. Sends retrieved text to **Mistral** via **Ollama**  
8. Prints the final response to stdout/logs

Sample output format:

    === LLM ANSWER ===
    <your generated answer here>

---

# 🧹 Troubleshooting

### Ollama not reachable
If using Docker Compose, ensure the `ollama` service is up:

    sudo docker compose ps

If running locally (no Docker), ensure Ollama is serving on host:

    ollama serve

### Mistral model missing
Docker Compose:

    sudo docker compose exec ollama ollama pull mistral

Local (no Docker):

    ollama pull mistral

### Docker error: "port is already allocated"
This happens if you expose the Ollama container port to the host and something is already using `11434` (often a host Ollama).

- Easiest: keep the default `docker-compose.yml` (no Ollama host port mapping).
- If you need host access to the container: stop the host Ollama or change the port mapping in `docker-compose.yml` (e.g., `11435:11434`).

### ChromaDB index problems (e.g., dimension mismatch, corrupted DB)
Reset the Chroma volume and rebuild:

    sudo docker compose down
    sudo docker compose up --build -d

If you changed `speech.txt` and want to re-index from scratch, remove the `chroma_data` volume:

    sudo docker compose down -v

### App container doesn't print an answer
The `app` service runs `sleep infinity` to stay alive. Run queries with:

    sudo docker compose exec app python main.py -q "your question"

---

# 🎯 Expected Output

- A concise, context-aware response printed in the logs  
- RAG-driven results produced using local LLM inference and vector search  
- Reproducible results on any machine with the prerequisites installed

---

# 🔗 Follow me on LinkedIn
https://www.linkedin.com/in/prathamesh-mete
