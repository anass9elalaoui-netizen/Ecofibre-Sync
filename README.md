# EcoFibre Sync 🚀

**EcoFibre Sync** is an intelligent, full-stack application designed to ingest, analyze, and query intervention reports (PDF, DOCX, TXT) using advanced **Retrieval-Augmented Generation (RAG)** and **LangGraph Agents**.

Built with modern technologies, it features an asynchronous architecture capable of processing heavy documents without blocking the user interface, backed by a vector database for semantic search.

---

## 🌟 Key Features

- **Asynchronous Document Ingestion:** Upload intervention reports which are processed in the background (parsing, chunking, embedding) via **Celery** and **Redis**.
- **Vector Search (RAG):** Uses **PostgreSQL + pgvector** to store and query document embeddings for highly relevant semantic search.
- **AI-Powered Troubleshooting:** Employs **LangGraph** to build intelligent agents that analyze reports and suggest resolutions to technical issues.
- **Modern Frontend:** A responsive UI built with **React**, **Vite**, and **Tailwind CSS**.
- **Robust API:** A **FastAPI** backend with automated interactive documentation (Swagger UI & ReDoc).
- **Dockerized Environment:** One-command setup using `docker-compose`.

---

## 📸 Screenshots

<details open>
<summary><b>Click to view Screenshots</b></summary>

### Base de Connaissance
<img src="screen shot/base de conaissance .png" alt="Base de connaissance" width="800">

### Ingestion Logs & Embeddings
<img src="screen shot/log embd.png" alt="Log Embeddings" width="800">
<br>
<img src="screen shot/log.png" alt="General Logs" width="800">

### Image & Test Results
<img src="screen shot/image test .png" alt="Image Test" width="800">
<br>
<img src="screen shot/result image test .png" alt="Result Image Test" width="800">

### Additional UI Views
<img src="screen shot/1.png" alt="UI View 1" width="800">
<br>
<img src="screen shot/2.png" alt="UI View 2" width="800">

</details>

---

## 🛠️ Technology Stack

### Backend
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
- **ORM:** [SQLAlchemy](https://www.sqlalchemy.org/) (Async)
- **AI / LLM:** [LangGraph](https://python.langchain.com/docs/langgraph/) & LangChain
- **Background Tasks:** [Celery](https://docs.celeryq.dev/en/stable/)

### Frontend
- **Framework:** [React 18](https://react.dev/) + [Vite](https://vitejs.dev/)
- **Styling:** [Tailwind CSS](https://tailwindcss.com/)
- **Icons:** Lucide React

### Infrastructure & Data
- **Database:** PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector)
- **Message Broker / Cache:** Redis 7
- **Deployment:** Docker, Docker Compose, Render (IaC via `render.yaml`)

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anass9elalaoui-netizen/Ecofibre-Sync.git
   cd Ecofibre-Sync
   ```

2. **Configure Environment Variables:**
   Copy the example environment file and fill in your secrets (e.g., `GOOGLE_API_KEY`).
   ```bash
   cp .env.example .env
   ```

3. **Start the application with Docker:**
   ```bash
   docker compose up --build -d
   ```
   *This command spins up the Database (pgvector), Redis, FastAPI backend, Celery worker, and the React frontend.*

4. **Access the Application:**
   - **Frontend App:** [http://localhost](http://localhost) or [http://localhost:5173](http://localhost:5173)
   - **Backend API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Project Structure

```text
ecofibre-sync/
├── backend/               # FastAPI Application & Celery Workers
│   ├── app/
│   │   ├── api/           # REST Endpoints (v1)
│   │   ├── core/          # Configuration & Database Connection
│   │   ├── models/        # SQLAlchemy ORM Models
│   │   ├── schemas/       # Pydantic Validation Models
│   │   ├── services/      # Business Logic
│   │   ├── tasks/         # Celery Background Tasks (Ingestion)
│   │   └── agents/        # LangGraph AI Agents
│   ├── main.py            # FastAPI Entry Point
│   └── Dockerfile         # Backend Docker Image
├── frontend/              # React + Vite UI
│   ├── src/               # React Components & Pages
│   └── Dockerfile         # Frontend Nginx/Vite Docker Image
├── screen shot/           # Project Screenshots
├── docker-compose.yml     # Full Stack Orchestration
└── render.yaml            # Render Infrastructure-as-Code Configuration
```

---

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is proprietary and confidential. All rights reserved.
