# NEXUS Deployment Guide

---

## Option 1: Local Development (Recommended to Start)

### Minimum setup — local models only, zero cloud cost

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull required models
ollama pull qwen2.5:7b      # Perception Engine + PRIVATE tasks
ollama pull qwen2.5:32b     # Medium complexity tasks (optional but recommended)

# 3. Clone and install NEXUS
git clone https://github.com/glen200392/nexus.git
cd nexus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — at minimum set:
#   OLLAMA_BASE_URL=http://localhost:11434

# 5. Initialize
python nexus.py init

# 6. Run
python nexus.py start
```

**What works with local-only setup:**
- All PRIVATE tier tasks (memory agent, maintenance)
- Research tasks using local models (slower, lower quality)
- Data analysis, ML pipeline
- Shell, code, browser agents
- All Skills and MCP servers (except cloud API ones)

---

## Option 2: Hybrid (Local + Cloud LLMs)

Adds cloud models for INTERNAL/PUBLIC tier tasks, keeping PRIVATE tasks local.

```bash
# .env additions:
ANTHROPIC_API_KEY=sk-ant-...    # claude-opus/sonnet/haiku
OPENAI_API_KEY=sk-...           # gpt-4o, gpt-4o-mini
```

**Recommended model profile for production:**
```
PRIVATE tasks    → qwen2.5:72b (local, best local quality)
MEDIUM tasks     → claude-sonnet-4-6 (quality + cost balance)
HIGH tasks       → claude-opus-4-6 (best reasoning)
LOW tasks        → claude-haiku-4-5 (fast + cheap)
```

---

## Option 3: Full Stack with Docker

### Services

| Service | Port | Purpose |
|---------|------|---------|
| NEXUS API | 8000 | REST API |
| NEXUS Dashboard | 7800 | Web UI + SSE |
| Ollama | 11434 | Local LLM serving |
| ChromaDB | 8001 | Vector store |
| Neo4j | 7474 / 7687 | Graph database (lineage) |
| Redis | 6379 | Session cache |
| MLflow | 5000 | ML experiment tracking |
| Prometheus | 9090 | Metrics |
| Grafana | 3000 | Dashboards |

### docker-compose.yml (minimal)

```yaml
version: "3.9"
services:
  nexus:
    build: .
    ports:
      - "8000:8000"
      - "7800:7800"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
    depends_on:
      - ollama
      - chromadb
      - redis
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    # GPU support (uncomment if available):
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  neo4j:
    image: neo4j:5.25-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  ollama_data:
  chroma_data:
  neo4j_data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium --with-deps

# Application
COPY . .

# Create data directories
RUN mkdir -p data/vector_store data/graph logs

# Initialize on start
ENTRYPOINT ["python", "nexus.py"]
CMD ["start"]
```

### Build and run

```bash
# Build
docker build -t nexus:latest .

# Start full stack
docker-compose up -d

# Pull Ollama models inside container
docker exec -it nexus-ollama-1 ollama pull qwen2.5:7b
docker exec -it nexus-ollama-1 ollama pull qwen2.5:32b

# Initialize NEXUS
docker exec -it nexus-nexus-1 python nexus.py init

# Check logs
docker-compose logs -f nexus
```

---

## Option 4: Cloud Deployment (Production)

### Environment Requirements

- **Compute**: 8+ vCPU, 32GB+ RAM (for qwen2.5:32b local model)
- **GPU**: NVIDIA GPU with 16GB+ VRAM for qwen2.5:72b (optional)
- **Storage**: 100GB+ SSD (model weights + vector store)
- **Network**: Private VPC recommended (internal LLM traffic)

### AWS/GCP/Azure

```bash
# Example: AWS EC2 g4dn.2xlarge (GPU instance)
# - 8 vCPU, 32GB RAM, 1x NVIDIA T4 (16GB)
# - $0.75/hr on-demand, ~$0.25/hr spot

# Deploy with Docker Compose
scp -r ./nexus ubuntu@your-instance:/home/ubuntu/nexus
ssh ubuntu@your-instance
cd /home/ubuntu/nexus
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes (Helm)

```yaml
# helm/values.yaml (basic)
nexus:
  replicas: 2
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "8"
      memory: "16Gi"

ollama:
  enabled: true
  model: qwen2.5:7b
  gpu:
    enabled: false  # set true if GPU nodes available
```

### Environment Variables for Production

```bash
# Security
NEXUS_SECRET_KEY=<32-char-random>
A2A_API_KEY=<random>

# Governance
NEXUS_DAILY_BUDGET_USD=20.00
NEXUS_MONTHLY_BUDGET_USD=400.00

# Performance
NEXUS_MAX_CONCURRENT_LLM_CALLS=10
NEXUS_MAX_CONCURRENT_AGENTS=20

# Monitoring
LOG_LEVEL=WARNING
PROMETHEUS_ENABLED=true
```

---

## Data Persistence

### What to persist across restarts

| Path | Contents | Importance |
|------|----------|------------|
| `data/vector_store/` | ChromaDB vector embeddings | **Critical** — losing this loses all memory |
| `data/audit.db` | Immutable audit log | **Critical** — compliance record |
| `data/lineage_graph.json` | Data lineage (if no Neo4j) | Important |
| `data/bias_reports/` | Bias audit reports | Important |
| `config/prompts/versions/` | Prompt version history | Recommended |

### Backup strategy

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR=/backups/nexus/$DATE

mkdir -p $BACKUP_DIR

# Vector store
cp -r /app/data/vector_store $BACKUP_DIR/
# Audit log
cp /app/data/audit.db $BACKUP_DIR/
# Config
cp -r /app/config/prompts/versions $BACKUP_DIR/prompts/
cp -r /app/data/bias_reports $BACKUP_DIR/

echo "Backup complete: $BACKUP_DIR"
```

---

## Health Checks

```bash
# System status
python nexus.py status

# API health (if dashboard running)
curl http://localhost:7800/api/status

# Check Ollama
curl http://localhost:11434/api/tags

# Check ChromaDB
curl http://localhost:8001/api/v1/heartbeat
```

---

## Monitoring

NEXUS exposes metrics via the Prometheus MCP server. Recommended alerts:

| Alert | Threshold | Action |
|-------|-----------|--------|
| Daily LLM cost > 90% budget | $budget × 0.9 | Auto-downgrade (built-in) |
| Task quality score < 0.65 (7-day avg) | 0.65 | Trigger prompt optimization |
| Agent failure rate > 20% | 20% | Investigate + alert Slack |
| EU AI Act blocked tasks | Any | Review + log |

---

## Security Hardening

```bash
# 1. Never commit .env
git check-ignore .env  # Should return .env

# 2. Restrict data directory permissions
chmod 700 data/
chmod 600 data/audit.db

# 3. Use environment-specific API key rotation
# Rotate ANTHROPIC_API_KEY, OPENAI_API_KEY monthly

# 4. Network: bind dashboard to internal IP only
uvicorn nexus.api.dashboard:app --host 127.0.0.1 --port 7800
# Use nginx/traefik as reverse proxy with TLS

# 5. Audit log review
sqlite3 data/audit.db "SELECT * FROM audit_log WHERE cost_usd > 0.5 ORDER BY timestamp DESC LIMIT 20"
```
