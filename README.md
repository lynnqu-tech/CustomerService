# Enterprise RAG Customer Service

面向电商场景的智能客服后端系统，支持订单查询、物流查询、退款咨询和 FAQ 自动回复。系统采用“规则路由优先 + RAG 兜底”的双路径架构，在结构化问题场景优先走数据库精确查询，在 FAQ 场景走向量检索与流式大模型回答。

## Features

- 规则路由优先：通过正则和关键词优先识别订单号、物流单号、退款意图
- LLM 分类兜底：规则未命中时，使用 LangChain + OpenAI Function Calling 进行意图分类
- DB 优先回答：订单与物流问题优先查询 PostgreSQL 并模板化返回
- RAG 兜底回答：FAQ 问题通过 Milvus 检索知识片段，再交给 LLM 生成回答
- SSE 流式输出：RAG 回答通过 `text/event-stream` 按 token 输出
- 多级缓存：Redis 缓存会话历史、热点响应和 Embedding
- 可观测性：Prometheus 指标暴露 + Grafana 仪表盘
- Docker Compose 一体化部署：包含 app、PostgreSQL、Redis、Milvus、Prometheus、Grafana

## Architecture

```text
[User Question]
      |
      v
[Query Router]
      |
      +--> Rule Match Success
      |        |
      |        v
      |   [PostgreSQL Query]
      |        |
      |        +--> Hit  -> Template Response
      |        |
      |        +--> Miss -> Fallback RAG
      |
      +--> Rule Miss / GENERAL_FAQ
               |
               v
         [Milvus Retrieval]
               |
               v
         [LLM Streaming Answer]
```

## Tech Stack

- Backend: FastAPI, Pydantic v2, pydantic-settings
- LLM / Orchestration: LangChain, langchain-openai
- Vector Search: Milvus, pymilvus
- Database: PostgreSQL, SQLAlchemy Async, asyncpg
- Cache: Redis
- Streaming: SSE
- Observability: Prometheus, Grafana, LangSmith
- Deployment: Docker Compose

## Project Structure

```text
app/
  api/
  core/
  services/
  models/
  schemas/
  db/
  utils/
tests/
docker-compose.yml
Dockerfile
prometheus.yml
requirements.txt
```

## Implemented Modules

- `app/services/query_router.py`
  Query Router，负责规则分流与 LLM 分类兜底

- `app/services/postgres_service.py`
  PostgreSQL 异步查询服务

- `app/services/vector_store.py`
  Milvus collection 初始化、upsert、Top-K 检索

- `app/services/document_service.py`
  PDF / TXT / MD 文档解析、分块、Embedding、向量入库

- `app/services/cache_service.py`
  Redis 会话缓存、响应缓存、Embedding 缓存

- `app/services/chat_service.py`
  主聊天链路，串联路由、数据库查询、RAG 检索与流式返回

- `app/services/response_service.py`
  DB 模板响应与 RAG 防幻觉提示

## Environment Variables

复制环境变量模板：

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

至少需要配置：

- `OPENAI_API_KEY`
- `API_TOKEN`

建议检查以下字段格式：

```env
CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
DOCUMENT_ALLOWED_EXTENSIONS=[".pdf",".txt",".md"]
```

## Local Development

### 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Run app locally

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Verify health

- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/api/v1/metrics`
- `http://127.0.0.1:8000/docs`

## Docker Deployment

启动完整依赖栈：

```powershell
docker compose up --build
```

查看容器状态：

```powershell
docker compose ps
```

查看应用日志：

```powershell
docker compose logs app --tail 100
```

### Exposed Ports

- App: `8000`
- PostgreSQL: `5432`
- Redis: `6379`
- Milvus: `19530`
- Milvus health: `9091`
- MinIO: `9000`
- MinIO Console: `9001`
- Prometheus: `9090`
- Grafana: `3001`

## API

### Health

```http
GET /api/v1/health
```

### Metrics

```http
GET /api/v1/metrics
```

### Chat

```http
POST /api/v1/chat
Authorization: Bearer <API_TOKEN>
Content-Type: application/json
```

Request body:

```json
{
  "session_id": "test-session-001",
  "question": "退款流程是什么？"
}
```

Response:

- DB 命中时返回 JSON
- RAG 场景返回 SSE

SSE format:

```text
data: {"type":"token","content":"..."}
data: {"type":"done"}
data: {"type":"error","content":"..."}
```

## FAQ Ingestion Example

先在容器中初始化 Milvus collection：

```powershell
@'
import asyncio
from app.core.config import get_settings
from app.services.vector_store import MilvusVectorStore

async def main():
    store = MilvusVectorStore(get_settings())
    await store.initialize()
    print("Milvus collection initialized.")

asyncio.run(main())
'@ | docker compose exec -T app python -
```

再导入 FAQ 文档：

```powershell
docker cp .\faq.txt rag-customer-service-app:/tmp/faq.txt
```

```powershell
@'
import asyncio
from app.core.config import get_settings
from app.services.vector_store import MilvusVectorStore
from app.services.document_service import DocumentService

async def main():
    settings = get_settings()
    store = MilvusVectorStore(settings=settings)
    await store.initialize()
    service = DocumentService(vector_store=store, settings=settings)
    result = await service.ingest_file("/tmp/faq.txt", metadata={"category": "faq"})
    print(result.model_dump())

asyncio.run(main())
'@ | docker compose exec -T app python -
```

## Monitoring

已接入指标：

- `rag_retrieval_duration_seconds`
- `postgres_query_duration_seconds`
- `llm_response_duration_seconds`
- `cache_hit_count`
- `total_requests`

可视化入口：

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3001`

## Testing

运行全部测试：

```powershell
python -m pytest
```

当前仓库已覆盖：

- 配置加载
- PostgreSQL 服务
- Milvus 向量服务
- 文档处理
- Query Router
- 缓存服务
- Chat 服务
- API 层
- Metrics
- Deployment 文件

## Current Status

当前项目已经完成：

- 基础服务启动
- Docker Compose 全栈编排
- Milvus collection 初始化
- FAQ 文档入库
- `/api/v1/chat` SSE 流式响应联调

当前仍可继续增强：

- 文档上传 API
- PostgreSQL 初始化脚本与测试数据
- 更丰富的 FAQ / 售后 / 物流知识库
- 更完整的生产级日志结构和 tracing

## License

仅用于学习、实验和项目展示。
