# IR-Trends Backend API

Information Retrieval Trends 项目的后端API服务，使用FastAPI构建。

## 功能特性

- **Trending Papers**: 热门论文列表API
- **Trending Topics**: 热门话题列表API  
- **Paper Q&A**: 论文问答聊天API (支持流式响应)

## 技术栈

- FastAPI - 现代高性能Web框架
- Pydantic - 数据验证和序列化
- Uvicorn - ASGI服务器
- Python 3.8+

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python run.py
```

或者使用uvicorn直接启动：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. 访问API文档

启动后访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API端点

### 论文相关

- `GET /api/papers` - 获取热门论文列表
  - 查询参数: search, category, time_range, limit, offset

### 话题相关

- `GET /api/topics` - 获取热门话题列表
  - 查询参数: search, source, time_range, sort_by, limit, offset

### 聊天相关

- `POST /api/chat` - 论文问答聊天 (流式响应)
  - 请求体: {"question": "your question"}

### 系统相关

- `GET /` - 根路径
- `GET /api/health` - 健康检查

## 项目结构




## 开发说明

当前版本使用Mock数据进行开发和测试。后续可以：

1. 集成真实数据库 (PostgreSQL/MongoDB)
2. 添加用户认证和授权
3. 集成外部API (arXiv, Google Scholar等)
4. 添加缓存层 (Redis)
5. 实现真实的AI问答功能

## CORS配置

已配置CORS支持前端开发服务器：
- http://localhost:5173 (Vite默认端口)
- http://localhost:3000 (React默认端口)