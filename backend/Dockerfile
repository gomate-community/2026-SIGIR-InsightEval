# 使用 Python 3.12 作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并安装依赖
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复制整个 backend 目录到 /app/backend
COPY backend /app/backend

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DOCKER_ENV=1

# 暴露端口
EXPOSE 8000

# 切换到 /app 目录并执行 main.py
WORKDIR /app
CMD ["python", "/app/backend/main.py"]

