from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import asyncio
from loguru import logger

from backend.models.schemas import ChatRequest
from backend.engines.llm_engine import LLMEngine, ChatMessage, LLMConfig, OneAINexusConfig
from backend.config import Config

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Initialize LLM engine with config from Config
# 默认使用配置文件中的 LLM 配置（OneAINexus）
llm_config = LLMConfig(**Config.get_llm_config())
llm_engine = LLMEngine(llm_config)

# 启动时检查 LLM 连接
@router.on_event("startup")
async def startup_event():
    """启动时检查 LLM 引擎连接"""
    try:
        is_healthy = await llm_engine.health_check()
        if is_healthy:
            logger.info("✅ LLM 引擎连接正常")
        else:
            logger.warning("⚠️ LLM 引擎连接异常，但服务仍会启动")
    except Exception as e:
        logger.error(f"❌ LLM 引擎初始化失败: {str(e)}")


@router.get("/health")
async def chat_health_check():
    """聊天服务健康检查"""
    try:
        is_healthy = await llm_engine.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "chat",
            "llm_connected": is_healthy
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "chat",
            "llm_connected": False,
            "error": str(e)
        }

@router.post("")
async def chat_with_papers(request: ChatRequest):
    """论文问答聊天接口 - 流式响应"""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    logger.info(f"收到聊天请求: {request.question[:100]}...")
    
    try:
        async def generate_response():
            try:
                # 构建聊天消息
                messages = [
                    ChatMessage(
                        role="system", 
                        content="你是一个专业的信息检索(Information Retrieval)领域的AI助手。你能够回答关于IR相关论文、技术和研究趋势的问题。请提供准确、详细且有帮助的回答。"
                    ),
                    ChatMessage(role="user", content=request.question)
                ]
                
                # 使用 LLM 引擎进行流式聊天
                async for content_chunk in llm_engine.chat_completion_stream(messages):
                    chunk = {
                        "choices": [{
                            "delta": {
                                "content": content_chunk
                            }
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                yield "data: [DONE]\n\n"
                logger.info("聊天响应完成")
                
            except Exception as e:
                logger.error(f"流式响应生成失败: {str(e)}")
                # 发送错误信息给前端
                error_chunk = {
                    "choices": [{
                        "delta": {
                            "content": f"\n\n❌ 抱歉，生成回答时出现错误: {str(e)}"
                        }
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
    except Exception as e:
        logger.error(f"聊天接口错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")