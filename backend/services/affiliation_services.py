
import asyncio
import json
import re
from io import BytesIO
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
import time
import requests
import pdfplumber
from backend.prompts.affiliation_prompt import AFFILIATION_EXTRACTION_PROMPT
from backend.engines.llm_engine import LLMEngine, ChatMessage, LLMConfig, OneAINexusConfig
from backend.engines.minio_engine import MinioEngine
from backend.services.mineru_service import MineruService
from urllib.parse import urlparse
import hashlib
class AffiliationService:
    """论文作者机构抽取服务，使用LLM从PDF中提取作者、机构、邮箱等结构化信息，并自动上传PDF到MinIO进行存储管理"""

    def __init__(self, llm_config: Optional[LLMConfig] = None, mineru_base_url: str = "http://192.168.30.10:8004"):
        """
        初始化机构抽取服务

        Args:
            llm_config: LLM配置，如果为None则使用默认OneAINexus配置
            mineru_base_url: MinerU服务的基础URL
        """
        self.llm_engine = LLMEngine(llm_config)
        self.minio_service = MinioEngine()
        self.mineru_service = MineruService(base_url=mineru_base_url)

    async def extract_affiliations_from_pdf(self, pdf_url: str) -> List[Dict[str, Any]]:
        """
        从PDF URL中提取作者机构信息

        Args:
            pdf_url: PDF文件的URL

        Returns:
            作者机构信息字典，包含affiliations列表（每个元素包含author, author_email, org, org_email, is_industry）
        """
        try:
            # 下载并提取PDF文本
            download_result = await self._download_and_extract_text(pdf_url)
            first_page_text = download_result["first_page_text"]
            # markdown_content = download_result["markdown_content"]
            pdf_minio_info = download_result.get("pdf_minio_info")
            markdown_minio_info = download_result.get("markdown_minio_info")

            # 使用LLM进行关系抽取
            affiliations = await self._extract_affiliations_with_llm(first_page_text)

            # 添加元数据
            result = {
                "affiliations": affiliations,
                "timestamp": datetime.now().isoformat(),
                "pdf_url": pdf_url,
                "text_length": len(first_page_text),
                # "markdown_content": markdown_content,
                "pdf_minio_info": pdf_minio_info,
                "markdown_minio_info": markdown_minio_info
            }

            logger.info(f"成功从PDF中提取了 {len(affiliations)} 个作者机构信息")
            return result

        except Exception as e:
            logger.error(f"PDF机构抽取失败: {str(e)}")
            # 返回空结果而不是抛出异常
            return {
                "affiliations": [],
                "timestamp": datetime.now().isoformat(),
                "pdf_url": pdf_url,
                "error": str(e),
                "fallback": True
            }

    async def _download_and_extract_text(self, pdf_url: str) -> Dict[str, Any]:
        """
        下载PDF并使用MinerU解析为markdown，同时上传PDF和markdown到MinIO

        Args:
            pdf_url: PDF URL

        Returns:
            包含markdown内容、PDF和markdown MinIO信息的字典
        """
        import tempfile
        import os

        temp_file_path = None
        pdf_minio_info = None
        markdown_minio_info = None

        try:
            # 下载PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            time.sleep(2)
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            # 上传PDF到MinIO
            pdf_filename = self._extract_filename_from_url(pdf_url)
            if '.pdf' not in pdf_filename:
                pdf_filename = pdf_filename+'.pdf'
            minio_result = await self.minio_service.upload_file(
                file_path=temp_file_path,
                object_name=pdf_filename,
                content_type="application/pdf"
            )

            if minio_result["success"]:
                pdf_minio_info = {
                    "bucket": minio_result["bucket"],
                    "object_name": minio_result["object_name"],
                    "uploaded_at": minio_result["uploaded_at"],
                    "size": minio_result["size"]
                }
                logger.info(f"PDF已上传到MinIO: {minio_result['bucket']}/{minio_result['object_name']}")
            else:
                logger.warning(f"PDF上传到MinIO失败: {minio_result.get('error', '未知错误')}")

            # 使用MinerU解析PDF为markdown，如果服务不可用则使用pdfplumber作为备选
            if self.mineru_service.is_available:
                logger.info(f"开始使用MinerU解析PDF: {pdf_filename}")
                markdown_content = self.mineru_service.parse_pdf_to_markdown(temp_file_path)
            else:
                markdown_content = None

            if not markdown_content:
                # MinerU服务不可用或解析失败，使用pdfplumber读取全文内容
                logger.info(f"使用pdfplumber读取PDF全文内容: {pdf_filename}")
                markdown_content = self._extract_full_text_with_pdfplumber(temp_file_path)

            # 将markdown内容上传到MinIO
            markdown_filename = pdf_filename.replace('.pdf', '.md')
            markdown_bytes = markdown_content.encode('utf-8')

            markdown_minio_result = await self.minio_service.upload_bytes(
                data=markdown_bytes,
                object_name=markdown_filename,
                bucket_name="md-files",
                content_type="text/markdown"
            )

            if markdown_minio_result["success"]:
                markdown_minio_info = {
                    "bucket": markdown_minio_result["bucket"],
                    "object_name": markdown_minio_result["object_name"],
                    "uploaded_at": markdown_minio_result["uploaded_at"],
                    "size": markdown_minio_result["size"]
                }
                logger.info(f"Markdown已上传到MinIO: {markdown_minio_result['bucket']}/{markdown_minio_result['object_name']}")
            else:
                logger.warning(f"Markdown上传到MinIO失败: {markdown_minio_result.get('error', '未知错误')}")

            # 提取第一页文本用于机构抽取（向后兼容）
            import pdfplumber
            with pdfplumber.open(temp_file_path) as pdf:
                if len(pdf.pages) == 0:
                    raise ValueError("PDF文件没有页面")

                # 提取第一页文本，通常包含标题、作者和机构信息
                first_page = pdf.pages[0]
                first_page_text = first_page.extract_text()

                if not first_page_text or len(first_page_text.strip()) < 100:
                    logger.warning("PDF第一页文本过少，可能不是有效的学术论文")

                return {
                    "first_page_text": first_page_text,
                    # "markdown_content": markdown_content,
                    "pdf_minio_info": pdf_minio_info,
                    "markdown_minio_info": markdown_minio_info,
                    "temp_file_path": temp_file_path
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"PDF下载失败: {str(e)}")
            raise ValueError(f"无法下载PDF文件: {str(e)}")
        except Exception as e:
            logger.error(f"PDF解析失败: {str(e)}")
            raise ValueError(f"PDF解析失败: {str(e)}")
        finally:
            # 清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"清理临时文件: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {str(e)}")

    def _extract_full_text_with_pdfplumber(self, pdf_file_path: str) -> str:
        """
        使用pdfplumber读取PDF全文内容作为markdown的替代

        Args:
            pdf_file_path: PDF文件路径

        Returns:
            PDF全文内容
        """
        try:
            full_text = []
            with pdfplumber.open(pdf_file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(f"# Page {page_num}\n\n{page_text}")
            
            content = "\n\n".join(full_text)
            logger.info(f"使用pdfplumber成功提取PDF全文，共{len(pdf.pages)}页")
            return content
        except Exception as e:
            logger.error(f"pdfplumber读取PDF失败: {e}")
            return ""

    def _extract_filename_from_url(self, url: str) -> str:
        """
        从URL中提取文件名

        Args:
            url: PDF URL

        Returns:
            提取的文件名
        """


        # 解析URL获取路径部分
        parsed = urlparse(url)
        path = parsed.path

        # 从路径中提取文件名
        filename = path.split('/')[-1] if '/' in path else 'unknown.pdf'

        # 如果没有有效的文件名，使用URL的哈希值
        if not filename or filename == '' or '.' not in filename:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"paper_{url_hash}.pdf"

        return filename

    async def cleanup_pdf_from_minio(self, minio_info: Optional[Dict[str, Any]]) -> bool:
        """
        从MinIO中删除临时PDF文件

        Args:
            minio_info: MinIO信息字典

        Returns:
            删除是否成功
        """
        if not minio_info:
            return True

        try:
            bucket = minio_info.get("bucket")
            object_name = minio_info.get("object_name")

            if bucket and object_name:
                result = await self.minio_service.remove_object(
                    object_name=object_name,
                    bucket_name=bucket
                )

                if result["success"]:
                    logger.info(f"成功从MinIO删除临时PDF: {bucket}/{object_name}")
                    return True
                else:
                    logger.warning(f"从MinIO删除临时PDF失败: {result.get('error', '未知错误')}")
                    return False
            else:
                logger.warning("MinIO信息不完整，无法删除")
                return False

        except Exception as e:
            logger.error(f"清理MinIO文件异常: {str(e)}")
            return False

    async def _extract_affiliations_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        使用LLM进行作者机构关系抽取

        Args:
            text: PDF文本内容

        Returns:
            作者机构信息列表，每个元素包含author, author_email, org, org_email, is_industry
        """
        try:
            # 构建提示词
            prompt = AFFILIATION_EXTRACTION_PROMPT.format(text=text)
            # 调用LLM
            response = await self.llm_engine.simple_chat(
                prompt=prompt,
                system_prompt="你是一位专业的学术论文分析助手，擅长从论文文本中提取作者和机构信息。"
            )
            # 解析响应
            affiliations = self._parse_affiliation_response(response)

            return affiliations

        except Exception as e:
            logger.error(f"LLM机构抽取失败: {str(e)}")
            return []

    def _parse_affiliation_response(self, response: str) -> List[Dict[str, Any]]:
        """
        解析LLM的机构抽取响应

        Args:
            response: LLM响应文本

        Returns:
            解析后的机构信息列表
        """
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                affiliations = json.loads(json_str)

                # 验证和清理数据
                validated_affiliations = []
                for item in affiliations:
                    if isinstance(item, dict) and self._validate_affiliation_item(item):
                        validated_affiliations.append(item)

                return validated_affiliations

            # 如果解析失败，尝试从文本中提取
            return self._extract_affiliations_from_text(response)

        except Exception as e:
            logger.error(f"解析机构响应失败: {str(e)}")
            return []

    def _validate_affiliation_item(self, item: Dict[str, Any]) -> bool:
        """
        验证机构信息项是否有效

        Args:
            item: 机构信息字典

        Returns:
            是否有效
        """
        required_fields = ["author", "author_email", "org", "org_email", "is_industry"]

        # 检查必需字段
        for field in required_fields:
            if field not in item:
                return False

        # 验证数据类型
        if not isinstance(item["author"], str) or not item["author"].strip():
            return False

        if not isinstance(item["author_email"], str):
            return False

        if not isinstance(item["org"], str) or not item["org"].strip():
            return False

        if not isinstance(item["org_email"], str):
            return False

        if not isinstance(item["is_industry"], bool):
            return False

        return True

    def _extract_affiliations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取机构信息（降级方案）

        Args:
            text: 响应文本

        Returns:
            提取的机构信息列表
        """
        # 这是一个简单的降级方案，实际项目中可能需要更复杂的逻辑
        affiliations = []

        # 尝试匹配作者-机构模式
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 'author' in line.lower() and 'org' in line.lower():
                # 简单的文本解析，实际使用中可能需要更复杂的方法
                pass

        return affiliations

    async def batch_extract_affiliations(self, pdf_urls: List[str]) -> List[Dict[str, Any]]:
        """
        批量提取多个PDF的机构信息

        Args:
            pdf_urls: PDF URL列表

        Returns:
            机构信息结果列表
        """
        results = []

        for pdf_url in pdf_urls:
            try:
                result = await self.extract_affiliations_from_pdf(pdf_url)
                results.append(result)

                # 添加延迟避免请求过快
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"批量处理PDF失败 {pdf_url}: {str(e)}")
                results.append({
                    "affiliations": [],
                    "timestamp": datetime.now().isoformat(),
                    "pdf_url": pdf_url,
                    "error": str(e),
                    "fallback": True
                })

        return results

    async def cleanup_processed_pdfs(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        清理已处理的PDF文件从MinIO

        Args:
            results: 处理结果列表

        Returns:
            清理统计信息
        """
        cleaned_count = 0
        failed_count = 0

        for result in results:
            minio_info = result.get("minio_info")
            if minio_info:
                success = await self.cleanup_pdf_from_minio(minio_info)
                if success:
                    cleaned_count += 1
                else:
                    failed_count += 1

        logger.info(f"PDF清理完成: 成功{cleaned_count}个，失败{failed_count}个")

        return {
            "total_processed": len(results),
            "cleaned_count": cleaned_count,
            "failed_count": failed_count,
            "timestamp": datetime.now().isoformat()
        }

    def get_affiliation_stats(self, affiliations_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        统计机构信息数据

        Args:
            affiliations_data: 机构数据列表

        Returns:
            统计信息
        """
        total_authors = 0
        industry_count = 0
        academic_count = 0
        unique_orgs = set()
        authors_with_email = 0
        orgs_with_email = 0

        for data in affiliations_data:
            affiliations = data.get("affiliations", [])
            total_authors += len(affiliations)

            for affiliation in affiliations:
                unique_orgs.add(affiliation.get("org", ""))
                if affiliation.get("is_industry", False):
                    industry_count += 1
                else:
                    academic_count += 1

                # 统计邮箱信息
                if affiliation.get("author_email", "").strip():
                    authors_with_email += 1
                if affiliation.get("org_email", "").strip():
                    orgs_with_email += 1

        return {
            "total_authors": total_authors,
            "industry_authors": industry_count,
            "academic_authors": academic_count,
            "unique_organizations": len(unique_orgs),
            "industry_ratio": industry_count / total_authors if total_authors > 0 else 0,
            "authors_with_email": authors_with_email,
            "orgs_with_email": orgs_with_email,
            "email_coverage": {
                "author_email_ratio": authors_with_email / total_authors if total_authors > 0 else 0,
                "org_email_ratio": orgs_with_email / len(unique_orgs) if unique_orgs else 0
            }
        }

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            服务是否正常
        """
        try:
            # 检查LLM引擎
            llm_healthy = await self.llm_engine.health_check()

            # 检查MinIO服务
            minio_healthy = await self.minio_service.health_check()

            if not llm_healthy:
                logger.error("LLM引擎健康检查失败")
            if not minio_healthy:
                logger.error("MinIO服务健康检查失败")

            return llm_healthy and minio_healthy
        except Exception as e:
            logger.error(f"机构抽取服务健康检查失败: {str(e)}")
            return False


# 示例使用
if __name__ == "__main__":
    async def main():
        service = AffiliationService()

        # 健康检查
        if await service.health_check():
            logger.info("✅ 机构抽取服务正常")
        else:
            logger.error("❌ 机构抽取服务异常")
            return

        # 检查MinIO存储桶
        try:
            buckets = service.minio_service.list_buckets()
            logger.info(f"✅ MinIO连接正常，发现 {len(buckets)} 个存储桶")
            for bucket in buckets:
                logger.info(f"  - {bucket['name']}")
        except Exception as e:
            logger.warning(f"⚠️ MinIO存储桶检查失败: {str(e)}")

        # 测试机构抽取
        test_pdf_url = "https://arxiv.org/pdf/2304.10864.pdf"

        result = await service.extract_affiliations_from_pdf(test_pdf_url)

        logger.info("机构抽取结果:")
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))

        # MinIO信息
        if result.get("minio_info"):
            logger.info(f"PDF已存储到MinIO: {result['minio_info']['bucket']}/{result['minio_info']['object_name']}")

        # 统计信息
        if result["affiliations"]:
            stats = service.get_affiliation_stats([result])
            logger.info(f"统计信息: {stats}")

        # 可选：清理MinIO中的PDF文件
        # await service.cleanup_pdf_from_minio(result.get("minio_info"))

    # 运行示例
    asyncio.run(main())
