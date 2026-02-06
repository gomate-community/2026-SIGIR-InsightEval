import aiohttp
import asyncio
import random
from typing import Optional, Dict, Any
from loguru import logger


class AlphaXivService:
    """AlphaXiv API 服务，用于获取论文详情"""
    
    def __init__(self):
        self.base_url = "https://api.alphaxiv.org/papers/v3/legacy"
        self.overview_base_url = "https://api.alphaxiv.org/papers/v3"
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # 模拟浏览器的请求头，避免被反爬虫机制拦截
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
            "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "Referer": "https://www.alphaxiv.org/",
            "Origin": "https://www.alphaxiv.org",
        }
        
        # 重试配置
        self.max_retries = 3
        self.retry_delay = 1  # 初始延迟（秒）
        self.request_delay_min = 0.5  # 请求之间的最小延迟（秒）
        self.request_delay_max = 2.0  # 请求之间的最大延迟（秒）
    
    async def _make_request_with_retry(
        self, 
        url: str, 
        identifier: str,
        retry_count: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        带重试机制的请求方法
        
        Args:
            url: 请求的 URL
            identifier: 用于日志的标识符（如 paper_id 或 paper_version_id）
            retry_count: 当前重试次数
            
        Returns:
            响应数据，如果失败返回 None
        """
        # 请求之间的随机延迟，模拟人类行为
        if retry_count == 0:
            delay = random.uniform(self.request_delay_min, self.request_delay_max)
            await asyncio.sleep(delay)
        else:
            # 重试时使用指数退避
            delay = self.retry_delay * (2 ** (retry_count - 1))
            await asyncio.sleep(delay)
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully fetched data for {identifier}")
                        return data
                    elif response.status == 500:
                        # 500 错误时重试
                        if retry_count < self.max_retries:
                            logger.warning(
                                f"Received 500 error for {identifier}, "
                                f"retrying ({retry_count + 1}/{self.max_retries})..."
                            )
                            return await self._make_request_with_retry(
                                url, identifier, retry_count + 1
                            )
                        else:
                            logger.error(
                                f"Failed to fetch {identifier} after {self.max_retries} retries: "
                                f"HTTP {response.status}"
                            )
                            return None
                    elif response.status == 429:
                        # 速率限制，等待更长时间后重试
                        if retry_count < self.max_retries:
                            wait_time = delay * 2
                            logger.warning(
                                f"Rate limited for {identifier}, waiting {wait_time}s "
                                f"before retry ({retry_count + 1}/{self.max_retries})..."
                            )
                            await asyncio.sleep(wait_time)
                            return await self._make_request_with_retry(
                                url, identifier, retry_count + 1
                            )
                        else:
                            logger.error(
                                f"Rate limited for {identifier} after {self.max_retries} retries"
                            )
                            return None
                    else:
                        logger.error(
                            f"Failed to fetch {identifier}: HTTP {response.status}"
                        )
                        return None
                        
        except asyncio.TimeoutError:
            if retry_count < self.max_retries:
                logger.warning(
                    f"Timeout when fetching {identifier}, "
                    f"retrying ({retry_count + 1}/{self.max_retries})..."
                )
                return await self._make_request_with_retry(
                    url, identifier, retry_count + 1
                )
            else:
                logger.error(f"Timeout when fetching {identifier} after {self.max_retries} retries")
                return None
        except aiohttp.ClientError as e:
            if retry_count < self.max_retries:
                logger.warning(
                    f"Client error when fetching {identifier}: {e}, "
                    f"retrying ({retry_count + 1}/{self.max_retries})..."
                )
                return await self._make_request_with_retry(
                    url, identifier, retry_count + 1
                )
            else:
                logger.error(
                    f"Client error when fetching {identifier} after {self.max_retries} retries: {e}"
                )
                return None
        except Exception as e:
            logger.error(f"Unexpected error when fetching {identifier}: {e}")
            return None
    
    async def get_paper_alphaxiv_detail(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 paper_id 获取论文详情
        
        Args:
            paper_id: 论文ID (如: 2510.18433)
            
        Returns:
            论文详情数据，如果失败返回 None
        """
        url = f"{self.base_url}/{paper_id}"
        return await self._make_request_with_retry(url, paper_id)
    
    async def get_paper_overview(self, paper_version_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 paper_version_id 获取论文概览
        
        Args:
            paper_version_id: 论文版本ID (如: 019a04b0-40de-7c72-89a7-a7ff2bc29ab0)
            
        Returns:
            论文概览数据，如果失败返回 None
        """
        url = f"{self.overview_base_url}/{paper_version_id}/overviews"
        overview = await self._make_request_with_retry(url, paper_version_id, retry_count=2)
        
        # 过滤 overviews，只保留中文（zh）和英文（en）
        if overview and isinstance(overview, dict) and "overviews" in overview:
            original_overviews = overview.get("overviews", {})
            if isinstance(original_overviews, dict):
                # 只保留 zh 和 en
                filtered_overviews = {
                    lang: content 
                    for lang, content in original_overviews.items() 
                    if lang in ["zh", "en"]
                }
                overview["overviews"] = filtered_overviews
        
        return overview
    
    def format_paper_detail(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        格式化论文详情数据
        
        Args:
            raw_data: 原始 API 响应数据
            
        Returns:
            格式化后的论文详情
        """
        try:
            paper = raw_data.get("paper", {})
            paper_version = paper.get("paper_version", {})
            paper_group = paper.get("paper_group", {})
            authors = paper.get("authors", [])
            pdf_info = paper.get("pdf_info", {})
            
            # 格式化作者信息
            author_names = [author.get("full_name", "") for author in authors if author.get("full_name")]
            
            # 格式化日期
            publication_date = paper_version.get("publication_date", "")
            if publication_date:
                try:
                    from datetime import datetime
                    date_obj = datetime.fromisoformat(publication_date.replace("GMT+0000 (Coordinated Universal Time)", "").strip())
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                except:
                    formatted_date = publication_date[:10] if len(publication_date) >= 10 else publication_date
            else:
                formatted_date = ""
            
            # 提取主题和分类
            topics = paper_group.get("topics", [])
            primary_category = topics[0] if topics else "Unknown"
            
            # 获取指标数据
            metrics = paper_group.get("metrics", {}) or {}
            visits = metrics.get("visits_count", {}) or {}
            
            # 获取资源链接
            resources = paper_group.get("resources", {}) or {}
            github_info = resources.get("github", {}) or {}
            
            # 获取引用信息
            citation = paper_group.get("citation", {}) or {}
            
            formatted_data = {
                "id": paper_version.get("id", ""),# paper_version_id
                "universal_paper_id": paper_version.get("universal_paper_id", ""),
                "title": paper_version.get("title", ""),
                "abstract": paper_version.get("abstract", ""),
                "authors": author_names,
                "publication_date": formatted_date,
                "license": paper_version.get("license", ""),
                "topics": topics,
                "primary_category": primary_category,
                "arxiv_url": (paper_group.get("source") or {}).get("url", ""),
                "pdf_url": pdf_info.get("fetcher_url", ""),
                "image_url": paper_version.get("imageURL", ""),
                "metrics": {
                    "upvotes": metrics.get("upvotes_count", 0),
                    "downvotes": metrics.get("downvotes_count", 0),
                    "total_votes": metrics.get("total_votes", 0),
                    "visits_24h": visits.get("last24Hours", 0),
                    "visits_7d": visits.get("last7Days", 0),
                    "visits_30d": visits.get("last30Days", 0),
                    "visits_all": visits.get("all", 0)
                },
                "github": {
                    "url": github_info.get("url", "") if github_info else "",
                    "language": github_info.get("language", "") if github_info else "",
                    "stars": github_info.get("stars", 0) if github_info else 0
                } if github_info and github_info.get("url") else None,
                "citation": {
                    "bibtex": citation.get("bibtex", "") if citation else ""
                },
                "comments": raw_data.get("comments", [])
            }
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error formatting paper detail: {e}")
            return None