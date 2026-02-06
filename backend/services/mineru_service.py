"""
MinerU服务 - PDF文档解析服务

该服务使用MinerU API将PDF文档解析为Markdown格式。

使用示例:
    # 初始化服务
    service = MineruService()

    # 解析PDF文件
    markdown_content = service.parse_pdf_to_markdown("path/to/document.pdf")

    # 解析PDF字节数据
    with open("document.pdf", "rb") as f:
        pdf_bytes = f.read()
    markdown_content = service.parse_pdf_bytes_to_markdown(pdf_bytes, "document.pdf")
"""

import requests
import os
from typing import Optional, Dict, Any
from pathlib import Path

class MineruService:
    def __init__(self, base_url: str = "http://192.168.30.10:8004"):
        self.base_url = base_url.rstrip('/')
        self.parse_endpoint = f"{self.base_url}/file_parse"
        self.is_available = self._test_connection()

    def _test_connection(self) -> bool:
        """
        测试MinerU服务是否可用

        Returns:
            服务是否可用
        """
        try:
            # 尝试连接服务端点，使用较短的超时时间
            response = requests.head(self.base_url, timeout=5)
            print(f"MinerU服务连接成功: {self.base_url}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"MinerU服务连接失败，将使用pdfplumber作为备选方案: {e}")
            return False

    def parse_pdf_to_markdown(self, pdf_file_path: str, lang_list: str = "ch,en") -> Optional[str]:
        """
        上传PDF文件并解析为markdown格式

        Args:
            pdf_file_path: PDF文件路径
            lang_list: 语言列表，默认为中文和英文

        Returns:
            解析后的markdown内容，如果失败则返回None
        """
        if not os.path.exists(pdf_file_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_file_path}")

        if not pdf_file_path.lower().endswith('.pdf'):
            raise ValueError("文件必须是PDF格式")

        try:
            # 准备文件
            with open(pdf_file_path, 'rb') as pdf_file:
                files = {
                    'files': (os.path.basename(pdf_file_path), pdf_file, 'application/pdf')
                }
                # 发送请求
                response = requests.post(
                    self.parse_endpoint,
                    files=files,
                    timeout=300  # 5分钟超时
                )

                response.raise_for_status()  # 检查HTTP状态码

                result = response.json()

                # 解析响应
                if 'results' in result and result['results']:
                    # 获取第一个文件的markdown内容
                    file_key = list(result['results'].keys())[0]
                    file_result = result['results'][file_key]

                    if 'md_content' in file_result:
                        return file_result['md_content']

                return None

        except requests.exceptions.RequestException as e:
            print(f"网络请求错误: {e}")
            return None
        except Exception as e:
            print(f"解析PDF时发生错误: {e}")
            return None

    def parse_pdf_bytes_to_markdown(self, pdf_bytes: bytes, filename: str = "document.pdf", lang_list: str = "ch,en") -> Optional[str]:
        """
        直接上传PDF字节数据并解析为markdown格式

        Args:
            pdf_bytes: PDF文件的字节数据
            filename: 文件名
            lang_list: 语言列表

        Returns:
            解析后的markdown内容，如果失败则返回None
        """
        try:
            # 准备文件
            files = {
                'files': (filename, pdf_bytes, 'application/pdf')
            }

            # 准备参数
            data = {
                'return_middle_json': 'false',
                'return_model_output': 'false',
                'return_md': 'true',
                'return_images': 'false',
                'end_page_id': '99999',
                'parse_method': 'auto',
                'start_page_id': '0',
                'lang_list': lang_list,
                'output_dir': './output',
                'server_url': '',
                'return_content_list': 'false',
                'backend': 'pipeline',
                'table_enable': 'true',
                'formula_enable': 'true'
            }

            # 发送请求
            response = requests.post(
                self.parse_endpoint,
                files=files,
                data=data,
                timeout=300  # 5分钟超时
            )

            response.raise_for_status()

            result = response.json()

            # 解析响应
            if 'results' in result and result['results']:
                file_key = list(result['results'].keys())[0]
                file_result = result['results'][file_key]

                if 'md_content' in file_result:
                    return file_result['md_content']

            return None

        except requests.exceptions.RequestException as e:
            print(f"网络请求错误: {e}")
            return None
        except Exception as e:
            print(f"解析PDF时发生错误: {e}")
            return None