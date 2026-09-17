# astrbot_plugin_knowledge_base/utils/file_utils.py
import os
import tempfile
import httpx
import re
from typing import Optional
from urllib.parse import urlparse, urljoin

from astrbot.api import logger
from ..core.constants import ALLOWED_FILE_EXTENSIONS, MAX_DOWNLOAD_FILE_SIZE_MB
from .security import check_url_safety, MAX_REDIRECTS


async def download_file(url: str, destination_folder: str) -> Optional[str]:
    """
    异步下载文件到指定文件夹。
    返回下载后的文件路径，如果失败则返回 None。
    安全：下载前与每次重定向前均做 SSRF 校验（禁内网/环回/保留地址段）。
    """
    max_size_bytes = MAX_DOWNLOAD_FILE_SIZE_MB * 1024 * 1024

    # SSRF 防护：校验入口 URL
    safe, reason = check_url_safety(url)
    if not safe:
        logger.error(f"URL 安全检查未通过: {reason}. URL: {url}")
        return None

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
            current_url = url
            final_response = None
            # 手动处理重定向，每跳都做 SSRF 校验，防止重定向到内网
            for _ in range(MAX_REDIRECTS + 1):
                response = await client.get(current_url)
                if response.is_redirect or response.has_redirect_location:
                    location = response.headers.get("Location")
                    if not location:
                        final_response = response
                        break
                    next_url = urljoin(current_url, location)
                    safe, reason = check_url_safety(next_url)
                    if not safe:
                        logger.error(
                            f"重定向目标安全检查未通过: {reason}. URL: {next_url}"
                        )
                        return None
                    current_url = next_url
                    continue
                final_response = response
                break

            if final_response is None:
                logger.error(f"文件下载失败：重定向次数过多。URL: {url}")
                return None

            final_response.raise_for_status()

            parsed_url = urlparse(str(final_response.url or current_url))
            filename = os.path.basename(parsed_url.path)
            if not filename:
                content_disposition = final_response.headers.get("Content-Disposition")
                if content_disposition:
                    match = re.search(r'filename="?([^"]+)"?', content_disposition)
                    if match:
                        filename = match.group(1)
                if not filename:
                    filename = (
                        f"downloaded_file_{tempfile._RandomNameSequence().next()}"
                    )

            filename = "".join(
                c for c in filename if c.isalnum() or c in [".", "_", "-"]
            ).strip()
            if not filename:
                filename = "untitled_download"

            content_length = final_response.headers.get("Content-Length")
            if content_length and int(content_length) > max_size_bytes:
                logger.error(
                    f"文件下载失败：文件过大 ({int(content_length) / (1024 * 1024):.2f} MB > {MAX_DOWNLOAD_FILE_SIZE_MB} MB)。URL: {url}"
                )
                return None

            _, extension = os.path.splitext(filename)
            if extension.lower() not in ALLOWED_FILE_EXTENSIONS:
                logger.error(
                    f"文件下载失败：不支持的文件类型 '{extension}'. URL: {url}"
                )
                return None

            temp_file_path = os.path.join(destination_folder, filename)

            with open(temp_file_path, "wb") as f:
                downloaded_size = 0
                async for chunk in final_response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if downloaded_size > max_size_bytes:
                        f.close()
                        os.remove(temp_file_path)
                        logger.error(
                            f"文件下载失败：文件在下载过程中超出大小限制。URL: {url}"
                        )
                        return None

            logger.info(f"文件已成功下载到: {temp_file_path} 从 URL: {url}")
            return temp_file_path
    except httpx.HTTPStatusError as e:
        logger.error(
            f"文件下载 HTTP 错误: {e.response.status_code} - {e.response.text}. URL: {url}"
        )
        return None
    except Exception as e:
        logger.error(f"文件下载失败: {e}. URL: {url}", exc_info=True)
        return None
