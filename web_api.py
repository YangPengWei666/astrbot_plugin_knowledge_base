import os
import time
import uuid
from astrbot.api.star import Context
from .vector_store.base import VectorDBBase, Document
from quart import request
try:
    # AstrBot >= 4.0：responses 模块级函数
    from astrbot.dashboard.responses import ok as _resp_ok, error as _resp_error

    def _resp_ok_compat(data=None, message=None):
        return _resp_ok(data=data, message=message)

    def _resp_error_compat(message: str, data=None):
        return _resp_error(message, data=data)
except ImportError:
    # AstrBot 3.5.x：Response 类（.ok()/.error() 返回对象， 转 dict）
    try:
        from astrbot.dashboard.routes.route import Response as _LegacyResponse
    except ImportError:
        from astrbot.dashboard.server import Response as _LegacyResponse

    def _resp_ok_compat(data=None, message=None):
        return _Legacy_resp_ok_compat(data=data, message=message)

    def _resp_error_compat(message: str, data=None):
        return _Legacy_resp_error_compat(message)
from .utils.text_splitter import TextSplitterUtil
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from .utils.file_parser import FileParser, LLM_Config
from astrbot.api import logger
from astrbot.api import AstrBotConfig
from astrbot.core.config.default import VERSION
from .core.user_prefs_handler import UserPrefsHandler


class KnowledgeBaseWebAPI:
    def __init__(
        self,
        vec_db: VectorDBBase,
        text_splitter: TextSplitterUtil,
        astrbot_context: Context,
        llm_config: LLM_Config,
        user_prefs_handler: UserPrefsHandler = None,
        plugin_config: AstrBotConfig = None,
    ):
        self.vec_db = vec_db
        self.text_splitter = text_splitter
        self.astrbot_context = astrbot_context
        self.user_prefs_handler = user_prefs_handler
        self.plugin_config = plugin_config

        # 注册 Web API 端点
        self.astrbot_context.register_web_api(
            "/alkaid/kb/create_collection",
            self.create_collection,
            ["POST"],
            "创建一个新的知识库集合",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collections",
            self.list_collections,
            ["GET"],
            "列出所有知识库集合",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/add_file",
            self.add_documents,
            ["POST"],
            "向指定集合添加文档",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/search",
            self.search_documents,
            ["GET"],
            "搜索指定集合中的文档",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/delete",
            self.delete_collection,
            ["GET"],
            "删除指定集合",
        )
        self.fp = FileParser(llm_config=llm_config)

    async def test_embedding_provider(self, collection_name: str):
        res = await self.vec_db.embedding_util.get_embedding_async(
            text="test", collection_name=collection_name
        )
        real_dim = len(res)
        dim = self.vec_db.embedding_util.get_dimensions(collection_name=collection_name)
        if real_dim != dim:
            raise ValueError(
                f"嵌入模型提供商配置中的嵌入维度有误，填写为 {dim}，实际为 {real_dim}，请前往修改。"
            )

    async def create_collection(self):
        """
        创建一个新的知识库集合。
        :param collection_name: 集合名称
        :return: 创建结果
        """
        data = await request.get_json()
        collection_name = data.get("collection_name")
        emoji = data.get("emoji", "🙂")
        description = data.get("description", "")
        embedding_provider_id = data.get("embedding_provider_id", None)
        if not collection_name:
            return _resp_error_compat("缺少集合名称")
        if await self.vec_db.collection_exists(collection_name):
            return _resp_error_compat("集合已存在")
        if not embedding_provider_id:
            return _resp_error_compat("缺少嵌入提供商 ID")
        try:
            # 添加集合元数据
            metadata = {
                "version": 1,  # metadata 配置版本
                "emoji": emoji,
                "description": description,
                "created_at": int(time.time()),
                "file_id": f"KBDB_{str(uuid.uuid4())}",  # 文件 ID
                "origin": "astrbot-webui",
                "embedding_provider_id": embedding_provider_id,  # AstrBot 嵌入提供商 ID
                "rerank_provider_id": data.get("rerank_provider_id", None),
            }
            collection_metadata = (
                self.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                )
            )
            collection_metadata[collection_name] = metadata
            self.user_prefs_handler.user_collection_preferences[
                "collection_metadata"
            ] = collection_metadata
            await self.user_prefs_handler.save_user_preferences()
            # 兼容性问题，create_collection 方法放在上一步之后执行。

            # test provider dim
            try:
                await self.test_embedding_provider(collection_name)
            except Exception as e:
                # delete
                collection_metadata.pop(collection_name, None)
                self.user_prefs_handler.user_collection_preferences[
                    "collection_metadata"
                ] = collection_metadata
                await self.user_prefs_handler.save_user_preferences()
                return _resp_error_compat(str(e))

            await self.vec_db.create_collection(collection_name)
            return _resp_ok_compat(message="集合创建成功")
        except Exception as e:
            return _resp_error_compat(f"创建集合失败: {str(e)}")

    async def list_collections(self):
        """
        列出所有知识库集合。
        :return: 集合列表
        """
        try:
            collections = await self.vec_db.list_collections()
            result = []
            collections_metadata = (
                self.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                )
            )
            for collection in collections:
                collection_md = collections_metadata.get(collection, {})
                if "embedding_provider_id" in collection_md:
                    p_id = collection_md.get("embedding_provider_id", "")
                    provider = self.astrbot_context.get_provider_by_id(p_id)
                    if provider:
                        collection_md["_embedding_provider_config"] = (
                            provider.provider_config
                        )
                count = await self.vec_db.count_documents(collection)
                result.append(
                    {"collection_name": collection, "count": count, **collection_md}
                )
            return _resp_ok_compat(data=result)
        except Exception as e:
            return _resp_error_compat(f"获取集合列表失败: {str(e)}")

    async def add_documents(self):
        """
        向指定集合添加文档。
        :param collection_name: 集合名称
        :param documents: 文档列表
        :return: 添加结果
        """
        upload_file = (await request.files).get("file")
        collection_name = (await request.form).get("collection_name")
        chunk_size = (await request.form).get("chunk_size", None)
        overlap = (await request.form).get("chunk_overlap", None)
        if not upload_file or not collection_name:
            return _resp_error_compat("缺少知识库名称")
        if not await self.vec_db.collection_exists(collection_name):
            return _resp_error_compat("目标知识库不存在")

        try:
            chunk_size = int(chunk_size) if chunk_size else None
            overlap = int(overlap) if overlap else None
            path = os.path.join(get_astrbot_data_path(), "temp", upload_file.filename)
            await upload_file.save(path)
            content = await self.fp.parse_file_content(path)
            if not content:
                raise ValueError("文件内容为空或不支持的格式")

            chunks = self.text_splitter.split_text(
                text=content, chunk_size=chunk_size, overlap=overlap
            )
            if not chunks:
                raise Exception("chunk 内容为空")

            documents_to_add = [
                Document(
                    text_content=chunk,
                    metadata={
                        "source": upload_file.filename,
                        "user": "astrbot_webui",
                    },
                )
                for chunk in chunks
            ]

            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"删除临时文件失败: {str(e)}")

            try:
                doc_ids = await self.vec_db.add_documents(
                    collection_name, documents_to_add
                )
                if not doc_ids:
                    raise Exception("添加文档失败，返回的文档 ID 为空")
                return _resp_ok_compat(
                    data=doc_ids,
                    message=f"成功从文件 '{upload_file.filename}' 添加 {len(doc_ids)} 条知识到 '{collection_name}'。",
                )
            except Exception as e:
                raise Exception(f"添加文档失败: {str(e)}。")

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            if os.path.exists(path):
                os.remove(path)
            return _resp_error_compat(f"添加文档失败: {str(e)}")

    async def search_documents(self):
        """
        搜索指定集合中的文档。
        :param collection_name: 集合名称
        :param query: 查询字符串
        :param top_k: 返回结果数量，默认为5
        :return: 搜索结果
        """
        # 从 URL 参数中获取查询参数
        collection_name = request.args.get("collection_name")
        query = request.args.get("query")
        try:
            top_k = int(request.args.get("top_k", 5))
        except ValueError:
            top_k = 5

        # 验证必要参数
        if not collection_name or not query:
            return _resp_error_compat("缺少集合名称或查询字符串")

        # 检查知识库是否存在
        if not await self.vec_db.collection_exists(collection_name):
            return _resp_error_compat("目标知识库不存在")

        try:
            # 执行搜索
            results = await self.vec_db.search(collection_name, query, top_k)

            # 格式化结果以便前端展示
            formatted_results = []
            for i, doc in enumerate(results):
                doc, score = doc
                formatted_results.append(
                    {
                        "id": doc.id,
                        "content": doc.text_content,
                        "metadata": doc.metadata,
                        "score": score,
                    }
                )
            return _resp_ok_compat(data=formatted_results)
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return _resp_error_compat(f"搜索失败: {str(e)}")

    async def delete_collection(self):
        """
        删除指定集合。
        :param collection_name: 集合名称
        """
        # 从 URL 参数中获取查询参数
        collection_name = request.args.get("collection_name")

        # 检查知识库是否存在
        if not await self.vec_db.collection_exists(collection_name):
            return _resp_error_compat("目标知识库不存在")

        try:
            # 执行删除
            await self.vec_db.delete_collection(collection_name)
            return _resp_ok_compat(f"删除 {collection_name} 成功")
        except Exception as e:
            logger.error(f"删除失败: {str(e)}")
            return _resp_error_compat(f"删除失败: {str(e)}")
