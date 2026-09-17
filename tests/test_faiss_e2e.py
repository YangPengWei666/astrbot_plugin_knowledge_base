# -*- coding: utf-8 -*-
"""faiss 全链路自测（任务2D 自测要求：创建新库+添加+搜索）

真实执行：插件 FaissStore 逻辑 + 真实 AstrBot FaissVecDB（faiss-cpu 后端读写）
Mock 仅：外部 embedding 服务（本地无 API key，用固定假向量）

运行：python tests/test_faiss_e2e.py
"""
import asyncio
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(REPO_ROOT))
sys.path.insert(0, REPO_ROOT)

from astrbot_plugin_knowledge_base.vector_store.astrbot_faiss_store import (
    FaissStore,
)
from astrbot_plugin_knowledge_base.vector_store.base import Document

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {detail}")


class FakePrefsHandler:
    def __init__(self):
        self.user_collection_preferences = {"collection_metadata": {}}
        self._file_id_map = {}

    async def save_user_preferences(self):
        pass

    def get_collection_name_by_file_id(self, file_id):
        return self._file_id_map.get(file_id)

    def get_metadata(self, collection_name):
        return None


class FakeEmbeddingUtil:
    """模拟 EmbeddingSolutionHelper（外部 embedding 服务 mock，固定 8 维假向量）"""

    def __init__(self):
        self.user_prefs_handler = FakePrefsHandler()

    def get_rerank_provider(self, collection_name):
        return None

    def get_dimensions(self, collection_name):
        return 8

    async def get_embedding_async(self, text: str, collection_name: str):
        # 确定性假向量：文本内容 + 索引位置影响向量，使检索可区分
        base = [0.1 * (i + 1) for i in range(8)]
        # 用文本长度微调，让不同文本向量不同
        return [v + (len(text) % 5) * 0.01 * (i + 1) for i, v in enumerate(base)]

    async def get_embeddings_async(self, texts, collection_name):
        return [await self.get_embedding_async(t, collection_name) for t in texts]


async def main():
    print("== faiss 全链路自测（真实 FaissVecDB 后端） ==")
    with tempfile.TemporaryDirectory() as tmp:
        store = FaissStore(FakeEmbeddingUtil(), tmp)

        # 1. 初始化
        await store.initialize()
        check("initialize 成功", True)

        # 2. 创建新知识库（真实 create_collection 内部）
        await store.create_collection("自测库")
        check(
            "创建新知识库成功",
            "自测库" in store._all_known_collections,
            f"known={list(store._all_known_collections)}",
        )

        # 3. 添加文档（真实 add_documents → vecdb.insert → faiss 写入）
        docs = [
            Document(text_content="北京是中国的首都，位于华北平原。", metadata={"source": "e2e"}),
            Document(text_content="上海是中国的经济中心，位于长江入海口。", metadata={"source": "e2e"}),
        ]
        doc_ids = await store.add_documents("自测库", docs)
        check(
            "添加文档成功（返回 doc_id）",
            len(doc_ids) == 2,
            f"ids={doc_ids}",
        )

        # 4. 文档计数
        cnt = await store.count_documents("自测库")
        check("文档计数 = 2", cnt == 2, f"cnt={cnt}")

        # 5. 搜索（真实 retrieve + faiss 索引检索）
        results = await store.search("自测库", "北京在哪里", top_k=2)
        check(
            "搜索返回结果",
            len(results) > 0,
            f"results={len(results)}",
        )
        if results:
            check(
                "搜索结果相关（北京文档排前）",
                "北京" in results[0][0].text_content,
                f"top1={results[0][0].text_content[:20]}",
            )

        # 6. 磁盘落盘验证（faiss 索引文件真实生成）
        files = os.listdir(tmp)
        check(
            "faiss 索引/存储文件已落盘",
            any(f.endswith(".index") for f in files) and any(f.endswith(".db") for f in files),
            f"files={files}",
        )

    print(f"\n==== 结果: {PASS} 通过 / {FAIL} 失败 ====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
