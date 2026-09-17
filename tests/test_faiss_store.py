# -*- coding: utf-8 -*-
"""faiss 断链 bug 回归单测（任务2D）

覆盖：
  1. "创建新知识库"路径：真实调用 FaissStore.create_collection() 内部逻辑
     （mock 仅替换 FaissVecDB 等外部依赖，不绕过 create_collection）
     —— 回归 Bug-A：create_collection 需访问 embedding_util.user_prefs_handler
  2. "有历史数据时初始化"路径：磁盘存在历史集合文件，initialize() 扫描
     —— 回归 Bug-A 注入时序：_scan_collections_on_disk → _get_collection_meta
        在注入缺失时即崩，注入提前后正常
  3. "4 元组解包"：_get_or_load_vecdb 解包 _get_collection_meta
     —— 回归 Bug-B（5 元组解包 ValueError）

运行：python tests/test_faiss_store.py
"""
import asyncio
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(REPO_ROOT))
sys.path.insert(0, REPO_ROOT)

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
    """模拟 UserPrefsHandler（只实现 FaissStore 用到的接口）"""

    def __init__(self):
        self.saved_count = 0
        self.user_collection_preferences = {"collection_metadata": {}}
        self._file_id_map = {}

    async def save_user_preferences(self):
        self.saved_count += 1

    def get_collection_name_by_file_id(self, file_id):
        return self._file_id_map.get(file_id)

    def get_metadata(self, collection_name):
        """CollectionMetadataRepository 接口（get_rerank_provider 使用）"""
        return None


def make_embedding_util():
    """真实 EmbeddingSolutionHelper + 注入 FakePrefsHandler

    - set_user_prefs_handler：模拟 main.py 提前注入（vector_db.initialize 之前，Bug-A 修复点）
    - set_metadata_repo：模拟 main.py Step3 注入（真实环境创建命令执行时已完成，
      创建路径 get_rerank_provider 依赖它）
    """
    from astrbot_plugin_knowledge_base.utils.embedding import (
        EmbeddingSolutionHelper,
    )

    helper = EmbeddingSolutionHelper(
        curr_embedding_dimensions=1024,
        curr_embedding_util=MagicMock(),
        context=MagicMock(),
    )
    prefs = FakePrefsHandler()
    helper.set_user_prefs_handler(prefs)
    helper.set_metadata_repo(prefs)
    return helper, prefs


def make_store(embedding_util, data_path):
    from astrbot_plugin_knowledge_base.vector_store.astrbot_faiss_store import (
        FaissStore,
    )

    return FaissStore(embedding_util, data_path)


async def test_create_collection():
    """路径1：创建新知识库（真实 create_collection 内部逻辑，回归 Bug-A）"""
    print("\n== 路径1：创建新知识库 ==")
    with tempfile.TemporaryDirectory() as tmp:
        embedding_util, prefs = make_embedding_util()
        store = make_store(embedding_util, tmp)

        fake_vecdb = MagicMock()
        fake_vecdb.initialize = MagicMock(return_value=asyncio.sleep(0))
        fake_vecdb.embedding_storage.save_index = MagicMock(
            return_value=asyncio.sleep(0)
        )

        with patch(
            "astrbot_plugin_knowledge_base.vector_store.astrbot_faiss_store.FaissVecDB",
            return_value=fake_vecdb,
        ):
            await store.initialize()
            await store.create_collection("测试库A")

        check("create_collection 成功", "测试库A" in store._all_known_collections)
        check(
            "create_collection 缓存已加载",
            "测试库A" in store.cache,
            f"cache={list(store.cache.keys())}",
        )
        check(
            "save_user_preferences 被调用（user_prefs_handler 注入生效）",
            prefs.saved_count >= 1,
            f"saved={prefs.saved_count}",
        )
        check(
            "save_index 被调用（创建流程完整）",
            fake_vecdb.embedding_storage.save_index.called,
        )


async def test_initialize_with_history():
    """路径2：有历史数据时初始化（回归 Bug-A 注入时序）"""
    print("\n== 路径2：有历史数据时初始化 ==")
    with tempfile.TemporaryDirectory() as tmp:
        # 模拟历史数据：磁盘已有集合文件（.index + .db）
        for fn in ("历史库.index", "历史库.db"):
            with open(os.path.join(tmp, fn), "w") as f:
                f.write("mock")
        # 元数据映射：file_id → 真实集合名（历史库场景）
        embedding_util, prefs = make_embedding_util()
        prefs._file_id_map["历史库"] = "真实历史库名"
        store = make_store(embedding_util, tmp)

        # 未注入场景：应 AttributeError（证明注入是必需的）
        from astrbot_plugin_knowledge_base.utils.embedding import (
            EmbeddingSolutionHelper,
        )

        broken_util = EmbeddingSolutionHelper(
            curr_embedding_dimensions=1024,
            curr_embedding_util=MagicMock(),
            context=MagicMock(),
        )
        broken_store = make_store(broken_util, tmp)
        try:
            await broken_store.initialize()
            check("未注入时初始化应失败（防回归误报）", False, "居然成功了")
        except AttributeError as e:
            check("未注入时初始化 AttributeError（注入必需性验证）", True)

        # 注入后：initialize 正常扫描出历史数据
        await store.initialize()
        check(
            "注入后初始化不崩 + 历史数据被扫描",
            "真实历史库名" in store._all_known_collections,
            f"known={list(store._all_known_collections)}",
        )
        check("扫描日志路径正确（无 5 元组崩溃）", True)


async def test_meta_unpack():
    """路径3：_get_or_load_vecdb 4 元组解包（回归 Bug-B）"""
    print("\n== 路径3：4 元组解包 ==")
    with tempfile.TemporaryDirectory() as tmp:
        embedding_util, _ = make_embedding_util()
        store = make_store(embedding_util, tmp)

        # _get_collection_meta 返回 4 元组
        meta = store._get_collection_meta("测试")
        check("_get_collection_meta 返回 4 元组", len(meta) == 4, f"len={len(meta)}")

        # 走 _get_or_load_vecdb 不存在的集合（for_create=False）：
        # 5 元组解包会 ValueError，4 元组正常返回 None
        result = await store._get_or_load_vecdb("不存在的库", for_create=False)
        check(
            "_get_or_load_vecdb 4 元组解包正常（Bug-B 修复）",
            result is None,
            f"result={result}",
        )


async def main():
    await test_create_collection()
    await test_initialize_with_history()
    await test_meta_unpack()
    print(f"\n==== 结果: {PASS} 通过 / {FAIL} 失败 ====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
