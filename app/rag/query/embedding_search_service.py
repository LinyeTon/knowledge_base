from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log


@step_log("get_data_and_validates")
def get_data_and_validates(state) -> tuple[list[str], str]:
    item_names = state.get('item_names')
    rewritten_query = state.get('rewritten_query')
    if not item_names or not rewritten_query:
        logger.error(f"item_names 或 rewritten_query 不存在，无法继续业务！！")
        raise ValueError(f"item_names 或 rewritten_query 不存在，无法继续业务！！")
    return item_names, rewritten_query


@step_log("build_item_name_expr")
def build_item_name_expr(item_names: list[str]) -> str:

    return f"item_name in {item_names}"


@step_log("nomalize_retrieved_chunk")
def nomalize_retrieved_chunk(chunk: dict) -> dict:
    """
    将 Milvus 检索结果归一化为查询链内部统一使用的文档结构。

    Args:
        chunk: Milvus 返回的原始切块结果。

    Returns:
        dict: 标准化后的检索文档。
    """
    entity = chunk.get("entity", {})
    return {
        "chunk_id": chunk.get("id") or entity.get("chunk_id"),
        "item_name": entity.get("item_name", ""),
        "title": entity.get("title"),
        "parent_title": entity.get("parent_title"),
        "part": entity.get("part"),
        "file_title": entity.get("file_title"),
        "content": entity.get("content", ""),
        "score": chunk.get("distance", 0.0),
        "type": "milvus",
        "url": None,
    }


@step_log("search_chunks")
def search_chunks(
        *,
        rewritten_query: str,
        item_names: list[str]
) -> list[dict]:
    """
    基于改写问题执行一次混合向量检索。

    Args:
        rewritten_query: 用于检索的改写后问题。
        item_names: 已确认的主体名称列表，用于过滤知识范围。
        limit: 最大返回文档数。

    Returns:
        list[dict]: 检索得到的切块结果列表。
    """

    embedding_result = llm_provider.embed_documents([rewritten_query])
    dense_vector = embedding_result['dense'][0]
    sparse_vector = embedding_result['sparse'][0]

    reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=build_item_name_expr(item_names),
        limit=5 * 2
    )

    resp = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_weights=(0.6, 0.4),
        norm_score=True,
        limit=5,
        output_fields=[
            "chunk_id",
            "title",
            "parent_title",
            "file_title",
            "item_name",
            "content",
            "part"
        ]
    )
    return [nomalize_retrieved_chunk(chunk) for chunk in (resp[0] if resp else [])]



@step_log("search_by_embedding")
def search_by_embedding(state: QueryGraphState) -> list[dict]:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """
    item_names, rewritten_query = get_data_and_validates(state)
    return search_chunks(rewritten_query=rewritten_query, item_names=item_names)
