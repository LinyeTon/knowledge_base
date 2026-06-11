from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def get_data_and_validates(state:QueryGraphState) -> tuple[str,list[str]]:
    """
    获取参数和校验
    :param state:
    :return: 重写问题以及关联的item_names
    """
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names",[])

    if not rewritten_query or len(item_names) == 0:
        logger.error(f"重写问题或者关联的主体为空,无法继续业务!")
        raise ValueError(f"重写问题或者关联的主体为空,无法继续业务!")

    return rewritten_query, item_names


def call_llm_by_rewritten_query(rewritten_query) -> str:
    """
      调用模型生成假设性答案
    :param rewritten_query:
    :return:
    """
    # 1. 获取模型对象
    llm_client = llm_provider.chat()
    # 2. 加载和封装提示词
    prompt_text = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    messages = [
        HumanMessage(content=prompt_text)
    ]
    # 3. 封装调用链
    chains = llm_client | StrOutputParser()
    # 4. 执行
    hyde_answer = chains.invoke(messages)

    return hyde_answer


def  milvus_search_hyde_entity(hyde_answer, rewritten_query, item_names):
    """
      使用重写问题,对向量库进行搜索!
      注意: 需要添加item_name的过滤条件
    :param hyde_answer:
    :param rewritten_query:
    :param item_names:
    :return: 返回处理一层后的列表
    """
    # 1. 向量化rewritten_query
    embedding_result = llm_provider.embed_documents([rewritten_query + ":" + hyde_answer])
    dense_vector = embedding_result['dense'][0]
    sparse_vector = embedding_result['sparse'][0]
    # 2. 创建 annSearchRequest
    ann_reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr= f"item_name in {item_names}",
        limit=5 * 2
    )
    # 3. 调用混合检索（设置输出列）
    milvus_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=ann_reqs,
        ranker_weights=(0.6, 0.4),
        limit=5,
        norm_score=True,
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
    # 4. 返回第一层结果
    return milvus_result[0] if milvus_result and len(milvus_result) > 0 else []


def normalize_retrieved_chunk(milvus_response: list[dict]) -> list[dict]:
    final_list_dict = []
    for milvus_dict in milvus_response:
        # milvus_dict {id , distance , entity : {} }
        entity = milvus_dict.get("entity",{})

        final_list_dict.append(
            {
                "chunk_id": milvus_dict.get("id") or entity.get("chunk_id"),  # 片段ID
                "item_name": entity.get("item_name", ""),  # 归属主体名称
                "title": entity.get("title"),  # 片段标题
                "parent_title": entity.get("parent_title"),  # 父标题/章节
                "part": entity.get("part"),  # 部分标识
                "file_title": entity.get("file_title"),  # 来源文件标题
                "content": entity.get("content", ""),  # 片段文本内容
                "score": milvus_dict.get("distance", 0.0),  # 相似度分数
                "type": "milvus",  # 来源类型（向量库）
                "url": None,  # 附件URL（无）
            }
        )
    return final_list_dict



def search_by_hyde(state: QueryGraphState):
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    """

    # 1. 参数获取和校验
    rewritten_query, item_names = get_data_and_validates(state)
    # 2. 根据问题获取答案
    hyde_answer = call_llm_by_rewritten_query(rewritten_query)
    # 3. 进行向量库混合内容检索
    milvus_result = milvus_search_hyde_entity(hyde_answer, rewritten_query, item_names)
    # 4. 进行数据格式化处理
    #  [dict {id , distance , entity : {} } -> 目标格式  {}]
    final_list_dict = normalize_retrieved_chunk(milvus_result)

    return final_list_dict