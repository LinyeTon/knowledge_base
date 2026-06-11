from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import step_log, logger


@step_log("get_data_and_validate")
def get_data_and_validate(state):
    """
     获取两路数据!
    :param state:
    :return:
    """
    embedding_chunks = state.get("embedding_chunks",[])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks",[])
    if len(embedding_chunks) == 0 or len(hyde_embedding_chunks) == 0:
        logger.error(f"查询数据为空列表,无法提取结果! 无法继续业务!")
        raise ValueError(f"查询数据为空列表,无法提取结果! 无法继续业务!")
    return embedding_chunks,hyde_embedding_chunks


@step_log("use_rrf_chunks_list")
def use_rrf_chunks_list(chunks_list: list[tuple[float, dict]], limit:int=5, k:int=60):
    """
    带有权重思维的rrf算法,计算top = limit  k 平滑参数! 减少排名对结果的过度影响!
    :param chunks_list:
    :param limit:
    :param k:
    :return:
    """
    # 1. 定义两个容器， 存储chunk_id: 累计分  存储chunk_id: chunk
    score_dict: dict[str, float] = {}
    chunk_dict: dict[str, dict] = {}
    # 2. 循环每路数据和对应的权重 （权重，列表） （ 权重， 列表）
    for weight, current_chunks in chunks_list:
        # 3. 循环当前路计算当前路得分
        # current_chunks = 5 2 3 4 1
        for rank, chunk in enumerate(current_chunks, start=1):
            score_dict[chunk['chunk_id']] = score_dict.get(chunk['chunk_id'], 0) + weight * (1 / (k + rank))
            # 相同的chunk除了chunk_id，其他都一样，直接覆盖
            # chunk_dict[chunk['chunk_id']] = chunk
            # 优雅写法
            chunk_dict.setdefault(chunk['chunk_id'], chunk)
    # 4. 处理chunk列表，并进行排序
    # chunk_id 分  chunk_id chunk score -> milvus
    chunk_list = []
    for chunk_id, score in score_dict.items():
        chunk = chunk_dict.get(chunk_id)
        chunk['score'] = score
        chunk_list.append(chunk)

    chunk_list.sort(key=lambda x: x['score'], reverse=True)
    # 5. 截取limit数量chunk列表
    rrf_chunks = chunk_list[:limit]
    return rrf_chunks


@step_log("fuse_by_rrf")
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """
    # 1. 获取数据和校验
    embedding_chunks, hyde_embedding_chunks = get_data_and_validate(state)
    # 2. 封装带有权重的结构
    chunks_list = [
        (1.0, embedding_chunks),
        (1.0, hyde_embedding_chunks)
    ]
    # 3.  使用rrf算法计算和解决内容
    rrf_chunks = use_rrf_chunks_list(chunks_list, limit=5, k=60)
    # 4. 返回综合积分高的chunk列表
    state['rrf_chunks'] = rrf_chunks
    return state