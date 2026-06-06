from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS


# 校验 chunks 和 file_title
@step_log("validate_chunks_and_title")
def validate_chunks_and_title(state) -> tuple[list[dict], str]:
    # 1. 获取数据 chunks 和 file_title
    chunks = state.get("chunks")
    file_title = state.get("file_title")
    # 2. 非空判断
    if not chunks:
        logger.error(f"chunks内容为空，无法继续业务")
        raise ValueError(f"chunks内容为空，无法继续业务")
    if not file_title:
        file_title = chunks[0]['file_title'] or "default_file__title"

    return chunks, file_title


# 1. 截取前 K 个切片（由 `ITEM_NAME_CONTEXT_CHUNK_K` 控制）
# 2. 遍历切片，拼接格式化字符串："切片:{index},标题:{title},内容:{content}"
# 3. 将所有切片字符串用换行符连接
# 4. 截断到最大字符数限制（由 `ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS` 控制）
# 5. 返回拼接后的上下文字符串
@step_log("build_document_context")
def build_document_context(chunks) -> str:
    # 1. 截取 top K chunk 内容
    top_chunk = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    # 2.  拼接上下文
    # 切片： 1 标题：x 父标题： x  内容： x \n
    context = ""
    for index, chunk in enumerate(top_chunk, start=1):
        context += f"切片:{index} 标题:{chunk['title']} 父标题: {chunk['parent_title']} 内容: {chunk['content']} \n"
    # 3. 最长上下文长度限制
    final_context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    return final_context


# 1. 获取 LLM 客户端
# 2. 加载系统提示词模板 `product_recognition_system`
# 3. 加载用户提示词模板 `item_name_recognition`，传入 `file_title` 和 `context`
# 4. 构造消息列表（SystemMessage + HumanMessage）
# 5. 调用 LLM 并解析输出
# 6. 如果识别结果为空，使用 `file_title` 兜底
# 7. 返回识别出的主体名称
@step_log("recognize_item_name")
def recognize_item_name(context:str, file_title: str) -> str:
    # 1. 获取 llm 的客户端对象
    chat_model = llm_provider.chat()
    # 2. 加载外部的提示词
    system_prompt_str = load_prompt("product_recognition_system")
    humman_prompt_str = load_prompt(
        "item_name_recognition",
        file_title=file_title,
        context=context,
    )
    # 3. 封装成我们提示词格式 HumanMessage  SystemMessage
    messages = [
        SystemMessage(content=system_prompt_str),
        HumanMessage(content=humman_prompt_str),
    ]
    # 4. 组装调用链
    chains = chat_model | StrOutputParser()
    item_name = chains.invoke(messages)
    logger.info(f"调用模型进行item_name识别完毕! item_name:{item_name}")
    # 非空判断和兜底
    if not item_name:
        item_name = file_title
    return item_name

# 给 chunks -> chunk -> 赋上item_name值
@step_log("apply_item_name")
def apply_item_name(chunks: list[dict], item_name: str):
    for chunk in chunks:
        chunk['item_name'] = item_name
    logger.info(f"完成chunks的item_name数据补充！ {chunks[0]['item_name']}")


# 根据 item_name 生成稠密和稀疏向量
@step_log("embed_item_name")
def embed_item_name(item_name: str):
    result = llm_provider.embed_documents([item_name])
    return result['dense'][0], result['sparse'][0]


@step_log("prepare_item_name_collection")
def prepare_item_name_collection():
    milvus_client = milvus_gateway.client
    # 判断集合是否存在
    if milvus_client.has_collection(collection_name=milvus_gateway.item_collection_name):
        # 存在
        logger.info(f"{milvus_gateway.item_collection_name}对应的集合存在,无需创建!")
        return
    # 创建集合对应的 schema [field列]
    # create schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )

    # add fields to schema
    # https://milvus.io/docs/zh/v2.6.x/sparse_vector.md
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 创建集合对应的index
    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        # 给哪个字段创建索引 字段应该是经常查询的字段
        field_name="dense_vector",
        # 索引的类型 索引就是外部创建一种高效的数据类型  [目录]-> 查询 -> 内存地址 -> 链接到对应的实体数据
        # 推荐: AUTOINDEX -> 自动创建索引 自动选择类型 我有点不推荐!
        # 为了减少学习曲线，Milvus 提供了AUTOINDEX。通过AUTOINDEX，Milvus 可以在建立索引的同时分析 Collections
        # 中的数据分布，并根据分析结果设置最优化的索引参数，从而在搜索性能和正确性之间取得平衡。
        # HNSW : 分层图 -> 类似地图搜索过程  [精度最高 / 内存在有最大]
        # IVF_FLAT : 分桶 nlist = 64 找到对应桶 / 细化筛选 [比 FLAT快, 占有内存中等]
        # FLAT :  直接所有向量搜索和比较 [最慢]
        index_type="HNSW",
        # 相识度算法 L2 [0-2] COSINE  IP  [-1 1]
        metric_type="COSINE",
        params={
            "M": 64,  # Maximum number of neighbors each node can connect to in the graph
            "efConstruction": 100  # Number of candidate neighbors considered for connection during index construction
        }  # I
    )

    index_params.add_index(
        field_name="sparse_vector",
        # 稀疏向量 2.6 只有倒排索引
        # 内容 -> 向量相似度
        # doc1 = {1:x 3:x}
        # doc2 = {1:x,4:x}
        # 1位置 = doc1 , doc2
        # 3位置 = doc1
        # 4位置 = doc2
        # 搜索的稀疏向量 {1:k} -> doc1 doc2
        index_type="SPARSE_INVERTED_INDEX",
        # IP (内积）：使用点积衡量相似性。
        metric_type="IP",
        # 算法识别 影响小的值跳过,提高相似度比较的效率
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )

    # 创建集合
    milvus_client.create_collection(
        collection_name=milvus_gateway.item_collection_name,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"{milvus_gateway.item_collection_name} 第一次完成初始化！！")


@step_log("upsert_item_name")
def upsert_item_name(item_name: str, file_title: str, dense_vector: list[float], sparse_vector: list[int | float]):
    # 先删除 / 再插入 幂等性
    milvus_client = milvus_gateway.client
    # 1. 先根据file_title 删除
    milvus_client.delete(
        collection_name=milvus_gateway.item_collection_name,
        filter=f"file_title == '{file_title}'"
    )
    result = milvus_client.insert(
        collection_name=milvus_gateway.item_collection_name,
        data=[{
            "item_name": item_name,
            "file_title": file_title,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector
        }]
    )
    logger.info(f"{item_name}对应的数据已经插入到{milvus_gateway.item_collection_name}对应的集合中！ 返回结果：{result}")





"""
主体识别服务：
1. 基于 chunks 构造上下文
2. 调用 LLM 识别 item_name
3. 将 item_name 回填到 state 和 chunks
4. 同步写入主体名称索引
"""
@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    # 1. 进行参数校验
    chunks , file_title =  validate_chunks_and_title(state)
    # 2. 进行上下文的拼接 chunks
    # chunk content title parent_title
    context =  build_document_context(chunks)
    # 3. 进行item_name的识别了 llm
    item_name = recognize_item_name(context,file_title)
    # 4. 修改所有chunks的item_name属性
    apply_item_name(chunks, item_name)
    # 5. 对 item_name 进行向量化，生成稠密和稀疏向量
    dense_vector, sparse_vector = embed_item_name(item_name)
    # 6. 创建 item_name 集合
    prepare_item_name_collection()
    # 7 . 更新或插入 item_name 到集合中
    upsert_item_name(item_name, file_title, dense_vector, sparse_vector)

    # 8 . 更新 state
    # state['chunks'] = chunks
    state['item_name'] = item_name

    return state