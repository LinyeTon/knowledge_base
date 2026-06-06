import datetime

from pymilvus import DataType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger
from app.shared.runtime.logger import step_log



@step_log("require_chunks")
def require_chunks(state: ImportGraphState) -> list[dict]:
    chunks = state.get("chunks", [])
    if not chunks:
        logger.error("chunks为空, 无法继业务！！")
        raise ValueError("chunks为空， 业务无法继续！！")

    return chunks


@step_log("prepare_chunks_collection")
def prepare_chunks_collection():
    """
    准备 Milvus 切片集合
    功能：检查集合是否存在，不存在则创建 schema 和索引
    :return: 无返回值
    """
    # 获取 Milvus 客户端
    milvus_client = milvus_gateway.client

    # 获取集合名称
    collection_name = milvus_gateway.item_collection_name

    # 如果集合已存在，直接返回，无需重复创建
    if milvus_client.has_collection(collection_name=collection_name):
        logger.info(f"{collection_name} 对应的集合已存在，无需创建，直接使用即可")
        return

    # ===================== 创建 Schema =====================
    # 创建 schema， 启用自动 ID 和动态字段
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 添加主键字段： chunk_id， INT64 类型，自增
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 添加文件标题字段： VARCHAR 类型，最大长度 512
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加主题名称字段： VARCHAR 类型， 最大长度 512
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片标题字段：VARCHAR 类型， 最大长度 512
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)

    # 添加父标题字段： VARCHAR 类型， 最大长度 512
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片序号字段： INT8 类型
    schema.add_field(field_name="part", datatype=DataType.INT8)

    # 添加内容字段： VARCHAR 类型，最大长度 65535 （支持长文本）
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    # 添加稠密向量字段： FLOAT_VECTOR 类型， 维度 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

    # 添加稀疏向量字段： FLOAT_VECTOR 类型
    schema.add_field(field_name="sparse_vector", datatype=DataType.FLOAT_VECTOR)


    # ===================== 创建索引 =====================
    # 准备索引参数
    index_params = milvus_client.prepare_index_params()

    # 为稠密向量创建索引：使用 AUTOINDEX，metric_type 为 IP（内积）
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={
            "M": 64,  # Maximum number of neighbors each node can connect to in the graph
            "efConstruction": 100  # Number of candidate neighbors considered for connection during index construction
        }  # Index building params
    )

    # 为稀疏向量创建索引：使用 SPARSE_INVERTED_INDEX，算法为 DAAT_MAXSCORE
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 创建集合并应用索引
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"{collection_name}完成对应的集合创建！")


# 根据文件名称删除已存在的切片记录
# 功能：实现幂等性，确保同一主体重复导入时覆盖旧数据
@step_log("remove_old_chunks")
def remove_old_chunks(file_title: str):
    milvus_gateway.client.delete(
        collection_name=milvus_gateway.item_collection_name,
        filter=f"file_title=='{file_title}'"
    )


# 批量插入切片数据到 Milvus 集合
@step_log("insert_chunks")
def insert_chunks(chunks: list[dict]):
    result = milvus_gateway.client.insert(
        collection_name=milvus_gateway.item_collection_name,
        data=chunks,
    )
    logger.info(f"插入数据成功! 总条数:{result.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显:{result.get('ids', [])}")

@step_log("index_chunks")
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    # 目标： 将 chunks 存储到向量数据库

    # 1. 获取chunks并校验
    chunks = require_chunks(state)
    # 2. 准备 collection集合，（chunk schema  indexes  collection）
    prepare_chunks_collection()
    # 3. 插入数据 （先删除，再插入）
    remove_old_chunks(state['file_title'])
    insert_chunks(chunks)
    # 4. log
    logger.info(f"{datetime.datetime.now().strftime('%Y%m%d')}完成{state['task_id']}导入文件数据入库操作!")
    return state