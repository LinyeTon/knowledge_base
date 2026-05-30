import copy
from typing import TypedDict

class ImportGraphState(TypedDict):

    # 任务状态
    task_id: str

    # 文件状态判断
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # 地址路径内容
    local_file_path: str # 要解析的文件地址
    local_dir: str  # 存储生成的md文件路径
    md_path: str    # 专门存储生成的md文件路径
    pdf_path: str    # 专门存储生成的pdf文件路径
    file_title: str # 文件标题名（去掉后缀）

    # 内容数据
    md_content:str  # Markdown 的全文内容
    chunks: list    # 切片后文本列表，包含metadata
    item_name: str  # 识别出的主体名称，用于增强检索

    # 数据库相关
    embeddings_content: list # 包含向量数据的列表，准备写入Milvus


# 提供创建原始state的方法

default_ImporetGraphState: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    copy_state = copy.deepcopy(default_ImporetGraphState)

    copy_state.update(overrides)

    return copy_state


def get_default_state() -> ImportGraphState:
    return copy.deepcopy(default_ImporetGraphState)



