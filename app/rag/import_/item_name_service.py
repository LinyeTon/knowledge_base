from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger

from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS


# 校验 chunks 和 file_title

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


"""
主体识别服务：
1. 基于 chunks 构造上下文
2. 调用 LLM 识别 item_name
3. 将 item_name 回填到 state 和 chunks
4. 同步写入主体名称索引
"""

def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    # 1. 进行参数校验
    chunks , file_title =  validate_chunks_and_title(state)
    # 2. 进行上下文的拼接 chunks
    # chunk content title parent_title
    context =  build_document_context(chunks)
    # 3. 进行item_name的识别了 llm
    item_name = recognize_item_name(context,file_title)

    logger.warning(item_name)

    return state