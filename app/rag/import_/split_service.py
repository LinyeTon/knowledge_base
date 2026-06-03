import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log
from app.rag.import_.config import *


@step_log("load_markdown_content")
def load_markdown_content(state: ImportGraphState) -> tuple[str, str, Path]:
    # 1. 获取参数 md_content md_path
    md_path = state.get("md_path")
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    # 2. md_path非空校验
    if not md_path:
        logger.error("md_path为空,无法获取图片地址等,业务无法继续!")
        raise ValueError("md_path为空,无法获取图片地址等,业务无法继续!")
    # 3. md_content进行非空校验 / 空给与默认值
    md_path_obj:Path = Path(md_path)
    if not md_content:
        logger.info(f"md_content没有内容,可能从md数据格式过来的!根据md_path二次读取即可!")
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"从{md_path}读取md_content内容失败,业务无法继续进行!!")
            raise ValueError(f"从{md_path}读取md_content内容失败,业务无法继续进行!!")
    # ===================== 处理 file_title 缺失场景 =====================
    # 如果标题为空，使用文件名（无后缀）作为标题；无路径则使用默认值
    if not file_title:
        file_title = Path(md_path).stem if md_path else "default"
        state["file_title"] = file_title  # 回填到状态
    # ===================== 统一文本格式 =====================
    # 替换所有换行符为 \n，解决 Windows/Linux 换行符不一致问题
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 返回处理好的文本内容 + 标题，给后续切块使用
    return md_content, file_title , Path(md_path)


@step_log("split_by_titles")
def split_by_titles(md_content: str, file_title: str) -> list[dict]:
    """
    按 Markdown 标题（#、##、###...）进行【语义化文档切块】
    特点：
        1. 自动识别标题，保证段落语义完整
        2. 跳过代码块内部的内容，不把 ``` 内的内容误判为标题
        3. 每个块包含：内容、当前标题、文档标题，方便后续检索
    :param md_content: Markdown 文本内容
    :param file_title: 文档名称（用于溯源）
    :return: 切块列表，每个元素是 {content, title, file_title}
    """
    # 正则：匹配 Markdown 标题（# ~ ###### 开头的行）
    reg = re.compile(r"^\s*#{1,6}\s.+")
    # 将全文按换行符切割成逐行处理
    lines = md_content.split("\n")
    # 存储最终切块结果
    chunks: list[dict] = []
    # 当前正在拼接的标题
    current_title = None
    # 当前块的所有行内容
    current_title_lines: list[str] = []
    # 标记： 是否处于代码块（```...```) 内部
    is_code_block = False
    # 记录切块数量
    chunk_size = 0

    # 逐行遍历 md 内容
    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            logger.warning("处理行为空行,跳过本次筛选!!")
            continue

        # ===================== 代码块判断 =====================
        # 遇到 ``` 或 ~~~ 标记，切换代码块状态
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line)
            continue

        # ===================== 识别标题并切分 =====================
        # 如果当前行是标题，并且**不在代码块内**，才进行切分
        if reg.match(line) and not is_code_block:
            # 如果已有上一个块内容，就把上一个块保存
            if current_title and len(current_title_lines) > 1:
                chunks.append({
                    "content": "\n".join(current_title_lines),
                    "title": current_title,
                    "file_title": file_title
                })

            # 以当前行作为新块的标题
            current_title = line
            current_title_lines = [current_title]
            chunk_size += 1

        else:
            current_title_lines.append(line)

    # ===================== 保存最后一个块 =====================
    if current_title and len(current_title_lines) > 1:
        chunks.append({
            "content": "\n".join(current_title_lines),
            "title": current_title,
            "file_title": file_title
        })

    # ===================== 兜底：全文无标题时 =====================
    if chunk_size == 0:
        chunks.append({
            "content": md_content,
            "title": "default",
            "file_title": file_title
        })
    logger.info(f"完成文档语义切割,共计切出:{chunk_size}块! 切块内容:{chunks}")
    return chunks


@step_log("_split_long_section")
def _split_long_section(section: dict[str, Any], max_length: int = CHUNK_MAX_SIZE) -> list[dict[str, Any]]:
    """
    内部工具函数：拆分【过长的文本块】，保证单个chunk不超过最大长度限制
    核心逻辑：
        1. 检查内容长度，不长则直接返回
        2. 标题单独保留，只拆分正文内容
        3. 使用语义化拆分器，按段落、句子拆分，保证语义完整
    :param section: 待拆分的切块（包含title、content等）
    :param max_length: 单个块最大字符长度
    :return: 拆分后的子块列表
    """
    # 获取块的正文内容
    content = section.get("content", "") or ""
    # 1 . content 的格式清理
    title = section.get("title")
    body = content
    if content.startswith(title):
        body = content[len(title):].lstrip()

    # 2. 定义每块的固定前缀 和 块的有效长度
    prefix = title + "\n"
    available_length = max_length - len(prefix)

    # 3. 定义初始化递归字符拆分器（Langchain官方工具）
    # 按 段落→换行→句子→空格  优先级拆分，保证语义完整
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_length,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？"],
    )
    sub_sections = []

    # 4. 遍历拆分后的正文片段，生成字块
    for index, chunk_text in enumerate(splitter.split_text(body), start=1):
        text = chunk_text.strip()
        # 跳过空内容
        if not text:
            continue
        # 拼接完整内容： 标题 + 拆分后的正文
        full_text = (prefix + text).strip()

        sub_sections.append({
            "title": f"{title}-{index}" if title else f"chunk-{index}",  # 子标题：原标题-序号
            "content": full_text,  # 完整内容
            "parent_title": title,  # 父标题（用于溯源）
            "part": index,  # 序号（同一章节下的第N部分）
            "file_title": section.get("file_title"),  # 文档原始标题
        })

    logger.debug(f"已经完成{title}对应块进行短切! 切后块数为:{len(sub_sections)} , 数据预览: {sub_sections}")
    # 返回拆分后的所有子块
    return sub_sections

@step_log("_merge_short_chunks")
def _merge_short_chunks(final_chunks: list[dict], max_length: int = CHUNK_MAX_SIZE, min_length: int = CHUNK_SIZE) -> list[dict]:
    """
       同一个标题,小于600,进行合并,合并后不能大于1000
    :param final_chunks:
    :param max_length:
    :param min_length:
    :return:
    """
    # 1. 声明合并后的列表结果
    final_merge_chunks = []
    # 2. 记录第一个指针chunk位置
    start_chunk = None
    # 3. 循环对后续chunk进行合并处理
    for next_chunk in final_chunks:
        # 4. 首次迭代
        if not start_chunk:
            start_chunk = next_chunk
            continue
        # 5. 判断content长度是否小于600， and next 是否为同一个父标题
        is_lt_chunk_size = len(next_chunk.get("content")) < min_length
        is_same_parent_title = start_chunk.get("parent_title") and start_chunk.get("parent_title") == next_chunk.get("parent_title")
        if is_lt_chunk_size and is_same_parent_title:
            # 6. 清理next的标题内容，判断合并长度
            next_content_to_title = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 2:]
            start_content = start_chunk.get("content")
            # 7. 长度校验
            merged_content = start_content + next_content_to_title
            if len(merged_content) <= max_length:
                start_chunk["content"] = merged_content

                logger.info(
                    f"父标题:{start_chunk['parent_title']}, start: {start_chunk['title']}  next: {next_chunk['title']} 完成合并!!")
            else:
                final_merge_chunks.append(start_chunk)
                start_chunk = next_chunk
                continue
        else:
            final_merge_chunks.append(start_chunk)
            start_chunk = next_chunk
        # 循环执行完毕，将最后的chunk加入结果列表
        if start_chunk is not None:
            final_merge_chunks.append(start_chunk)
    return final_merge_chunks


@step_log("refine_chunks")
def refine_chunks(sections: list[dict], max_len: int = CHUNK_MAX_SIZE, min_len: int = CHUNK_SIZE) -> list[dict]:
    """
        【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
        执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
        :param sections: 步骤3处理后的章节列表
        :param max_len: 单个Chunk最大字符长度
        :return: 长度适中、低碎片化的最终Chunk列表
    """
    # 边界处理： 最大长度无效（为空或0），直接返回原章节，避免切分异常
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4： Chunk最大长度配置无效（{max_len}）， 跳过精细化处理")
        return sections

    # 阶段1： 切分超长章节 -> 所有章节长度控制在max_len内
    refined_split = []
    for sec in sections:
        if len(sec["content"]) > max_len:
            # 对每个章节执行超长切分，结果平铺加入列表（避免嵌套）
            refined_split.extend(_split_long_section(sec, max_len))
        else:
            refined_split.append(sec)
    logger.info(f"步骤4-1： 超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")
    # 阶段2： 合并过短章节
    final_merged_chunks = _merge_short_chunks(refined_split)

    # 阶段3： 父标题兜底
    for chunk in final_merged_chunks:
        if not chunk.get("parent_title"):
            chunk["parent_title"] = chunk["title"]
        if "part" not in chunk:
            chunk["part"] = 1

    return final_merged_chunks


@step_log("backup_chunks_json")
def backup_chunks_json(final_chunks: list[dict], md_path_obj: Path):
    """
       备份Chunks到JSON文件
    :param final_chunks:
    :param md_path_obj:
    :return:
    """
    json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}.json"
    # 写出内容即可 .josn → 字符串
    json_path_obj.write_text(json.dumps(final_chunks, ensure_ascii=False, indent=4), encoding="utf-8")


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
        文档切块核心节点（RAG 最关键步骤）
        功能：加载增强后的 Markdown 内容 → 按标题智能切块 → 优化块大小 → 备份切块结果 → 写入状态
        输出：将分块后的文本列表存入 state，供后续向量化、入库使用
    """
    # 1. 从状态中加载【增强后的Markdown内容】和【文档标题】
    md_content, file_title, md_path_obj = load_markdown_content(state)
    # 2. 按 Markdown 标题 （#、##、###）进行【智能语义切块】  （保持段落完整性）
    chunks = split_by_titles(md_content, file_title)
    # 3. 精细切割：长切短合
    final_chunks = refine_chunks(chunks)
    # 4. 备份final_chunks到JSON文件
    backup_chunks_json(final_chunks, md_path_obj)
    # 5. 修改state状态 chunks
    state["chunks"] = final_chunks

    return state

    return state