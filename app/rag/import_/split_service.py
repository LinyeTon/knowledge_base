import re
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log


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
    # 4. 获取file_title
    if not file_title:
        file_title = Path(md_path).stem

    # 5. 返回结果
    return md_content, file_title, md_path_obj

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
    reg = re.compile(r"^\s*#[1:6]\s.+")
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
            current_title_lines = []
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
            "title": "defaule",
            "file_title": file_title
        })

    return chunks










def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """

    return state