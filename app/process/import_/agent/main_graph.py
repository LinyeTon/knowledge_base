from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus
from app.shared.runtime.logger import logger

load_dotenv()

main_builder = StateGraph(ImportGraphState)

main_builder.add_node("node_entry", node_entry)
main_builder.add_node("node_pdf_to_md", node_pdf_to_md)
main_builder.add_node("node_md_img", node_md_img)
main_builder.add_node("node_document_split", node_document_split)
main_builder.add_node("node_item_name_recognition", node_item_name_recognition)
main_builder.add_node("node_bge_embedding", node_bge_embedding)
main_builder.add_node("node_import_milvus", node_import_milvus)

main_builder.set_entry_point("node_entry")

def after_entry_node(state: ImportGraphState):
    """
    入口节点后的路由函数：
    - Markdown 文件：直接进入图片处理节点
    - PDF 文件：先进入 PDF 转 Markdown 节点
    - 其他类型：直接结束
    """
    if state["is_md_read_enabled"]:
        logger.info(f"node_entry节点判断的文件{state['local_file_path']}类型 md,跳转到:node_md_img 节点")
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        logger.info(f"node_entry节点判断的文件{state['local_file_path']}类型 pdf,跳转到:node_pdf_to_md 节点")
        return "node_pdf_to_md"
    else:
        logger.warning(f"node_entry节点获取的文件: {state['local_file_path']} 无法处理对应的类型,直接跳转到END节点!")
        return END

main_builder.add_conditional_edges(
    "node_entry",
    after_entry_node,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END,
    },
)

main_builder.add_edge("node_pdf_to_md", "node_md_img")
main_builder.add_edge("node_md_img", "node_document_split")
main_builder.add_edge("node_document_split", "node_item_name_recognition")
main_builder.add_edge("node_item_name_recognition", "node_bge_embedding")
main_builder.add_edge("node_bge_embedding", "node_import_milvus")
main_builder.add_edge("node_import_milvus", END)

kb_import_app = main_builder.compile()



if __name__ == "__main__":
    from app.shared.utils.path_util import PROJECT_ROOT
    import os
    from app.shared.runtime.logger import logger

    # 全流程测试：验证PDF导入→Milvus入库完整链路
    logger.info("===== 开始执行知识图谱导入全流程测试 =====")

    # 1. 构造测试文件路径（复用你项目的doc目录）
    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 2. 构造输出目录（存放MD/图片等中间文件）
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)  # 不存在则创建

    # 3. 校验测试PDF文件是否存在
    if not os.path.exists(test_pdf_path):
        logger.error(f"全流程测试失败：测试PDF文件不存在，路径：{test_pdf_path}")
        logger.info("请检查文件路径，或手动将测试文件放入项目根目录的doc文件夹中")
    else:
        # 4. 构造测试状态（贴合实际业务入参，开启PDF解析开关）
        test_state = ImportGraphState({
            "task_id": "test_kg_import_workflow_001",  # 测试任务ID
            "local_file_path": test_pdf_path,  # 测试PDF文件路径
            "local_dir": test_output_dir,  # 中间文件输出目录
            "is_pdf_read_enabled": False,  # 开启PDF解析（核心开关）
            "is_md_read_enabled": False  # 关闭MD解析
        })
        try:
            logger.info(f"测试任务启动，PDF文件路径：{test_pdf_path}")
            logger.info(f"中间文件输出目录：{test_output_dir}")
            logger.info("开始执行全流程节点，依次执行：entry→pdf2md→md_img→split→item_name→embedding→milvus")

            # 5. 执行LangGraph全流程（流式执行，打印节点执行进度）
            final_state = kb_import_app.invoke(test_state)

            # 6. 全流程执行完成，结果预览和核心指标打印
            if final_state:
               logger.info(f"最终结果: {final_state}")
        except Exception as e:
            logger.exception(f"===== 全流程测试运行失败 =====")
    logger.info("===== 知识图谱导入全流程测试结束 =====")