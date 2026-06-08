from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
import time
import sys

def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    ""
    print("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    session_id = state["session_id"]
    is_stream = state.get("is_stream", True)
    base_answer = state.get("answer") or f"这是关于「{state.get('original_query', '当前问题')}」的测试回答，正在演示打字机流式输出效果。今天是个大晴天!天气非常好!晚上要[跑20公里!哈哈哈哈 你跑!阿斯卡等哈时间跨度哈市登记卡哈受打击看哈手机打卡萨哈久啊回到家卡仕达酱卡受打击阿莎扩大阿德手机哈时间跨度哈市登记卡花洒登记卡哈萨阿贾克斯等哈数据库打火机啊苏卡达合计阿萨达哈"
    final_text = ""

    if is_stream:
        for ch in base_answer:
            final_text += ch
            # pust_to_session see (session_id ,delta , {delta:ch})
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.06)

        push_to_session(session_id, SSEEvent.DELTA, {"delta": "哈哈哈"})
        push_to_session(session_id, SSEEvent.DELTA, {"delta": "哈哈哈"})
        push_to_session(session_id, SSEEvent.DELTA, {"delta": "哈哈哈"})

        time.sleep(0.66)
        logger.info(f"流式输出完成，总长度: {len(final_text)}")
    else:
        final_text = base_answer


    # 存储答案
    history_repository.save_message(session_id=state['session_id'], role="assistant",
                                    text=final_text, rewritten_query=state['rewritten_query'],
                                    item_names=state["item_names"], image_urls=state['image_urls'])

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---node_answer_output 节点处理结束---")
    # 关键点：return 必须保留 session_id！
    return {
        "session_id": session_id,  # 必须带回去
        "answer": "你的回答内容",
        "is_stream": state.get("is_stream")
    }