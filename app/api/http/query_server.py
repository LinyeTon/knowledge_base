import shutil
import uuid
from mimetypes import guess_type

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.query_schema import QueryRequestParam, QueryStreamResponse, QueryNotStreamResponse, \
    HistoryCleanResponse, HistoryResponse, HistoryItemResponse
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.main_graph import query_graph_app
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.utils.sse_utils import sse_generator, create_sse_queue, SSEEvent, push_to_session
from app.process.query.agent.state import QueryGraphState, create_query_default_state
from app.shared.utils.task_utils import clear_task, update_task_status, TASK_STATUS_FAILED, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, get_done_task_list

app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域问题  CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/html")
def chat_html():
    chat_html_path_obj = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"
    return FileResponse(
        path = chat_html_path_obj,
        media_type = guess_type(chat_html_path_obj.name)[0],
    )

@app.get("/health")
def health():
    return {
        "code": 200,
        "message": "可以访问！"
    }

@app.get("/stream/{session_id}")
def stream(session_id, request:Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )

def invoke_query_graph(session_id: str, query: str, is_stream:bool=False):
    # 执行  动态测试
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )

    #  创建一个队列 session_id <-- 数据

    # 清空task_utils 的数据
    clear_task(session_id)

    if is_stream:
        create_sse_queue(session_id)

    try:
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        logger.info(f"开始执行，执行参数为：{state}")
        result_state = query_graph_app.invoke(state)
        logger.info(f"执行结束，执行结果为：{result_state}")
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        image_urls = ["http://www.baidu.com/img/bd_logo.png"]

        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": result_state['answer'],
                "status": "completed",
                "image_urls": image_urls
            }
        )
        # 返回结果! 非流式需要
        return result_state
    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        logger.exception(f"{session_id}执行出现了异常!!")


@app.post("/query")
def query(backgroundtasks: BackgroundTasks, request: QueryRequestParam):
    """
      1. 获取stream状态
      2. true 异步 后台执行 图的调用过程 backgroundtask
         异步的返回结果
      3. false 同步 直接调用
         同步的返回结果
    """

    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    query = request.query

    # 是否异步
    if is_stream:
        # 异步执行
        backgroundtasks.add_task(
            invoke_query_graph,
            session_id=session_id,
            query=query,
            is_stream=is_stream
        )

        return  QueryStreamResponse(
            message=f"开启:{session_id}异步任务执行",
            session_id=session_id
        )
    else:
        # 同步执行
        final_state:QueryGraphState = invoke_query_graph(
            session_id=session_id,
            query=query,
            is_stream=is_stream
        )

        return QueryNotStreamResponse(
            message=f"{session_id}对应的任务已经处理完毕!!",
            session_id=session_id,
            answer=final_state.get("answer"),
            done_list=get_done_task_list(session_id),
            image_urls=final_state.get("image_urls")
        )

# 清空历史对话记录
@app.delete("/history/{session_id}")
def remove_history(session_id:str):
    delete_count = history_repository.clear_session(session_id=session_id)
    logger.info(f"清空:{session_id}对应的历史记录!清空数量:{delete_count}")
    return HistoryCleanResponse(
        message=f"清空:{session_id}对应的历史记录!清空数量:{delete_count}",
        deleted_count=delete_count
    )


@app.get("/history/{session_id}")
def get_history(session_id:str,limit:int = 10):

    message_list =  history_repository.list_recent(session_id=session_id,limit=limit)

    logger.info(f"完成:{session_id}对应的历史记录查询!查询的数据数量:{len(message_list)}")

    return HistoryResponse(
        session_id=session_id,
        items=[
              HistoryItemResponse(
                  id=str(message.get("_id")),
                  session_id=message.get("session_id"),
                  role=message.get("role"),
                  text=message.get("text"),
                  rewritten_query=message.get("rewritten_query"),
                  item_names = message.get("item_names"),
                  image_urls = message.get("image_urls"),
                  ts =message.get("ts")
              )
              for message in  message_list
        ]
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)