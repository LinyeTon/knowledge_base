from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_SUMMARY_CHAR_RATIO, RERANK_MIN_SUMMARY_CHARS, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_RATIO, RERANK_GAP_ABS


@step_log("")
def get_rewritten_query_and_validate(state):
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    rewritten_query = state.get("rewritten_query")

    if len(rrf_chunks) == 0 or len(web_search_docs) == 0 or not rewritten_query:
        logger.error(f"关键参数为空， 业务无法继续进行！！")
        raise ValueError(f"关键参数为空，业务无法继续进行")

    return rrf_chunks, web_search_docs, rewritten_query

def merge_rrf_and_web(rrf_chunks, web_search_docs):
    """
        进行两路数据融合
        rrf_chunks ->  chunk_id title parent_title part file_title content type url item_name
        web_search_docs -> snippet title  url
    """
    final_chunk_list = []
    # 1. 循环rrf_chunks
    for chunk in rrf_chunks:
        final_chunk_list.append({
            "title": chunk.get("title"),
            "text": chunk.get("content"),
            "type": "milvus",
            "url": None,
            "score": 0.0
        })
    # 2. 循环 web_search_docs
    for doc in web_search_docs:
        final_chunk_list.append({
            "title": doc.get("title"),
            "text": doc.get("snippet"),
            "type": "web",
            "url": doc.get("url"),
            "score": 0.0
        })

    return final_chunk_list


# 对超长文本调用llm进行精简
@step_log("summarize_long_rerank_text")
def summarize_long_rerank_text(answer: str, question: str, limit: int) -> str:
    chat_client = llm_provider.chat()
    prompt_text = load_prompt("reranker_text_refine", question=question, answer=answer, limit=limit)
    messages = [
        HumanMessage(content=prompt_text)
    ]
    chain = chat_client | StrOutputParser()
    refine_answer = chain.invoke(messages)
    return refine_answer


@step_log("build_question_answer_pair_list")
def build_question_answer_pair_list(rewritten_query, final_chunk_list) -> list[list[str]]:
    #生成问题和答案对列表! 超长,调用 summarize_long_rerank_text
    question_answer_pair_list = []
    # 1. 检查问题的长度 rewritten_query
    reranker_model = llm_provider.reranker_model()
    tokenizer = reranker_model.tokenizer
    # add_special_tokens = False单纯算我这个字符串对应token列表! 不用关注我前后的特殊字符
    question_token_ids_list = tokenizer.encode(rewritten_query, add_special_tokens=False)
    question_token_len = len(question_token_ids_list)
    # 2. 循环答案final_chunk_list
    for chunk in final_chunk_list:
        current_answer = chunk.get("text", "")
        # 3. 检查答案的长度
        current_answer_token_len = len(tokenizer.encode(current_answer, add_special_tokens=False))
        # 4. 超长就进行模型压缩
        if current_answer_token_len + question_token_len + 4 > RERANK_MAX_INPUT_TOKENS:
            # 计算字符串长度
            limit = max(
                RERANK_MIN_SUMMARY_CHARS,
                # 转成整数
                int((RERANK_MAX_INPUT_TOKENS - question_token_len - 4) / RERANK_SUMMARY_CHAR_RATIO)
            )
            # 调用模型
            current_answer = summarize_long_rerank_text(rewritten_query, current_answer, limit)
        # 5. 添加到列表中
        question_answer_pair_list.append(current_answer)

    return question_answer_pair_list


@step_log("reranker_scpre_pair_list")
def reranker_score_pair_list(question_answer_pair_list):
    # 调用reranker模型打分
    reranker_model = llm_provider.reranker_model()
    # normalize=True 归一化 将分值拉倒 0 -1之间!方便进行后续算法统计!!
    score_list = reranker_model.compute_score(question_answer_pair_list, normalize=True)
    logger.info(f"reranker_scpre_pair_list打分的分数为: {score_list}")
    return score_list


@step_log("")
def sort_final_chunk_list(final_chunk_list, score_list):
    for chunk, score in zip(final_chunk_list, score_list):
        chunk['score'] = score

    # 获得是带有打分的列表数据，没有排序
    logger.info(f"没排序前的顺序: {final_chunk_list}")
    logger.info("*"*60)
    final_chunk_list.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"排序后的顺序: {final_chunk_list}")
    return final_chunk_list


@step_log("dynamic_topk")
def dynamic_topk(final_chunk_list) -> list[dict]:
    # 动态断崖截取chunk_list
    max_num = RERANK_MAX_TOPK
    min_num = RERANK_MIN_TOPK
    gap_abs = RERANK_GAP_ABS
    gap_ratio = RERANK_GAP_RATIO

    #处理max_number 大于列表长度的可能
    max_num = min(max_num, len(final_chunk_list))
    # 声明 topk 并赋值 max_num  没有断崖，默认截取全部
    top_k = max_num
    # 循环寻找断崖
    # 有可能 设置的min > max
    if max_num > min_num:
        for index in range(min_num - 1, max_num - 1):
            score_1 = final_chunk_list[index].get("score", 0.0)
            score_2 = final_chunk_list[index + 1].get("score", 0.0)
            abs_score = score_1 - score_2
            # 相对断崖率， 加极小项防止分数为零
            ratio_score = abs_score / (score_1 + 1e-7)
            # 断崖判断
            if abs_score > gap_abs or ratio_score > gap_ratio:
                top_k = index + 1
                break

    logger.info(f"已经完成断崖数据截取，进入数量: {len(final_chunk_list)}, 截取数据: {top_k}")
    # 截取数据
    return final_chunk_list[:top_k]



@step_log("rerank_documents")
def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    # 1. 获取数据并校验
    rrf_chunks, web_search_docs, rewritten_query = get_rewritten_query_and_validate(state)

    # 2. 多路数据融合格式统一
    final_chunk_list = merge_rrf_and_web(rrf_chunks, web_search_docs)

    # 3. 生成问题和答案列表 [[问题, 答案], ...]
    question_answer_pair_list = build_question_answer_pair_list(rewritten_query, final_chunk_list)

    # 4. rerank模型进行打分
    scorelist = reranker_score_pair_list(question_answer_pair_list)

    # 5. 原始数据进行赋分和排序
    final_chunk_list = sort_final_chunk_list(final_chunk_list, scorelist)

    # 6. 动态截取数据 topk
    final_chunk_list = dynamic_topk(final_chunk_list)

    state['reranked_docs'] = final_chunk_list

    return state