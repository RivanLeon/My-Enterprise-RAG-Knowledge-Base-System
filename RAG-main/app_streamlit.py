"""
企业知识库 RAG 问答界面
启动方式（在项目根目录 RAG-main 下执行）：
    streamlit run app_streamlit.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pyprojroot import here

# 保证从项目根目录可导入 src
PROJECT_ROOT = here()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline, max_config, RunConfig
from src.retrieval import HybridRetriever, VectorRetriever
from test_frontend_e2e import run_frontend_e2e_check

load_dotenv()

DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"

# 界面「问题类型」与 pipeline kind 映射
KIND_MAP = {
    "text": "string",
    "boolean": "boolean",
    "number": "number",
    "name": "name",
}

CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(90deg, #6a5acd 0%, #7b68ee 40%, #5b8def 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.6rem !important;
        margin: 0 0 0.35rem 0 !important;
        font-weight: 700 !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.92) !important;
        margin: 0.15rem 0 !important;
        font-size: 0.92rem !important;
    }
    .panel-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.6rem;
    }
    .result-card {
        background: #fafafa;
        border: 1px solid #e8e8ef;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .result-meta {
        color: #555;
        font-size: 0.88rem;
        line-height: 1.7;
    }
    .answer-box {
        background: linear-gradient(135deg, #f8f6ff 0%, #f0f4ff 100%);
        border-left: 4px solid #6a5acd;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6a5acd, #7b68ee) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] .stRadio > div {
        gap: 0.4rem;
    }
</style>
"""


def load_companies() -> list[str]:
    """从 subset.csv 加载公司列表（去重，保留首次出现顺序）"""
    subset_path = DATA_ROOT / "subset.csv"
    try:
        df = pd.read_csv(subset_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(subset_path, encoding="gbk")
    names = df["company_name"].dropna().astype(str).tolist()
    # subset 中同一公司可对应多份文档，下拉框只展示公司名一次
    return list(dict.fromkeys(names))


def build_full_question(company: str, question: str) -> str:
    """将所选公司名写入问题，便于 pipeline 识别公司"""
    question = question.strip()
    if company in question:
        return question
    return f"{company}：{question}"


@st.cache_resource(show_spinner="正在初始化 RAG 流水线...")
def get_pipeline(enable_rerank: bool, top_n: int, answering_model: str) -> Pipeline:
    """按检索配置缓存 Pipeline 实例"""
    run_config = RunConfig(
        use_serialized_tables=max_config.use_serialized_tables,
        parent_document_retrieval=True,
        llm_reranking=enable_rerank,
        llm_reranking_sample_size=max_config.llm_reranking_sample_size,
        top_n_retrieval=top_n,
        parallel_requests=1,
        api_provider=max_config.api_provider,
        answering_model=answering_model,
        full_context=False,
        config_suffix=max_config.config_suffix,
    )
    return Pipeline(DATA_ROOT, run_config=run_config)


def clear_pipeline_cache():
    """清理 Pipeline 缓存，便于知识库更新后重新加载"""
    get_pipeline.clear()


def search_documents(
    company: str,
    question: str,
    enable_rerank: bool,
    top_n: int,
    vector_db_dir: Path,
    documents_dir: Path,
) -> list[dict]:
    """仅检索文档片段，不调用 LLM 生成答案"""
    if enable_rerank:
        retriever = HybridRetriever(vector_db_dir, documents_dir)
        return retriever.retrieve_by_company_name(
            company_name=company,
            query=question,
            llm_reranking_sample_size=max(10, top_n * 3),
            top_n=top_n,
            return_parent_pages=True,
        )
    retriever = VectorRetriever(vector_db_dir, documents_dir)
    return retriever.retrieve_by_company_name(
        company_name=company,
        query=question,
        top_n=top_n,
        return_parent_pages=True,
    )


def get_similarity_score(result: dict) -> float:
    """统一取相似度/相关度分数用于展示"""
    if "combined_score" in result:
        return float(result["combined_score"])
    if "relevance_score" in result:
        return float(result["relevance_score"])
    return float(result.get("distance", 0))


def render_retrieval_results(
    results: list[dict],
    company: str,
    question: str,
    elapsed: float,
):
    """渲染右侧检索结果列表"""
    st.markdown(
        f"**检索结果**（耗时: {elapsed:.2f}秒）  \n"
        f"公司: **{company}** | 问题: {question} | 找到 **{len(results)}** 个文档片段"
    )
    if not results:
        st.info("未检索到相关文档，请检查向量库是否已构建。")
        return
    for i, item in enumerate(results, 1):
        score = get_similarity_score(item)
        page = item.get("page", "-")
        source = item.get("source_file") or "未知来源"
        text = item.get("text", "")
        preview = text[:500] + ("..." if len(text) > 500 else "")
        st.markdown(
            f'<div class="result-card"><div class="panel-title">结果 {i}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"- 相似度: **{score:.4f}**  \n"
            f"- 来源: **{source}**  \n"
            f"- 页码: **{page}**  \n"
            f"- 内容:"
        )
        st.markdown(preview)


def render_answer(answer: dict, elapsed: float, model_name: str = ""):
    """渲染 LLM 生成的答案"""
    final = answer.get("final_answer", "N/A")
    reasoning = answer.get("reasoning_summary") or answer.get("step_by_step_analysis", "")
    pages = answer.get("relevant_pages", [])
    refs = answer.get("references", [])

    title = f"### 智能回答（耗时: {elapsed:.2f}秒）"
    if model_name:
        title += f"  \n模型: `{model_name}`"
    st.markdown(title)
    with st.container(border=True):
        st.markdown(str(final))
    if str(final).strip() == "N/A":
        st.warning("模型返回了 N/A。开放性问题请将问题类型选为 text；并查看下方推理摘要。")
        with st.expander("原始返回", expanded=True):
            st.json(answer)
    if reasoning:
        with st.expander("推理摘要", expanded=False):
            st.write(reasoning)
    if pages:
        st.caption(f"相关页码: {', '.join(str(p) for p in pages)}")
    if refs:
        with st.expander("引用来源", expanded=False):
            st.json(refs)


def main():
    st.set_page_config(
        page_title="企业知识库 RAG",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="main-header">
            <h1>企业知识库 RAG 问答系统</h1>
            <p>基于向量检索 + LLM 重排序 + 通义千问 | 企业年报智能问答</p>
            <p>Vector Search + LLM Reranking + Qwen-Flash</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    companies = load_companies()
    if not companies:
        st.error("未找到公司数据，请先配置 data/stock_data/subset.csv")
        st.stop()

    if "retrieval_results" not in st.session_state:
        st.session_state.retrieval_results = None
    if "answer_result" not in st.session_state:
        st.session_state.answer_result = None
    if "e2e_result" not in st.session_state:
        st.session_state.e2e_result = None

    col_left, col_right = st.columns([1, 2.2], gap="large")

    with col_left:
        st.markdown('<div class="panel-title">查询设置</div>', unsafe_allow_html=True)

        company = st.selectbox("选择公司", companies, index=0)
        question = st.text_area(
            "输入问题",
            value="中芯国际在晶圆制造行业中的地位如何？",
            height=100,
            placeholder="请输入您的问题...",
        )

        # 问题或公司变更时清空旧答案，避免误以为仍是当前提问的结果
        query_key = f"{company}||{question.strip()}"
        if st.session_state.get("last_query_key") != query_key:
            st.session_state.last_query_key = query_key
            st.session_state.answer_result = None
            st.session_state.retrieval_results = None

        kind_label = st.radio(
            "问题类型",
            options=["text", "boolean", "number", "name"],
            index=0,
            horizontal=True,
            format_func=lambda x: {
                "text": "text",
                "boolean": "boolean",
                "number": "number",
                "name": "name",
            }[x],
        )
        kind = KIND_MAP[kind_label]
        if kind_label != "text":
            st.caption("提示：开放性问题（如同时问营收和产能利用率）请选 text，number 只适合单一数值。")

        enable_rerank = st.checkbox("启用 LLM 重排序", value=True)
        top_n = st.slider("检索文档数量", min_value=1, max_value=10, value=5)
        if st.button("刷新知识库缓存", use_container_width=True):
            clear_pipeline_cache()
            st.session_state.retrieval_results = None
            st.session_state.answer_result = None
            st.success("已清理 Pipeline 缓存，下次查询将重新加载。")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            search_clicked = st.button("搜索文档", use_container_width=True)
        with btn_col2:
            answer_clicked = st.button("生成答案", type="primary", use_container_width=True)

        st.divider()
        st.markdown('<div class="panel-title">连通性测试</div>', unsafe_allow_html=True)
        st.caption("校验数据、检索、LLM 问答全流程是否打通（默认不启用重排，更快）")
        e2e_clicked = st.button("运行全流程测试", use_container_width=True)

    with col_right:
        st.markdown('<div class="panel-title">检索与回答</div>', unsafe_allow_html=True)

        if not question.strip():
            st.info("请在左侧输入问题后，点击「搜索文档」或「生成答案」。")
            st.stop()

        full_question = build_full_question(company, question)
        # 固定使用 qwen-flash，避免旧进程/缓存仍走 qwen-turbo-latest
        answering_model = "qwen-flash"
        pipeline = get_pipeline(enable_rerank, top_n, answering_model)
        paths = pipeline.paths

        if e2e_clicked:
            with st.spinner("正在运行全流程测试，请稍候..."):
                try:
                    st.session_state.e2e_result = run_frontend_e2e_check(
                        company=company,
                        question=question.strip() or "中芯国际在晶圆制造行业中的地位如何？",
                        answering_model="qwen-flash",
                        top_n=min(top_n, 3),
                    )
                except Exception as e:
                    st.session_state.e2e_result = {
                        "success": False,
                        "steps": [{"step": "异常", "passed": False, "detail": str(e)}],
                        "final_answer": None,
                        "elapsed_sec": 0,
                        "model": "qwen-flash",
                    }

        if st.session_state.e2e_result:
            e2e = st.session_state.e2e_result
            if e2e.get("success"):
                st.success(
                    f"全流程测试通过（模型 `{e2e.get('model')}`，耗时 {e2e.get('elapsed_sec', 0):.2f}s）"
                )
            else:
                st.error(
                    f"全流程测试未通过（模型 `{e2e.get('model')}`，耗时 {e2e.get('elapsed_sec', 0):.2f}s）"
                )
            for item in e2e.get("steps", []):
                if item.get("passed"):
                    st.markdown(f"- 通过 **{item['step']}**：{item.get('detail', '')}")
                else:
                    st.markdown(f"- 失败 **{item['step']}**：{item.get('detail', '')}")
            if e2e.get("final_answer"):
                with st.expander("测试生成的答案", expanded=True):
                    st.write(e2e["final_answer"])
            st.divider()

        if search_clicked:
            st.session_state.answer_result = None
            with st.spinner("正在检索相关文档..."):
                t0 = time.time()
                try:
                    results = search_documents(
                        company=company,
                        question=full_question,
                        enable_rerank=enable_rerank,
                        top_n=top_n,
                        vector_db_dir=paths.vector_db_dir,
                        documents_dir=paths.documents_dir,
                    )
                    elapsed = time.time() - t0
                    st.session_state.retrieval_results = (results, company, question, elapsed)
                except Exception as e:
                    st.error(f"检索失败: {e}")
                    st.session_state.retrieval_results = None

        if answer_clicked:
            st.session_state.answer_result = None
            st.session_state.retrieval_results = None
            with st.spinner("正在检索并生成答案，请稍候..."):
                t0 = time.time()
                try:
                    # 每次生成前清理缓存，确保用上最新知识库与提示词
                    clear_pipeline_cache()
                    pipeline = get_pipeline(enable_rerank, top_n, answering_model)
                    paths = pipeline.paths
                    answer = pipeline.answer_single_question(full_question, kind=kind)
                    elapsed = time.time() - t0
                    st.session_state.answer_result = (answer, elapsed, answering_model)
                    # 同步展示本次检索片段，便于核对是否命中季报/纯文本
                    results = search_documents(
                        company=company,
                        question=full_question,
                        enable_rerank=enable_rerank,
                        top_n=top_n,
                        vector_db_dir=paths.vector_db_dir,
                        documents_dir=paths.documents_dir,
                    )
                    st.session_state.retrieval_results = (
                        results, company, question, elapsed
                    )
                except Exception as e:
                    st.error(f"生成答案失败: {e}")
                    st.session_state.answer_result = None

        if st.session_state.answer_result:
            if len(st.session_state.answer_result) == 3:
                answer, ans_elapsed, model_name = st.session_state.answer_result
            else:
                answer, ans_elapsed = st.session_state.answer_result
                model_name = ""
            render_answer(answer, ans_elapsed, model_name)
            st.divider()

        if st.session_state.retrieval_results:
            results, disp_company, disp_question, elapsed = st.session_state.retrieval_results
            render_retrieval_results(results, disp_company, disp_question, elapsed)
        elif not st.session_state.answer_result:
            st.markdown(
                """
                **使用说明**
                1. 选择公司并输入问题（开放性问题请选 **text**，对应 `kind=string`）
                2. 点击 **搜索文档** 仅查看检索片段
                3. 点击 **生成答案** 调用 RAG 流水线生成回答
                4. 点击左侧 **运行全流程测试** 可一键校验数据/检索/问答是否打通
                """
            )


if __name__ == "__main__":
    main()
