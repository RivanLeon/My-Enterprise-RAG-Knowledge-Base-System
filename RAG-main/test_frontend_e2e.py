"""
前端全流程连通性测试：校验数据、检索、LLM 问答是否打通。

用法（在 RAG-main 目录下）：
    python test_frontend_e2e.py

也可被 app_streamlit.py 导入调用。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pyprojroot import here

PROJECT_ROOT = here()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline, RunConfig
from src.retrieval import VectorRetriever

load_dotenv()

DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"
DEFAULT_MODEL = "qwen-flash"
DEFAULT_QUESTION = "中芯国际在晶圆制造行业中的地位如何？"


def _ok(step: str, detail: str = "") -> dict[str, Any]:
    return {"step": step, "passed": True, "detail": detail}


def _fail(step: str, detail: str) -> dict[str, Any]:
    return {"step": step, "passed": False, "detail": detail}


def run_frontend_e2e_check(
    company: str | None = None,
    question: str = DEFAULT_QUESTION,
    answering_model: str = DEFAULT_MODEL,
    top_n: int = 3,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    跑通与前端一致的问答主路径（不含 LLM 重排，加快测试）。

    返回：
        {
          "success": bool,
          "steps": [...],
          "final_answer": str | None,
          "elapsed_sec": float,
          "model": str,
        }
    """
    load_dotenv()
    root = Path(data_root) if data_root else DATA_ROOT
    steps: list[dict[str, Any]] = []
    t_all = time.time()

    # 1. API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        steps.append(_fail("环境变量", "未检测到 DASHSCOPE_API_KEY（系统环境变量或 .env）"))
        return {
            "success": False,
            "steps": steps,
            "final_answer": None,
            "elapsed_sec": time.time() - t_all,
            "model": answering_model,
        }
    steps.append(_ok("环境变量", f"DASHSCOPE_API_KEY 已设置（长度 {len(api_key)}）"))

    # 2. 数据文件
    subset_path = root / "subset.csv"
    vector_db_dir = root / "databases" / "vector_dbs"
    documents_dir = root / "databases" / "chunked_reports"
    missing = []
    if not subset_path.exists():
        missing.append(str(subset_path))
    if not vector_db_dir.exists():
        missing.append(str(vector_db_dir))
    if not documents_dir.exists():
        missing.append(str(documents_dir))
    faiss_files = list(vector_db_dir.glob("*.faiss")) if vector_db_dir.exists() else []
    doc_files = list(documents_dir.glob("*.json")) if documents_dir.exists() else []
    if missing:
        steps.append(_fail("数据目录", "缺少: " + ", ".join(missing)))
        return {
            "success": False,
            "steps": steps,
            "final_answer": None,
            "elapsed_sec": time.time() - t_all,
            "model": answering_model,
        }
    if not faiss_files or not doc_files:
        steps.append(
            _fail(
                "向量库/分块",
                f"faiss={len(faiss_files)} 个, chunked_json={len(doc_files)} 个，请先跑 pipeline 建库",
            )
        )
        return {
            "success": False,
            "steps": steps,
            "final_answer": None,
            "elapsed_sec": time.time() - t_all,
            "model": answering_model,
        }
    steps.append(
        _ok(
            "数据目录",
            f"subset.csv 存在；faiss={len(faiss_files)}；chunked_json={len(doc_files)}",
        )
    )

    # 3. 公司名
    import pandas as pd

    try:
        df = pd.read_csv(subset_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(subset_path, encoding="gbk")
    companies = df["company_name"].dropna().astype(str).tolist()
    if not companies:
        steps.append(_fail("公司列表", "subset.csv 中无 company_name"))
        return {
            "success": False,
            "steps": steps,
            "final_answer": None,
            "elapsed_sec": time.time() - t_all,
            "model": answering_model,
        }
    company_name = company or companies[0]
    if company_name not in companies:
        steps.append(_fail("公司列表", f"公司「{company_name}」不在 subset.csv 中"))
        return {
            "success": False,
            "steps": steps,
            "final_answer": None,
            "elapsed_sec": time.time() - t_all,
            "model": answering_model,
        }
    steps.append(_ok("公司列表", f"使用公司: {company_name}"))

    # 4. 向量检索
    full_question = question if company_name in question else f"{company_name}：{question}"
    try:
        t0 = time.time()
        retriever = VectorRetriever(vector_db_dir, documents_dir)
        retrieval = retriever.retrieve_by_company_name(
            company_name=company_name,
            query=full_question,
            top_n=top_n,
            return_parent_pages=True,
        )
        t_ret = time.time() - t0
        if not retrieval:
            steps.append(_fail("向量检索", f"未检索到文档（耗时 {t_ret:.2f}s）"))
            return {
                "success": False,
                "steps": steps,
                "final_answer": None,
                "elapsed_sec": time.time() - t_all,
                "model": answering_model,
            }
        preview = (retrieval[0].get("text") or "")[:80].replace("\n", " ")
        steps.append(
            _ok(
                "向量检索",
                f"命中 {len(retrieval)} 条，耗时 {t_ret:.2f}s；首条预览: {preview}...",
            )
        )
    except Exception as e:
        steps.append(_fail("向量检索", str(e)))
        return {
            "success": False,
            "steps": steps,
            "final_answer": None,
            "elapsed_sec": time.time() - t_all,
            "model": answering_model,
        }

    # 5. LLM 生成答案（与前端同路径：answer_single_question）
    try:
        run_config = RunConfig(
            parent_document_retrieval=True,
            llm_reranking=False,
            top_n_retrieval=top_n,
            parallel_requests=1,
            api_provider="dashscope",
            answering_model=answering_model,
            full_context=False,
        )
        pipeline = Pipeline(root, run_config=run_config)
        t1 = time.time()
        answer = pipeline.answer_single_question(full_question, kind="string")
        t_ans = time.time() - t1
        final = answer.get("final_answer") if isinstance(answer, dict) else None
        if final is None or str(final).strip() == "" or str(final).strip() == "N/A":
            steps.append(
                _fail(
                    "LLM 问答",
                    f"模型={answering_model}，耗时 {t_ans:.2f}s，final_answer={final!r}",
                )
            )
            return {
                "success": False,
                "steps": steps,
                "final_answer": final,
                "raw_answer": answer,
                "elapsed_sec": time.time() - t_all,
                "model": answering_model,
            }
        steps.append(
            _ok(
                "LLM 问答",
                f"模型={answering_model}，耗时 {t_ans:.2f}s，答案长度={len(str(final))}",
            )
        )
        return {
            "success": True,
            "steps": steps,
            "final_answer": str(final),
            "raw_answer": answer,
            "elapsed_sec": time.time() - t_all,
            "model": answering_model,
            "question": full_question,
        }
    except Exception as e:
        steps.append(_fail("LLM 问答", str(e)))
        return {
            "success": False,
            "steps": steps,
            "final_answer": None,
            "elapsed_sec": time.time() - t_all,
            "model": answering_model,
        }


def main() -> int:
    print("=" * 60)
    print("前端全流程 E2E 测试")
    print("=" * 60)
    result = run_frontend_e2e_check()
    for item in result["steps"]:
        mark = "[通过]" if item["passed"] else "[失败]"
        print(f"{mark} {item['step']}: {item['detail']}")
    print("-" * 60)
    print(f"模型: {result['model']}")
    print(f"总耗时: {result['elapsed_sec']:.2f}s")
    if result.get("final_answer"):
        print(f"答案预览: {result['final_answer'][:200]}")
    print("=" * 60)
    if result["success"]:
        print("结果: 全流程打通，答案生成正常")
        return 0
    print("结果: 全流程未通过")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
