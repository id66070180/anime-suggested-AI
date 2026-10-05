"""Local Streamlit anime search, recommendations, and human assessment."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from anime_engine import ROOT, AnimeEngine, Filters, settings
from model_status import model_ready

st.set_page_config(page_title="Anime Compass", page_icon="✦", layout="wide")


@st.cache_resource
def get_engine(data_mtime, meta_mtime):
    return AnimeEngine()


def display_value(value, suffix=""):
    if value is None or value == "":
        return "ไม่ทราบ"
    if isinstance(value, (float, int)):
        return f"{value:g}{suffix}"
    return str(value) + suffix


def run_search(engine, query, liked, filters, semantic, rerank):
    cfg = settings()
    embedder = reranker = None
    mode = "embedding" if semantic else "tfidf"
    try:
        if semantic:
            meta = engine.load_embeddings()
            if query.strip():
                from ai_models import Embedder
                embedder = Embedder(meta["model"], cfg["device"], cfg["batch_size"], meta["max_length"], cfg["quantize_4bit"],
                                    local_files_only=cfg.get("local_files_only", True))
        initial = engine.search(query, liked, filters, mode, top_k=cfg["candidate_count"] if rerank else 10,
                                candidate_count=cfg["candidate_count"], embedder=embedder)
        if embedder:
            embedder.close()
            embedder = None
        if rerank and initial:
            from ai_models import QwenReranker
            reranker = QwenReranker(cfg["reranker_model"], cfg["device"], 1, 2048, cfg["quantize_4bit"],
                                    local_files_only=cfg.get("local_files_only", True))
            request = query.strip()
            if liked:
                request += "\nFind anime similar to these favorites:\n" + "\n".join(
                    engine.df.iloc[engine.id_to_index[mid]].document for mid in liked)
            docs = [engine.df.iloc[engine.id_to_index[r["mal_id"]]].document for r in initial]
            values = reranker.score(request, docs)
            for result, score in zip(initial, values):
                result["rerank_score"] = float(score)
            initial.sort(key=lambda r: r["rerank_score"], reverse=True)
        return initial[:10]
    finally:
        if embedder:
            embedder.close()
        if reranker:
            reranker.close()


cfg = settings()
data_path = ROOT / cfg["data"]
if not data_path.exists():
    st.error("ยังไม่มี anime_clean.csv กรุณารัน clean_anime.py กับไฟล์ต้นฉบับก่อน")
    st.stop()
meta_path = ROOT / cfg["artifacts"] / "embeddings.meta.json"
engine = get_engine(data_path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns if meta_path.exists() else 0)
df = engine.df

st.markdown("<p style='color:#A69AFF;letter-spacing:3px;font-size:13px'>ANIME COMPASS / ค้นพบเรื่องถัดไป</p>", unsafe_allow_html=True)
st.title("เรื่องที่ใช่ เริ่มจากสิ่งที่คุณชอบ")
st.write("เล่าบรรยากาศหรือเนื้อเรื่องที่อยากดู หรือเลือกอนิเมะที่ชอบเพื่อค้นหาเรื่องคล้ายกัน")

with st.sidebar:
    st.subheader("ปรับให้ตรงใจ")
    types = st.multiselect("ประเภท", sorted(t for t in df.type.unique() if t),
                            default=[t for t in ["TV", "Movie", "OVA", "ONA", "Special", "TV Special"] if t in set(df.type)])
    all_tags = sorted({t for ts in df.tag_list for t in ts})
    required_tags = st.multiselect("ต้องมีทุกแท็กที่เลือก", all_tags)
    excluded_tags = st.multiselect("ไม่เอาแท็กเหล่านี้", all_tags)
    use_year = st.checkbox("จำกัดปีที่ฉาย")
    years = df.year.dropna()
    year_range = st.slider("ช่วงปี", int(years.min()), int(years.max()),
                            (int(years.min()), int(years.max())), disabled=not use_year) if len(years) else (None, None)
    use_episodes = st.checkbox("จำกัดจำนวนตอน")
    max_episodes = st.number_input("จำนวนตอนไม่เกิน", min_value=1, max_value=10000, value=26, disabled=not use_episodes)
    use_score = st.checkbox("กำหนดคะแนนขั้นต่ำ")
    min_score = st.slider("คะแนน MAL ขั้นต่ำ", 0.0, 10.0, 6.0, 0.1, disabled=not use_score)
    statuses = st.multiselect("สถานะ", sorted(s for s in df.status.unique() if s))
    all_studios = sorted({t for value in df.studios for t in value.split("|") if t})
    studios = st.multiselect("สตูดิโอ (ตรงกับอย่างน้อยหนึ่งรายการ)", all_studios)
    st.caption("เมื่อใช้ตัวกรองปี ตอน หรือคะแนน เรื่องที่ไม่ทราบค่านั้นจะไม่ผ่านเงื่อนไข")
    st.divider()
    st.subheader("วิธีค้นหา")
    methods = ["TF-IDF · คำสำคัญ"]
    embedding_ready = False
    embedding_error = ""
    if meta_path.exists():
        try:
            meta = engine.load_embeddings()
            embedding_ready = True
            methods.append("Embedding · ความหมาย")
        except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
            embedding_error = str(exc)
    method = st.selectbox("เลือกวิธี", methods, index=1 if embedding_ready else 0)
    semantic = method.startswith("Embedding")
    if embedding_ready:
        st.caption(f"Embedding: {meta['model']}")
    else:
        st.caption("ยังไม่มี embedding ที่พร้อมใช้ สร้างได้ด้วย notebook สำหรับ Colab")
        if embedding_error:
            st.warning(embedding_error)
    reranker_ready = model_ready(ROOT, cfg["reranker_model"])
    if not reranker_ready:
        st.session_state["use_reranker"] = False
    rerank = st.checkbox("จัดอันดับซ้ำด้วย Qwen Reranker", value=False,
                        key="use_reranker", disabled=not reranker_ready)
    if not reranker_ready:
        st.caption("ยังไม่มีโมเดล reranker บนเครื่อง จึงปิดตัวเลือกนี้ไว้ — TF-IDF ใช้ได้โดยไม่ต้องโหลดโมเดล")
    if rerank:
        st.caption(f"โหลด {cfg['reranker_model']} บน {cfg['device']} เมื่อค้นหา")
    st.caption("TF-IDF จากเรื่องย่ออังกฤษรองรับคำค้นอังกฤษเป็นหลัก; คำค้นไทยควรใช้ embedding หลายภาษา")

filters = Filters(types=types, required_tags=required_tags, excluded_tags=excluded_tags,
                  year_min=year_range[0] if use_year else None, year_max=year_range[1] if use_year else None,
                  max_episodes=int(max_episodes) if use_episodes else None,
                  min_score=min_score if use_score else None, statuses=statuses, studios=studios)

search_tab, evaluate_tab, data_tab = st.tabs(["ค้นหาและแนะนำ", "ประเมินผล", "ข้อมูลระบบ"])
with search_tab:
    with st.form("search_form"):
        query = st.text_area("อยากดูอนิเมะแบบไหน?", placeholder="เช่น A lonely bounty hunter travelling through space with a found family", height=110)
        options = df.mal_id.tolist()
        title_labels = {int(r.mal_id): f"{r.title_english} · {r.title} [{r.mal_id}]" for r in df.itertuples()}
        liked = st.multiselect("เรื่องที่คุณชอบ (เลือกได้หลายเรื่อง)", options, format_func=lambda mid: title_labels[mid])
        submitted = st.form_submit_button("ค้นหาเรื่องที่ใช่", type="primary", use_container_width=True)
    if submitted:
        if not query.strip() and not liked:
            st.warning("ใส่คำอธิบายหรือเลือกเรื่องที่ชอบก่อนครับ")
        else:
            try:
                with st.spinner("กำลังค้นหาอนิเมะ…"):
                    started = time.perf_counter()
                    results = run_search(engine, query, liked, filters, semantic, rerank)
                    elapsed = time.perf_counter() - started
                for state_key in list(st.session_state):
                    if state_key.startswith("explanation_"):
                        del st.session_state[state_key]
                st.session_state["results"] = results
                st.session_state["query_context"] = query or "คล้ายกับ: " + ", ".join(title_labels[mid] for mid in liked)
                st.session_state["method_context"] = method + (" + reranker" if rerank else "")
                st.session_state["elapsed"] = elapsed
            except Exception as exc:
                st.session_state.pop("results", None)
                st.error(f"ค้นหาไม่สำเร็จ: {exc}")
                if rerank:
                    st.info("ตรวจโมเดล reranker: รัน download_models.py reranker ในโฟลเดอร์ชุด 0.6B หรือปิดตัวเลือกจัดอันดับซ้ำ")
                elif semantic:
                    st.info("ตรวจโมเดล embedding: เวกเตอร์จาก Colab ไม่รวมตัวโมเดลสำหรับแปลงคำค้น")
                else:
                    st.info("TF-IDF ไม่ต้องใช้โมเดล Qwen กรุณาตรวจไฟล์ข้อมูลและแคช TF-IDF")
    if "results" in st.session_state:
        results = st.session_state["results"]
        st.caption(f"{st.session_state['method_context']} · {st.session_state['elapsed']:.2f} วินาที · {len(results)} ผลลัพธ์")
        st.write(st.session_state["query_context"])
        if not results:
            st.info("ไม่พบเรื่องที่ตรงกับคำค้นและตัวกรอง ลองลดเงื่อนไข หรือใช้คำค้นอังกฤษสำหรับ TF-IDF")
        else:
            st.download_button("ดาวน์โหลดผลลัพธ์ CSV", pd.DataFrame(results).to_csv(index=False).encode("utf-8-sig"),
                               file_name="anime_recommendations.csv", mime="text/csv")
        for rank, result in enumerate(results, 1):
            with st.container(border=True):
                cover, body = st.columns([1, 5])
                with cover:
                    url = result.get("image_url", "")
                    if url.startswith("https://"):
                        st.image(url, width=135)
                    else:
                        st.caption("ไม่มีภาพปก")
                with body:
                    st.subheader(f"{rank:02d} · {result['title_english']}")
                    if result["title"] != result["title_english"]:
                        st.caption(result["title"])
                    st.write(f"{result['type']} · {display_value(result['episodes'], ' ตอน')} · "
                             f"ปี {display_value(result['year'])} · MAL {display_value(result['score'])}")
                    st.caption(result["tags"].replace("|", " · "))
                    st.write(result["reason"])
                    with st.expander("เรื่องย่อและรายละเอียด"):
                        st.write(result["synopsis"])
                        st.caption(f"สตูดิโอ: {result.get('studios') or 'ไม่ทราบ'} · ระดับอายุ: {result['rating'] or 'ไม่ทราบ'}")
                        st.caption(f"ความคล้าย: {result['similarity']:.4f}" + (
                            f" · Reranker: {result['rerank_score']:.4f}" if result['rerank_score'] is not None else ""))
                    st.link_button("รายละเอียดบน MyAnimeList ↗", result["url"])
                    if st.button("ดูข้อมูลล่าสุดจากแหล่งต้นทาง", key=f"source_{result['mal_id']}"):
                        try:
                            from source_details import fetch_details
                            with st.spinner("กำลังอ่านข้อมูลจากแหล่งต้นทาง…"):
                                st.session_state[f"source_details_{result['mal_id']}"] = fetch_details(result["mal_id"])
                        except Exception as exc:
                            st.warning(f"อ่านข้อมูลล่าสุดไม่ได้: {exc}")
                    live = st.session_state.get(f"source_details_{result['mal_id']}")
                    if live:
                        with st.expander(f"ข้อมูลเสริมล่าสุด · {live['source']}"):
                            st.write(live.get("title"))
                            st.write(live.get("synopsis") or "ไม่มีเรื่องย่อ")
                            st.caption(f"สถานะ: {live.get('status')} · จำนวนตอน: {live.get('episodes')}")
                            st.caption("ข้อมูลเสริมนี้ยังไม่ได้นำไปสร้าง embedding หรือเปลี่ยนผลจัดอันดับ")
                    explanation_ready = model_ready(ROOT, cfg["explanation_model"])
                    if not explanation_ready:
                        st.caption("โมเดลอธิบายยังไม่พร้อม: ดาวน์โหลดให้ครบด้วย download_models.py explanation ในชุด 0.6B")
                    if st.button("ให้โมเดลอธิบายเพิ่มเติม", key=f"explain_{result['mal_id']}",
                                 disabled=not explanation_ready):
                        explainer = None
                        try:
                            from ai_models import LocalExplainer
                            with st.spinner("กำลังโหลดโมเดลและสร้างคำอธิบาย…"):
                                explainer = LocalExplainer(cfg["explanation_model"], cfg["device"], cfg["quantize_4bit"],
                                                          local_files_only=cfg.get("local_files_only", True))
                                explanation = explainer.explain(st.session_state["query_context"], result)
                            st.session_state[f"explanation_{result['mal_id']}"] = explanation
                        except Exception as exc:
                            st.error(f"สร้างคำอธิบายไม่สำเร็จ: {exc}")
                        finally:
                            if explainer:
                                explainer.close()
                    if f"explanation_{result['mal_id']}" in st.session_state:
                        st.info(st.session_state[f"explanation_{result['mal_id']}"])
                        st.caption("คำอธิบายจากโมเดล ควรตรวจสอบกับเรื่องย่อ")

with evaluate_tab:
    st.subheader("ผลลัพธ์ตรงกับคำค้นแค่ไหน?")
    st.write("ค้นหาก่อน แล้วให้คะแนนผลลัพธ์ 0 = ไม่เกี่ยวข้อง, 1 = เกี่ยวข้องบางส่วน, 2 = ตรงมาก")
    current = st.session_state.get("results", [])
    if current:
        st.write(st.session_state["query_context"])
        assessor = st.text_input("รหัสผู้ประเมิน", value="reviewer_1")
        query_id = st.text_input("รหัสคำค้น", value="q001")
        assessment = pd.DataFrame([{"rank": i, "mal_id": r["mal_id"], "title": r["title_english"], "grade": None}
                                    for i, r in enumerate(current, 1)])
        edited = st.data_editor(assessment, hide_index=True, disabled=["rank", "mal_id", "title"],
                    column_config={"grade": st.column_config.NumberColumn("ความเกี่ยวข้อง (0–2)", min_value=0, max_value=2, step=1)},
                    key="human_assessment_" + str(hash((st.session_state["query_context"], tuple(r["mal_id"] for r in current)))))
        grades = pd.to_numeric(edited.grade, errors="coerce")
        valid_grades = grades.notna().all() and grades.between(0, 2).all() and (grades % 1 == 0).all()
        if valid_grades and query_id.strip() and assessor.strip():
            export = edited.assign(query_id=query_id, query=st.session_state["query_context"],
                                   method=st.session_state["method_context"], assessor=assessor)
            st.download_button("ดาวน์โหลดคะแนนประเมิน", export.to_csv(index=False).encode("utf-8-sig"),
                               file_name="human_assessment.csv", mime="text/csv")
        else:
            st.caption("ให้คะแนนจำนวนเต็ม 0–2 ครบทุกเรื่อง และใส่รหัสคำค้น/ผู้ประเมินเพื่อดาวน์โหลด")
    else:
        st.info("ยังไม่มีผลลัพธ์สำหรับประเมิน")
    st.caption("การวัด NDCG และ Recall ต้องมี ground truth ที่ประเมินจากหลายวิธีค้นหา ไม่ใช้เฉพาะผลจากระบบเดียว")

with data_tab:
    st.subheader("ข้อมูลและความพร้อม")
    a, b, c = st.columns(3)
    a.metric("อนิเมะในระบบ", f"{len(df):,}")
    b.metric("แท็ก", len(all_tags))
    c.metric("Embedding", "พร้อม" if embedding_ready else "ยังไม่สร้าง")
    st.write("ดัชนีค้นหาเป็น snapshot ตามเอกสาร เรื่องย่ออาจถูกตัดที่ 1,000 ตัวอักษร; ปุ่มข้อมูลล่าสุดอ่านข้อมูลเสริมจาก Jikan และใช้ AniList สำรอง")
    st.write("เหตุผลเริ่มต้นสร้างจาก metadata; Qwen3.5 เปิดใช้จากปุ่มอธิบายเพิ่มเติม")
    st.caption("ปีคำนวณจากวันเริ่มฉายและใช้ปีต้นฉบับสำรอง วันที่จบที่เติมสำหรับเรื่องหนึ่งตอนเป็นค่าที่อนุมาน")
    missing = [{"column": col, "missing": int(df[col].isna().sum() if pd.api.types.is_numeric_dtype(df[col]) else (df[col] == "").sum())}
               for col in df.columns if col not in {"document", "tag_list"}]
    st.dataframe(pd.DataFrame(missing), hide_index=True)
    report_path = data_path.with_suffix(".report.json")
    if report_path.exists():
        with st.expander("รายงาน cleansing"):
            st.json(json.loads(report_path.read_text(encoding="utf-8")))
    st.link_button("คู่มือโมเดล Qwen Embedding", "https://huggingface.co/" + cfg["embedding_model"])
    st.write({"Embedding model": cfg["embedding_model"],
              "Reranker พร้อม offline": reranker_ready,
              "Explanation พร้อม offline": model_ready(ROOT, cfg["explanation_model"])})
