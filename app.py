#!/usr/bin/env python3
"""
Gradio web app for STEM Animation Generator
Simple UI: Choose course from Neo4j -> Generate concept -> Video player
"""

from pathlib import Path

import gradio as gr
from neo4j import GraphDatabase

from pipeline import Pipeline

# ==========================
# 1. KẾT NỐI NEO4J
# ==========================

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "@Danh2004"   # nhớ đổi nếu bạn đổi password

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def get_course_options():
    """
    Lấy danh sách tất cả học phần từ Neo4j
    Trả về list string dạng: 'MAN104 - Quản Lý Dự Án Công Nghệ Thông Tin'
    """
    query = """
    MATCH (c:Course)
    RETURN
      c.code      AS code,
      c.name_vi   AS name_vi
    ORDER BY code
    """
    with driver.session() as session:
        result = session.run(query)
        options = []
        for record in result:
            code = record["code"]
            name_vi = record["name_vi"]
            label = f"{code} - {name_vi}"
            options.append(label)
        return options


def _line(char="-", length=200) -> str:
    return char * length


def _build_course_text(record) -> str:
    """
    Chuyển toàn bộ thông tin course (y như hàm test()) thành 1 chuỗi text dài
    để đưa vào concept cho pipeline.
    """
    course = record["c"]
    blocks          = [x for x in record["blocks"] if x is not None]
    sub_blocks      = [x for x in record["sub_blocks"] if x is not None]
    clos            = [x for x in record["clos"] if x is not None]
    plos            = [x for x in record["plos"] if x is not None]
    topics          = [x for x in record["topics"] if x is not None]
    concepts        = [x for x in record["concepts"] if x is not None]
    topic_methods   = [x for x in record["topic_methods"] if x is not None]
    topic_resources = [x for x in record["topic_resources"] if x is not None]
    course_methods  = [x for x in record["course_methods"] if x is not None]
    course_resources= [x for x in record["course_resources"] if x is not None]
    assessments     = [x for x in record["assessments"] if x is not None]
    prereq_courses  = [x for x in record["prereq_courses"] if x is not None]
    teachers        = [x for x in record["teachers"] if x is not None]

    lines = []

    # ========== COURSE INFO ==========
    lines.append(_line("="))
    lines.append(f"HỌC PHẦN: {course.get('name_vi')} [{course.get('code')}]")
    lines.append(_line("="))
    lines.append(f"Tên tiếng Anh : {course.get('name_en')}")
    lines.append(f"Trình độ      : {course.get('level')}")
    lines.append(
        f"Tổng tín chỉ  : {course.get('total_credits')} "
        f"(Lý thuyết: {course.get('theory_credits')})"
    )
    if blocks:
        lines.append(
            "Khối kiến thức: " + ", ".join(b.get("title") for b in blocks)
        )
    else:
        lines.append("Khối kiến thức: N/A")
    if sub_blocks:
        lines.append(
            "Thuộc khối con: " + ", ".join(sb.get("title") for sb in sub_blocks)
        )
    lines.append("")
    lines.append("Mục tiêu học phần:")
    if course.get("objectives"):
        lines.append(f"  - {course.get('objectives')}")
    else:
        lines.append("  - (Chưa cập nhật)")
    lines.append("")
    lines.append("Tóm tắt nội dung:")
    if course.get("summary"):
        lines.append(f"  - {course.get('summary')}")
    else:
        lines.append("  - (Chưa cập nhật)")
    lines.append("")

    # ========== CLOs ==========
    if clos:
        lines.append(_line())
        lines.append("CLOs (Course Learning Outcomes):")
        lines.append(_line())
        for clo in clos:
            lines.append(
                f"{clo.get('short_id')}: {clo.get('description')} "
                f"(PLO: {clo.get('corresponding_plo')})"
            )
        lines.append("")

    # ========== PLOs ==========
    if plos:
        lines.append(_line())
        lines.append("PLOs / PIs liên quan:")
        lines.append(_line())
        for plo in plos:
            lines.append(f"- {plo.get('id')}")
        lines.append("")

    # ========== TOPICS & CONCEPTS ==========
    if topics:
        lines.append(_line())
        lines.append("CẤU TRÚC MÔN HỌC (Topics & Concepts):")
        lines.append(_line())

        for t in sorted(topics, key=lambda x: x.get("id")):
            t_id    = t.get("id")
            t_title = t.get("title")
            t_short = t.get("short_id")
            th      = t.get("theory_hours", 0.0)
            ph      = t.get("practice_hours", 0.0)

            lines.append(f"{t_short} - {t_title} (LT: {th}h, TH: {ph}h)")

            t_concepts = [
                c for c in concepts
                if c.get("id", "").startswith(t_id)
            ]
            for c in sorted(t_concepts, key=lambda x: x.get("id")):
                lines.append(f"    {c.get('short_id')} {c.get('title')}")
            lines.append("")

    # ========== PHƯƠNG PHÁP GIẢNG DẠY ==========
    if course_methods or topic_methods:
        lines.append(_line())
        lines.append("PHƯƠNG PHÁP GIẢNG DẠY:")
        lines.append(_line())
        if course_methods:
            lines.append("  Ở mức học phần:")
            for m in course_methods:
                lines.append(f"    - {m.get('name')}")
        if topic_methods:
            unique_tm = sorted({m.get("name") for m in topic_methods})
            lines.append("  Ở mức chủ đề (topic):")
            for name in unique_tm:
                lines.append(f"    - {name}")
        lines.append("")

    # ========== TÀI NGUYÊN HỌC TẬP ==========
    if course_resources or topic_resources:
        lines.append(_line())
        lines.append("TÀI NGUYÊN HỌC TẬP:")
        lines.append(_line())
        if course_resources:
            lines.append("  Ở mức học phần:")
            for r in course_resources:
                lines.append(f"    - {r}")
        if topic_resources:
            lines.append("  Ở mức topic:")
            for r in topic_resources:
                lines.append(f"    - {r}")
        lines.append("")

    # ========== ĐÁNH GIÁ ==========
    if assessments:
        lines.append(_line())
        lines.append("ĐÁNH GIÁ HỌC PHẦN:")
        lines.append(_line())
        for a in assessments:
            lines.append(f"- Loại        : {a.get('type')}")
            lines.append(f"  Hình thức   : {a.get('method')}")
            lines.append(f"  Trọng số    : {a.get('weight')}")
            if a.get("evaluation") and a.get("evaluation") != "null":
                lines.append(f"  Cách đánh giá: {a.get('evaluation')}")
            lines.append("")
        # bỏ bớt 1 dòng trống cuối cho đẹp
        if lines and lines[-1] == "":
            lines.pop()

    # ========== HỌC PHẦN TIÊN QUYẾT ==========
    if prereq_courses:
        lines.append("")
        lines.append(_line())
        lines.append("HỌC PHẦN TIÊN QUYẾT:")
        lines.append(_line())
        for pc in prereq_courses:
            # tùy DB của bạn là 'name' hay 'name_vi'
            name = pc.get("name") or pc.get("name_vi") or ""
            lines.append(f"- {pc.get('code')}: {name}")
        lines.append("")

    # ========== GIẢNG VIÊN PHỤ TRÁCH ==========
    if teachers:
        lines.append(_line())
        lines.append("GIẢNG VIÊN PHỤ TRÁCH:")
        lines.append(_line())
        for t in teachers:
            lines.append(f"- {t.get('title')} {t.get('name')}")
            lines.append(f"  Email : {t.get('email')}")
            lines.append(f"  SĐT   : {t.get('phone')}")
            lines.append("")

    lines.append(_line("="))
    lines.append("KẾT THÚC THÔNG TIN HỌC PHẦN")
    lines.append(_line("="))

    return "\n".join(lines)


def build_concept_from_course_code(code: str) -> str | None:
    """
    Lấy full thông tin học phần từ Neo4j (giống hàm test())
    rồi build thành concept natural language cho pipeline.run(...)
    """
    query = """
    MATCH (c:Course {code: $code})
    OPTIONAL MATCH (kb:KnowledgeBlock)-[:CONTAINS]->(c)
    OPTIONAL MATCH (c)-[:PART_OF]->(subKb:KnowledgeBlock)
    OPTIONAL MATCH (c)-[:HAS_CLO]->(clo)
    OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t)
    OPTIONAL MATCH (t)-[:HAS_CONCEPT]->(concept)
    OPTIONAL MATCH (t)-[:USES_METHOD]->(tMethod)
    OPTIONAL MATCH (t)-[:USES_RESOURCE]->(tRes)
    OPTIONAL MATCH (c)-[:USES_METHOD]->(cMethod)
    OPTIONAL MATCH (c)-[:USES_RESOURCE]->(cRes)
    OPTIONAL MATCH (c)-[:HAS_ASSESSMENT]->(ass)
    OPTIONAL MATCH (ass)-[:EVALUATES]->(assClo)
    OPTIONAL MATCH (clo)-[:MEASURED_BY]->(plo)
    OPTIONAL MATCH (c)-[:REQUIRES]->(preCourse)
    OPTIONAL MATCH (c)-[:TAUGHT_BY]->(teacher)
    RETURN
      c,
      collect(DISTINCT kb)        AS blocks,
      collect(DISTINCT subKb)     AS sub_blocks,
      collect(DISTINCT clo)       AS clos,
      collect(DISTINCT plo)       AS plos,
      collect(DISTINCT t)         AS topics,
      collect(DISTINCT concept)   AS concepts,
      collect(DISTINCT tMethod)   AS topic_methods,
      collect(DISTINCT tRes)      AS topic_resources,
      collect(DISTINCT cMethod)   AS course_methods,
      collect(DISTINCT cRes)      AS course_resources,
      collect(DISTINCT ass)       AS assessments,
      collect(DISTINCT assClo)    AS assessed_clos,
      collect(DISTINCT preCourse) AS prereq_courses,
      collect(DISTINCT teacher)   AS teachers
    """
    with driver.session() as session:
        record = session.run(query, code=code).single()

    if not record:
        return None

    course_text = _build_course_text(record)

    # concept cuối cùng: 1 câu chỉ đạo + full syllabus text
    concept = (
        "Tạo một video giáo dục 3–5 phút, bằng tiếng Việt, giới thiệu học phần dưới đây. "
        "Video cần trình bày mục tiêu học phần, chuẩn đầu ra (CLOs), cấu trúc các chủ đề, "
        "phương pháp giảng dạy, hình thức đánh giá và ý nghĩa của môn học đối với sinh viên. "
        "Dựa trên toàn bộ thông tin chi tiết sau đây:\n\n"
        f"{course_text}"
    )

    return concept


# ==========================
# 2. PIPELINE
# ==========================

pipeline = Pipeline()


def generate_animation(
    selected_course: str,
    language: str = "Vietnamese",
    progress=gr.Progress()
):
    """
    Main generation function called by Gradio
    """
    if not selected_course:
        return None

    # Lấy mã môn từ chuỗi, ví dụ: 'MAN104 - ...' -> 'MAN104'
    course_code = selected_course.split("-")[0].strip()

    # Xây dựng concept từ Neo4j
    concept = build_concept_from_course_code(course_code)
    if not concept:
        return None

    def update_progress(message: str, percentage: float):
        progress(percentage, desc=message)

    result = pipeline.run(
        concept,
        progress_callback=update_progress,
        target_language=language
    )

    if result["status"] == "success" and result.get("video_result"):
        video_path = result["video_result"]["output_path"]
        if Path(video_path).exists():
            return video_path
        else:
            return None
    else:
        return None


# ==========================
# 3. GRADIO UI
# ==========================

course_choices = get_course_options()

with gr.Blocks(title="STEMViz") as demo:
    gr.Markdown("# STEMViz – Course to Animation")
    gr.Markdown(
        "Chọn một học phần từ Neo4j, hệ thống sẽ tạo video giáo dục từ dữ liệu chương trình đào tạo."
    )

    with gr.Row():
        with gr.Column():
            course_dropdown = gr.Dropdown(
                choices=course_choices,
                label="Chọn học phần (Course)",
                value=course_choices[0] if course_choices else None,
                interactive=True
            )

            language_dropdown = gr.Dropdown(
                choices=["Vietnamese"],
                value="Vietnamese",
                label="Narration Language"
            )

            generate_btn = gr.Button("Generate Animation", variant="primary")

    with gr.Row():
        video_output = gr.Video(
            label="Generated Animation",
            autoplay=True
        )

    generate_btn.click(
        fn=generate_animation,
        inputs=[course_dropdown, language_dropdown],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
