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


def build_concept_from_course_code(code: str, language: str = "Vietnamese") -> str | None:
    """
    Lấy full thông tin học phần từ Neo4j (giống hàm test())
    rồi build thành concept natural language cho pipeline.run(...)
    
    Args:
        code: Mã học phần (ví dụ: 'MAN104')
        language: Ngôn ngữ cho video ('Vietnamese' hoặc 'English')
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

    # Build concept header based on language
    if language == "English":
        concept_header = (
    "Create a 3–5 minute educational video in English introducing the course "
    "STRICTLY based on the backend data provided below.\n\n"

    "MANDATORY REQUIREMENTS:\n"
    "- The video MUST be divided into 7 to 10 scenes (no fewer than 7 scenes).\n"
    "- Each scene must focus on ONE clear group of information only.\n"
    "- DO NOT omit any information if it exists in the backend data.\n"
    "- DO NOT invent or assume information that is not provided.\n"
    "- Use short bullet-style phrases. Avoid long paragraphs.\n\n"

    "REQUIRED SCENE STRUCTURE:\n"
    "Scene 1: Course overview (course name, English name, course code, level, credits, knowledge block).\n"
    "Scene 2: Course objectives.\n"
    "Scene 3: Course Learning Outcomes (CLOs) and related PLOs / PIs.\n"
    "Scene 4: Course content summary.\n"
    "Scene 5: Course structure – main topics and lessons.\n"
    "Scene 6: Teaching methods.\n"
    "Scene 7: Assessment methods and grading weights.\n"
    "Scene 8: Prerequisite course(s).\n"
    "Scene 9 (if applicable): Lecturer(s) in charge.\n"
    "Scene 10 (if needed): Course value, skills gained, and career relevance.\n\n"

    "CONTENT GENERATION GUIDELINES:\n"
    "- If a scene contains many items (e.g., CLOs or Topics), summarize them clearly "
    "while preserving the core meaning from the backend.\n"
    "- Prefer concise bullet points.\n"
    "- Do NOT merge multiple scenes into one.\n\n"

    "The detailed backend course information is provided below:\n\n"
)

    else:
        # Default to Vietnamese
        concept_header = (
    "Tạo một video giáo dục dài 3–5 phút, bằng tiếng Việt, giới thiệu học phần dựa HOÀN TOÀN "
    "trên dữ liệu backend được cung cấp bên dưới.\n\n"

    "YÊU CẦU BẮT BUỘC:\n"
    "- Video PHẢI được chia thành từ 7 đến 10 scene (không ít hơn 7).\n"
    "- Mỗi scene trình bày MỘT nhóm thông tin rõ ràng.\n"
    "- KHÔNG được bỏ sót bất kỳ thông tin nào nếu backend có cung cấp.\n"
    "- Không tự bịa thêm nội dung ngoài dữ liệu backend.\n"
    "- Văn bản ngắn gọn, gạch đầu dòng, tránh đoạn văn dài.\n\n"

    "CẤU TRÚC SCENE BẮT BUỘC:\n"
    "Scene 1: Thông tin tổng quan học phần (tên, tên tiếng Anh, mã, tín chỉ, trình độ, khối kiến thức).\n"
    "Scene 2: Mục tiêu học phần.\n"
    "Scene 3: Chuẩn đầu ra học phần (CLOs) và PLO/PI liên quan.\n"
    "Scene 4: Tóm tắt nội dung học phần.\n"
    "Scene 5: Cấu trúc môn học – các bài học/chủ đề chính.\n"
    "Scene 6: Phương pháp giảng dạy.\n"
    "Scene 7: Hình thức và trọng số đánh giá học phần.\n"
    "Scene 8: Học phần tiên quyết.\n"
    "Scene 9 (nếu còn nội dung): Giảng viên phụ trách.\n"
    "Scene 10 (nếu cần): Ý nghĩa học phần và kỹ năng sinh viên đạt được.\n\n"

    "LƯU Ý KHI SINH NỘI DUNG:\n"
    "- Nếu một scene có nhiều dữ liệu (ví dụ CLOs hoặc Topics), hãy trình bày chọn lọc, "
    "nhưng vẫn phản ánh đầy đủ ý chính từ backend.\n"
    "- Ưu tiên liệt kê theo bullet.\n"
    "- Không gộp nhiều scene thành một.\n\n"

    "Dữ liệu chi tiết học phần như sau:\n\n"
)


    concept = concept_header + course_text

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

    # Xây dựng concept từ Neo4j với ngôn ngữ đã chọn
    concept = build_concept_from_course_code(course_code, language)
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
# 3. GRADIO UI – HUTECH STYLE
# ==========================

custom_css = """
body {
    background-color: #F4F6F9;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
}

.hutech-header {
    background: linear-gradient(90deg, #0054A6, #003F7D);
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.hutech-header h1 {
    margin-bottom: 5px;
    font-size: 32px;
}

.hutech-header p {
    font-size: 16px;
    opacity: 0.9;
}

.hutech-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.hutech-step {
    color: #0054A6;
    font-weight: 600;
    margin-bottom: 10px;
}

button.primary {
    background-color: #0054A6 !important;
    border-radius: 8px !important;
    font-size: 16px !important;
}

button.primary:hover {
    background-color: #003F7D !important;
}

.footer-note {
    text-align: center;
    font-size: 14px;
    color: #6B7280;
    margin-top: 20px;
}
"""

course_choices = get_course_options()

with gr.Blocks(
    title="STEMViz – HUTECH",
    css=custom_css
) as demo:

    # ===== HEADER =====
    gr.HTML("""
    <div class="hutech-header">
        <h1>🎓 STEMViz – HUTECH</h1>
        <p>Hệ thống tạo video bài giảng tự động từ chương trình đào tạo</p>
    </div>
    """)

    with gr.Row():
        # ===== LEFT COLUMN =====
        with gr.Column(scale=1):
            gr.Markdown("### 📘 Thông tin học phần")
            with gr.Group(elem_classes="hutech-card"):
                gr.Markdown("<div class='hutech-step'>Bước 1: Chọn học phần</div>")
                course_dropdown = gr.Dropdown(
                    choices=course_choices,
                    label="Học phần",
                    value=course_choices[0] if course_choices else None
                )

                language_dropdown = gr.Dropdown(
    choices=["Vietnamese", "English"],
    value="Vietnamese",
    label="Ngôn ngữ thuyết minh"
)


                generate_btn = gr.Button(
                    "🎬 Tạo video bài giảng",
                    variant="primary"
                )

        # ===== RIGHT COLUMN =====
        with gr.Column(scale=1):
            gr.Markdown("### 📺 Video bài giảng")
            with gr.Group(elem_classes="hutech-card"):
                video_output = gr.Video(
                    label="Video học tập",
                    autoplay=True
                )

    # ===== FOOTER =====
    gr.HTML("""
    <div class="footer-note">
        © 2025 HUTECH – Trường Đại học Công nghệ TP.HCM<br>
        Ứng dụng AI trong giáo dục STEM
    </div>
    """)

    generate_btn.click(
        fn=generate_animation,
        inputs=[course_dropdown, language_dropdown],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)

