from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "@Danh2004"

driver = GraphDatabase.driver(uri, auth=(user, password))

def print_line(char="-", length=200):
    print(char * length)

def test():
    query = """
    MATCH (c:Course {code: 'MAN104'})
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
      collect(DISTINCT teacher)   AS teachers;
    """
    with driver.session() as session:
        record = session.run(query).single()
        if not record:
            print("Không tìm thấy học phần cần tìm.")
            return

        course = record["c"]
        blocks         = [x for x in record["blocks"] if x is not None]
        sub_blocks     = [x for x in record["sub_blocks"] if x is not None]
        clos           = [x for x in record["clos"] if x is not None]
        plos           = [x for x in record["plos"] if x is not None]
        topics         = [x for x in record["topics"] if x is not None]
        concepts       = [x for x in record["concepts"] if x is not None]
        topic_methods  = [x for x in record["topic_methods"] if x is not None]
        topic_resources= [x for x in record["topic_resources"] if x is not None]
        course_methods = [x for x in record["course_methods"] if x is not None]
        course_resources = [x for x in record["course_resources"] if x is not None]
        assessments    = [x for x in record["assessments"] if x is not None]
        prereq_courses = [x for x in record["prereq_courses"] if x is not None]
        teachers       = [x for x in record["teachers"] if x is not None]

        # ========== COURSE INFO ==========
        print_line("=")
        print(f"HỌC PHẦN: {course.get('name_vi')} [{course.get('code')}]")
        print_line("=")
        print(f"Tên tiếng Anh : {course.get('name_en')}")
        print(f"Trình độ      : {course.get('level')}")
        print(f"Tổng tín chỉ  : {course.get('total_credits')} (Lý thuyết: {course.get('theory_credits')})")
        print(f"Khối kiến thức: {', '.join(b['title'] for b in blocks) if blocks else 'N/A'}")
        if sub_blocks:
            print(f"Thuộc khối con: {', '.join(sb['title'] for sb in sub_blocks)}")
        print()
        print("Mục tiêu học phần:")
        print(f"  - {course.get('objectives')}")
        print()
        print("Tóm tắt nội dung:")
        print(f"  - {course.get('summary')}")
        print()

        # ========== CLOs ==========
        print_line()
        print("CLOs (Course Learning Outcomes):")
        print_line()
        for clo in clos:
            print(f"{clo.get('short_id')}: {clo.get('description')} (PLO: {clo.get('corresponding_plo')})")
        print()

        # ========== PLOs ==========
        if plos:
            print_line()
            print("PLOs / PIs liên quan:")
            print_line()
            for plo in plos:
                print(f"- {plo.get('id')}")
            print()

        # ========== TOPICS & CONCEPTS ==========
        print_line()
        print("CẤU TRÚC MÔN HỌC (Topics & Concepts):")
        print_line()

        # Group concept theo topic bằng prefix của id
        for t in sorted(topics, key=lambda x: x.get('id')):
            t_id = t.get("id")
            t_title = t.get("title")
            t_short = t.get("short_id")
            th = t.get("theory_hours", 0.0)
            ph = t.get("practice_hours", 0.0)

            print(f"{t_short} - {t_title} (LT: {th}h, TH: {ph}h)")

            # Concepts thuộc topic này
            t_concepts = [
                c for c in concepts
                if c.get("id", "").startswith(t_id)
            ]
            for c in sorted(t_concepts, key=lambda x: x.get("id")):
                print(f"    {c.get('short_id')} {c.get('title')}")
            print()

        # ========== PHƯƠNG PHÁP GIẢNG DẠY ==========
        if course_methods or topic_methods:
            print_line()
            print("PHƯƠNG PHÁP GIẢNG DẠY:")
            print_line()
            if course_methods:
                print("  Ở mức học phần:")
                for m in course_methods:
                    print(f"    - {m.get('name')}")
            if topic_methods:
                # Có thể trùng lặp, dùng set tên để gọn
                unique_tm = sorted({m.get('name') for m in topic_methods})
                print("  Ở mức chủ đề (topic):")
                for name in unique_tm:
                    print(f"    - {name}")
            print()

        # ========== TÀI NGUYÊN HỌC TẬP ==========
        # (hiện tại bạn chưa có USES_RESOURCE nên chắc sẽ rỗng)
        if course_resources or topic_resources:
            print_line()
            print("TÀI NGUYÊN HỌC TẬP:")
            print_line()
            if course_resources:
                print("  Ở mức học phần:")
                for r in course_resources:
                    print(f"    - {r}")
            if topic_resources:
                print("  Ở mức topic:")
                for r in topic_resources:
                    print(f"    - {r}")
            print()

        # ========== ĐÁNH GIÁ ==========
        if assessments:
            print_line()
            print("ĐÁNH GIÁ HỌC PHẦN:")
            print_line()
            for a in assessments:
                print(f"- Loại        : {a.get('type')}")
                print(f"  Hình thức   : {a.get('method')}")
                print(f"  Trọng số    : {a.get('weight')}")
                if a.get("evaluation") and a.get("evaluation") != "null":
                    print(f"  Cách đánh giá: {a.get('evaluation')}")
                print()
        
        # ========== HỌC PHẦN TIÊN QUYẾT ==========
        if prereq_courses:
            print_line()
            print("HỌC PHẦN TIÊN QUYẾT:")
            print_line()
            for pc in prereq_courses:
                print(f"- {pc.get('code')}: {pc.get('name')}")
            print()

        # ========== GIẢNG VIÊN PHỤ TRÁCH ==========
        if teachers:
            print_line()
            print("GIẢNG VIÊN PHỤ TRÁCH:")
            print_line()
            for t in teachers:
                print(f"- {t.get('title')} {t.get('name')}")
                print(f"  Email : {t.get('email')}")
                print(f"  SĐT   : {t.get('phone')}")
                print()

        print_line("=")
        print("KẾT THÚC THÔNG TIN HỌC PHẦN")
        print_line("=")


if __name__ == "__main__":
    test()
