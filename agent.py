# ============================================================
# Week 10 Lab: ADK Data Agent - UCI ML Repository Query Agent
# ============================================================
# การติดตั้ง (Step 1):
#   pip install google-adk
#   pip install -U ucimlrepo
#
# การรัน (Step 5):
#   adk web --port 8000
#   แล้วเปิด http://localhost:8000 ใน Browser
# ============================================================

from google.adk.agents.llm_agent import Agent

# ----------------------------------------------------------
# Step 2 & 3: Tool Definition - สร้าง Tool สำหรับดึงข้อมูล
# จาก UCI ML Repository
# ----------------------------------------------------------

def get_uci_metadata(dataset_id: int) -> dict:
    """
    สืบค้นข้อมูล Metadata ของชุดข้อมูลจาก UCI ML Repository โดยใช้ ID

    Args:
        dataset_id: รหัส ID ของชุดข้อมูลใน UCI ML Repository
                    เช่น 225 = Diabetes, 53 = Iris, 17 = Breast Cancer Wisconsin

    Returns:
        dict ที่มี name, instances, features, subject_area, task และ summary
    """
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=dataset_id)

    # ดึง variable info ถ้ามี
    try:
        variables = data.variables[['name', 'role', 'type']].to_dict(orient='records')
    except Exception:
        variables = []

    return {
        "name": data.metadata.name,
        "instances": data.metadata.num_instances,
        "features": data.metadata.num_features,
        "subject_area": data.metadata.area,
        "task": data.metadata.task,
        "summary": data.metadata.additional_info.summary,
        "variables_preview": variables[:10]  # แสดงตัวแปรแรก 10 ตัว
    }


def search_uci_datasets(keyword: str) -> dict:
    """
    ค้นหาชุดข้อมูลจาก UCI ML Repository ด้วย keyword

    Args:
        keyword: คำค้นหา เช่น 'diabetes', 'cancer', 'iris'

    Returns:
        dict ที่มีรายการชุดข้อมูลที่เกี่ยวข้อง
    """
    from ucimlrepo import list_available_datasets

    try:
        datasets = list_available_datasets(filter=keyword)
        results = []
        for _, row in datasets.iterrows():
            results.append({
                "id": int(row.get("id", 0)),
                "name": str(row.get("name", "")),
                "area": str(row.get("area", "")),
                "tasks": str(row.get("tasks", "")),
                "num_instances": str(row.get("num_instances", ""))
            })
        return {
            "keyword": keyword,
            "count": len(results),
            "datasets": results[:10]  # จำกัดผลลัพธ์ 10 รายการแรก
        }
    except Exception as e:
        return {"error": str(e), "keyword": keyword}


# ----------------------------------------------------------
# Step 4: Agent Identity & Instructions
# ----------------------------------------------------------

root_agent = Agent(
    name="uci_query_expert",
    model="gemini-2.5-flash",
    description="Expert in querying and analyzing UCI ML Repository datasets.",
    instruction="""คุณคือนักวิทยาศาสตร์ข้อมูล (Data Scientist) ผู้เชี่ยวชาญด้านชุดข้อมูลจาก UCI ML Repository

**ความสามารถของคุณ:**
- ใช้เครื่องมือ `get_uci_metadata` เพื่อดึงข้อมูล Metadata ของชุดข้อมูลจาก ID
- ใช้เครื่องมือ `search_uci_datasets` เพื่อค้นหาชุดข้อมูลด้วย keyword

**แนวทางการตอบ:**
- ตอบเป็นภาษาไทยเสมอ เว้นแต่ผู้ใช้จะขอภาษาอื่น
- นำเสนอผลลัพธ์เป็นตารางหรือ Markdown ที่อ่านง่าย
- อธิบายความหมายของ Metadata แต่ละส่วนให้ผู้ใช้เข้าใจ
- หากผู้ใช้ถามเกี่ยวกับสถิติหรือบริบทของข้อมูล ให้วิเคราะห์จาก Metadata ที่มี

**รูปแบบการตอบ (Output Format):**
ใช้ Markdown ดังนี้:
- **ชื่อชุดข้อมูล**: ...
- **จำนวน Instances**: ...
- **จำนวน Features**: ...
- **ประเภทงาน (Task)**: ...
- **สรุปข้อมูล**: ...

ตัวอย่าง Dataset IDs ที่น่าสนใจ:
- 225 = Diabetes
- 53  = Iris  
- 17  = Breast Cancer Wisconsin
- 186 = Wine Quality
- 45  = Heart Disease
""",
    tools=[get_uci_metadata, search_uci_datasets]
)
