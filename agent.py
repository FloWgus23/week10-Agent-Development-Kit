import os

from google.adk.agents.llm_agent import Agent
from ucimlrepo import fetch_ucirepo

# --- Step 3: Creating the Tool Contract ---
def get_uci_metadata(dataset_id: int) -> dict:
    """สืบค้นข้อมูล Metadata ของชุดข้อมูลจาก UCI Repo โดยใช้ ID"""
    
    # ดึงข้อมูลจาก UCI Repository
    data = fetch_ucirepo(id=dataset_id)
    
    # ส่งคืนเฉพาะข้อมูลที่จำเป็น
    return {
        "name": data.metadata.name,
        "instances": data.metadata.num_instances,
        "summary": data.metadata.additional_info.summary
    }

# --- Step 4: Agent Identity & Instructions ---
root_agent = Agent(
    name="uci_query_expert",
    model="gemini-1.5-flash", # แนะนำให้ใช้ 1.5 flash หรือ 2.0 flash หาก 2.5 ยังไม่เปิด public ใน SDK
    description="Expert in querying UCI datasets.",
    instruction="คุณคือนักวิทยาศาสตร์ข้อมูล ใช้เครื่องมือ get_uci_metadata เพื่อตอบคำถามเกี่ยวกับสถิติและบริบทของข้อมูล",
    tools=[get_uci_metadata]
)