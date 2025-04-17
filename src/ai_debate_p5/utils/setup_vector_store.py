import os
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
file_path = "2023_P5_Report_Single_Pages.pdf"
vector_store_name = "P5_Vector_Store"

# === REST API headers ===
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2"
}

# === Upload file ===
print("📤 Uploading file...")
with open(file_path, "rb") as f:
    response = requests.post(
        "https://api.openai.com/v1/files",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": f},
        data={"purpose": "assistants"}
    )
response.raise_for_status()
file_id = response.json()["id"]
print(f"✅ File uploaded: {file_id}")

# === Create vector store ===
print("📦 Creating vector store...")
response = requests.post(
    "https://api.openai.com/v1/vector_stores",
    headers=headers,
    json={"name": vector_store_name}
)
response.raise_for_status()
vector_store_id = response.json()["id"]
print(f"✅ Vector store created: {vector_store_id}")

# === Attach existing uploaded file by ID ===
print("📎 Attaching file to vector store...")
response = requests.post(
    f"https://api.openai.com/v1/vector_stores/{vector_store_id}/files",
    headers=headers,
    json={"file_id": file_id}  # ✅ Correct: use JSON, not files parameter
)
response.raise_for_status()
print(f"✅ File attached to vector store {vector_store_id}")

# === Output .env variables ===
print("\n🎉 Setup complete! Add the following to your .env file:\n")
print(f"FILE_ID={file_id}")
print(f"VECTOR_STORE_ID={vector_store_id}")
