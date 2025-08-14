import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import requests
import base64
import os
import pandas as pd
import io
from PIL import Image
from transformers import pipeline
app = FastAPI(
    title="Drug Interaction and Dosage API",
    description="An API to analyze drug interactions, provide dosage recommendations, extract drug info, and describe images."
)

# ===== DRUG DATA =====
DRUG_DATA = {
    "ibuprofen": {
        "alternatives": ["naproxen", "acetaminophen"],
        "interactions": ["warfarin", "aspirin"],
        "dosages": {
            "child": "10 mg/kg every 6-8 hours",
            "adult": "200-400 mg every 4-6 hours",
            "elderly": "200 mg every 6 hours (with caution)"
        }
    },
    "dolo-650": {
        "alternatives": ["Crocin 650 mg", "Dolopar 650 mg"],
        "interactions": [
            "chloramphenicol", "lamotrigine", "aspirin",
            "anticonvulsants", "liver disease", "Gilbert’s syndrome"
        ],
        "dosages": {
            "child": "Not recommended under 12 years",
            "adult": "650 mg (1 tablet) every 4-6 hours as needed; maximum 4 doses daily",
            "elderly": "Use with caution—likely same dosing with medical supervision"
        }
    },
    "eldoper": {
        "alternatives": ["Imodium 2 mg", "Loopra 2 mg"],
        "interactions": ["Alcohol", "Domperidone", "acute ulcerative colitis", "dysentery", "pseudomembranous colitis"],
        "dosages": {
            "child": {
                "2–5 years": "approx. 3 mg/day (~1.5 capsules)",
                "6–8 years": "approx. 4 mg/day (2 capsules)",
                "8–12 years": "approx. 6 mg/day (3 capsules)",
                "under 2 years": "Not recommended"
            },
            "adult": "4 mg after first loose stool, then 2 mg after each unformed stool; max 16 mg/day; stop within 48 hours for acute cases",
            "elderly": "Same as adult, with medical supervision"
        }
    },
    "b complex": {
        "alternatives": ["Becozyme C Forte", "Neurobion Forte", "Becosules"],
        "interactions": ["Levodopa", "Chloramphenicol", "Alcohol (excessive use)"],
        "dosages": {
            "child": "As directed by physician; typical dose 1 capsule/tablet daily",
            "adult": "1 capsule/tablet daily or as prescribed",
            "elderly": "Same as adult dose; monitor for absorption issues"
        }
    },
    "telmikind 80": {
        "alternatives": ["Micardis 80 mg", "Telma 80 mg"],
        "interactions": ["Alcohol", "Potassium supplements or potassium-sparing diuretics", "Insulin and other antidiabetic medications"],
        "dosages": {
            "child": "Not recommended under 18 years",
            "adult": "Typical antihypertensive dose: 80 mg once daily; can also be used at lower doses (20–40 mg) depending on response",
            "elderly": "Same as adult, with medical supervision; monitor kidney function, BP, and potassium levels"
        }
    },
    "warfarin": {
        "alternatives": ["dabigatran", "rivaroxaban"],
        "interactions": ["ibuprofen", "aspirin", "omeprazole"],
        "dosages": {
            "child": "Initial dose based on age/weight; requires strict monitoring",
            "adult": "2-5 mg per day, adjusted based on INR",
            "elderly": "Lower initial dose; requires strict monitoring"
        }
    },
    "omeprazole": {
        "alternatives": ["esomeprazole", "pantoprazole"],
        "interactions": ["warfarin", "diazepam"],
        "dosages": {
            "child": "10-20 mg once daily",
            "adult": "20-40 mg once daily",
            "elderly": "20 mg once daily"
        }
    },
    "aspirin": {
        "alternatives": ["ibuprofen", "acetaminophen"],
        "interactions": ["warfarin", "ibuprofen"],
        "dosages": {
            "child": "Not recommended due to Reye's syndrome risk",
            "adult": "325-650 mg every 4 hours",
            "elderly": "75-100 mg once daily (low-dose for cardiovascular protection)"
        }
    }
}

# ===== Normalization Helpers =====
def normalize_drug_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace(" ", "")

DRUG_LOOKUP = {normalize_drug_name(k): k for k in DRUG_DATA.keys()}

def find_drug_key(drug_name: str):
    return DRUG_LOOKUP.get(normalize_drug_name(drug_name))

# ===== HuggingFace API Settings =====
HF_API_KEY = os.getenv("HF_API_KEY", "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")  # <-- Replace or set env var
HF_API_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-vision-3.3-2b"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Load dataset if available
try:
    dataset = pd.read_csv("my_dataset.csv")
except FileNotFoundError:
    dataset = pd.DataFrame()

# ===== Pydantic Models =====
class DrugInteractionRequest(BaseModel):
    drugs: List[str]

class DosageRequest(BaseModel):
    drug: str
    age: int

class AlternativeRequest(BaseModel):
    drug: str

class NlpRequest(BaseModel):
    text: str

# ===== Helper =====
def get_age_category(age: int) -> str:
    if age < 18:
        return "child"
    elif age >= 65:
        return "elderly"
    else:
        return "adult"
fallback_pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
# ===== API Endpoints =====
@app.post("/check_interactions")
def check_interactions(request: DrugInteractionRequest):
    interactions = []
    drug_keys = [find_drug_key(d) for d in request.drugs if find_drug_key(d)]

    for i, drug1 in enumerate(drug_keys):
        for drug2 in drug_keys[i+1:]:
            if drug2 in DRUG_DATA[drug1]["interactions"]:
                interactions.append(f"Harmful interaction between {drug1} and {drug2}.")

    if not interactions:
        return {"status": "success", "message": "No harmful interactions detected."}
    else:
        return {"status": "warning", "interactions": interactions}

@app.post("/recommend_dosage")
def recommend_dosage(request: DosageRequest):
    drug_key = find_drug_key(request.drug)
    age_category = get_age_category(request.age)

    if drug_key:
        dosage = DRUG_DATA[drug_key]["dosages"].get(age_category, "Dosage information not available for this age group.")
        return {"drug": drug_key, "age_category": age_category, "recommendation": dosage}
    else:
        return {"drug": request.drug, "recommendation": "Drug not found in database."}

@app.post("/suggest_alternatives")
def suggest_alternatives(request: AlternativeRequest):
    drug_key = find_drug_key(request.drug)
    if drug_key:
        alternatives = DRUG_DATA[drug_key].get("alternatives", [])
        return {"drug": drug_key, "alternatives": alternatives}
    else:
        return {"drug": request.drug, "message": "Drug not found in database."}

@app.post("/extract_info")
def extract_info(request: NlpRequest):
    return {"status": "error", "message": "NLP model not loaded in API mode."}
from difflib import SequenceMatcher
def fuzzy_match(text, keyword, threshold=0.7):
    """Return True if two strings are similar enough."""
    return SequenceMatcher(None, text.lower(), keyword.lower()).ratio() >= threshold
    
@app.post("/image_to_text")
async def image_to_text(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # Try Granite Vision API first
        payload = {
            "inputs": {
                "image": encoded_image,
                "parameters": {
                    "prompt": "Describe this image succinctly and clearly."
                }
            }
        }
        try:
            response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                caption = result[0].get("generated_text", "").strip()
            else:
                caption = ""
        except Exception:
            caption = ""

        # If Granite Vision failed, use fallback
        if not caption:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = fallback_pipe(image)
            caption = result[0]["generated_text"].strip() if result else "No description generated."

        # ==== NEW: Match caption to DRUG_DATA ====
        matched_drug_info = None
        matched_drug_name = None
        for drug_key in DRUG_DATA.keys():
            if drug_key.lower() in caption.lower():
                matched_drug_info = DRUG_DATA[drug_key]
                matched_drug_name = drug_key
                break

        # Build response
        response_data = {
            "status": "success",
            "caption": caption,
            "matches": []
        }

        # Add drug details if matched
        if matched_drug_info:
            response_data["matches"].append({
                "drug": matched_drug_name,
                "alternatives": matched_drug_info.get("alternatives", []),
                "interactions": matched_drug_info.get("interactions", []),
                "dosages": matched_drug_info.get("dosages", {})
            })

        return response_data

    except Exception as e:
        return {"status": "error", "message": str(e)}


        

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
