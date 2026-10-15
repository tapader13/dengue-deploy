from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import json

app = FastAPI(title="Dengue Prediction API")

templates = Jinja2Templates(directory="templates")


class Patient(BaseModel):
    """Dengue prediction input model — matches best_model.joblib features."""
    
    Gender: Literal["Female", "Male"] = Field(
        ..., description="Patient gender"
    )
    Age: float = Field(
        ..., ge=8.0, le=65.0, description="Age in years (8-65)"
    )
    AreaType: Literal["Developed", "Undeveloped"] = Field(
        ..., description="Area type"
    )
    HouseType: Literal["Building", "Other", "Tinshed"] = Field(
        ..., description="House type"
    )
    Joint_Pain: Literal["None", "Moderate", "Severe"] = Field(
        ..., description="Joint pain severity"
    )
    Headache: Literal[0, 1] = Field(
        ..., description="1 = yes, 0 = no"
    )
    Retro_Orbital_Pain: Literal[0, 1] = Field(
        ..., description="1 = yes, 0 = no"
    )
    Myalgia: Literal[0, 1] = Field(
        ..., description="1 = yes, 0 = no"
    )
    Rash: Literal[0, 1] = Field(
        ..., description="1 = yes, 0 = no"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Gender": "Male",
                "Age": 35.0,
                "AreaType": "Developed",
                "HouseType": "Building",
                "Joint_Pain": "Moderate",
                "Headache": 1,
                "Retro_Orbital_Pain": 0,
                "Myalgia": 1,
                "Rash": 0
            }]
        }
    }

# @app.get("/")
# def read_root():
#     return {"Hello": "World 2"}

with open("best_model.joblib", "rb") as f:
    model = joblib.load(f)

with open("model_meta.json") as f:
    META = json.load(f)    

FEATURE_ORDER = META["feature_order"]
THRESHOLD = META["threshold"] 


def _make_prediction(data: dict) -> dict:
    X = pd.DataFrame([data])[FEATURE_ORDER]
    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= THRESHOLD)

    if proba >= 0.7:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "is_positive": bool(pred),
        "probability": round(proba, 4),
        "probability_pct": f"{proba * 100:.2f}%",
        "risk_level": risk,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None}
    )


@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    Gender: str = Form(...),
    Age: float = Form(...),
    AreaType: str = Form(...),
    HouseType: str = Form(...),
    Joint_Pain: str = Form(...),
    Headache: int = Form(...),
    Retro_Orbital_Pain: int = Form(...),
    Myalgia: int = Form(...),
    Rash: int = Form(...),
):
    data = {
        "Gender": Gender, "Age": Age,
        "AreaType": AreaType, "HouseType": HouseType,
        "Joint_Pain": Joint_Pain,
        "Headache": Headache,
        "Retro_Orbital_Pain": Retro_Orbital_Pain,
        "Myalgia": Myalgia, "Rash": Rash,
    }
    result = _make_prediction(data)

    prediction = {
        "label": "🦟 Dengue Positive" if result["is_positive"] else "✅ Dengue Negative",
        "probability": result["probability_pct"],
        "risk_level": result["risk_level"],
        "color": "#e74c3c" if result["is_positive"] else "#27ae60",
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": prediction}
    )
@app.post("/predict")
async def api_predict(patient: Patient):
    result = _make_prediction(patient.model_dump())
    return {
        "model": "LogisticRegression",
        "input": patient.model_dump(),
        "prediction": "Dengue Positive" if result["is_positive"] else "Dengue Negative",
        "probability": result["probability"],
        "risk_level": result["risk_level"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}