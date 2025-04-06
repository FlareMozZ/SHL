from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import qa
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str

@app.post("/recommend")
async def get_recommendations(query: QueryRequest):
    try:
        result = qa({"query": query.text})
        return json.loads(result['result'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))