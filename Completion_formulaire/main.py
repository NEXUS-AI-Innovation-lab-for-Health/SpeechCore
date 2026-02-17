from fastapi import FastAPI, HTTPException
from .schemas import ExtractionRequest, ExtractionResponse
from .extractor import FormExtractor
from .llm_client import LLMClient

app = FastAPI()

llm_client = LLMClient()
extractor = FormExtractor(llm_client)

@app.post("/extract", response_model=ExtractionResponse)
async def extract_form(req: ExtractionRequest):
    try:
        data = await extractor.extract(req.form, req.text)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
