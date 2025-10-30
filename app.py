import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import config
from model import NewsSummaryModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = config.t5_tokenizer
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "model_checkpoint/t5-best-checkpoint.ckpt")

def load_model():
    model = None
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model = NewsSummaryModel.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
            model.freeze()
            model.model.to(DEVICE)
            return model
        except Exception:
            pass
    model = NewsSummaryModel()
    model.model.to(DEVICE)
    model.freeze()
    return model

MODEL = load_model()
app = FastAPI()

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    status: str = "success"
    summary: str

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required and cannot be empty.")
    enc = TOKENIZER(
        text,
        max_length=config.text_token_max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        generated_ids = MODEL.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.summary_token_max_length,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
    summary = TOKENIZER.decode(
        generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print("Summary: ", summary)
    return SummarizeResponse(summary=summary)

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app:app", host=host, port=port, reload=True)
