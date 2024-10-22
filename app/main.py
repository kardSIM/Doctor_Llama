import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from load import load_model, generate_rep
class QADataModel(BaseModel):
    question: str

app = FastAPI()

model, tokenizer=load_model()

instruction = """
You are an AI trained to provide accurate health information and interpret scientific research. 
Your goal is to answer medical questions, suggest treatments, offer general health advice based on established guidelines, 
and provide evidence-based insights by analyzing relevant research studies. When necessary, summarize clinical findings to support your advice.
"""


@app.post("/medical_question_answering")
async def qa(input_data: QADataModel):


    query = input_data.question    
    response=generate_rep(query, instruction, model,tokenizer)
    return {"answer": response} 