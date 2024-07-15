from model import generate_answer, milvus_client, COLLECTION_NAME
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ask")
def ask(data: Data, request: Request):
    question = data.question
    result = generate_answer(
        milvus_client=milvus_client,
        COLLECTION_NAME=COLLECTION_NAME,
        user_question=question)
    return {"answer": result}
