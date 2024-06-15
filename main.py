import signal
import asyncio
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import service
import util
import os
import json
from bson import json_util
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")
load_dotenv()

app = FastAPI()


@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down")
    asyncio.get_event_loop().close()


client = util.crateMongoConnection(os.getenv("MONGODB_URI"))


@app.get(
    "/",
    tags=["system"],
)
def read_root():
    return {
        "status": "ok",
        "message": "Welcome to the Teleperformance LLM Engine API!",
    }


@app.get(
    "/references",
    tags=["system"],
)
async def test():
    references = await service.get_references(client)
    references = json.loads(json_util.dumps(references))
    return {"references": references}


@app.post(
    "/init",
    tags=["engine"],
    summary="Initialize the search index",
    description="Initialize the search index with the references from the database.",
)
async def init():
    try:
        sources = await service.get_references(client)
        docsearch = service.init_datasource(client, sources)
        return {
            "status": "ok",
            "message": "initialized",
            "index_name": os.getenv("INDEX_NAME"),
            "collection_name": os.getenv("COLLECTION_NAME"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


class AskQuery(BaseModel):
    question: str


@app.post(
    "/ask",
    tags=["engine"],
    summary="Search the index",
    description="Search the index for a query.",
)
async def ask(body: AskQuery):
    try:
        question = body.question
        results = service.gpt_ask(client, question)
        return {"status": "ok", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
