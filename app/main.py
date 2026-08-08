"""FastAPI entrypoint -- extends the Assignment 1 service (Module 1 / L4)."""
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.routers import health, recommend, skin  # noqa: E402

app = FastAPI(
    title="Skincare Advisor API",
    description="CNN skin analysis + RAG + LoRA/RL post-trained LLM advisor",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(health.router)
app.include_router(skin.router)
app.include_router(recommend.router)


@app.get("/")
def root():
    return {"service": "skincare-advisor", "docs": "/docs"}
