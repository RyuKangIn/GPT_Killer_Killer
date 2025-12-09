from fastapi import APIRouter
from app.api.routes import ai

api_router = APIRouter()
api_router.include_router(ai.router, tags=["ai"])