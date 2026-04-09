from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.get("/status")
def api_status() -> dict[str, str]:
    return {"message": "API router is ready"}


@router.post("/analyze")
def analyze_placeholder() -> dict[str, str]:
    return {"message": "Inference endpoint placeholder"}
