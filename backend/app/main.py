from fastapi import FastAPI

from backend.app.api.routes import router as api_router

app = FastAPI(
    title="Acoustic Anomaly Detection API",
    description="Starter API for first-shot acoustic anomaly detection under domain shift.",
    version="0.1.0",
)

app.include_router(api_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
