"""Placeholder FastAPI app — DELETE when Niranjan's real API lands.

This exists only so the Dockerfile can build and the healthcheck passes.
Niranjan (P4) will replace this with the full src/api/ package.
"""
from fastapi import FastAPI

app = FastAPI(title="Health Wellness API", version="0.0.1-placeholder")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": "placeholder — waiting for Niranjan's register + export",
        "version": None,
    }
