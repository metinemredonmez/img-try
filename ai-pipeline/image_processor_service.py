"""
Image Processor Standalone Service
==================================

FastAPI service for CV image processing.
Runs as a separate microservice for scalability.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import router
from image_processor.api import router as image_processor_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Image Processor Service starting...")
    print(f"OCR Provider: {os.getenv('OCR_PROVIDER', 'tesseract')}")
    print(f"Languages: {os.getenv('OCR_LANGUAGES', 'eng+tur')}")

    yield

    # Shutdown
    print("Image Processor Service shutting down...")


# Create FastAPI app
app = FastAPI(
    title="VidCV Image Processor",
    description="CV Document Image Processing Service - OCR, Layout Analysis, Text Extraction",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(image_processor_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "VidCV Image Processor",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/image-processor/health",
            "process": "/image-processor/process",
            "ocr": "/image-processor/ocr",
            "layout": "/image-processor/layout",
            "batch": "/image-processor/batch",
        }
    }


@app.get("/health")
async def health():
    """Service health check."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "image_processor_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8002)),
        reload=os.getenv("ENV", "production") == "development",
        workers=int(os.getenv("MAX_WORKERS", 2)),
    )
