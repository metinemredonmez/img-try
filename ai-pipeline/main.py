"""
VidCV AI Pipeline Service
Handles CV parsing, video generation, and AI matching
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os

from cv_parser.parser import CVParser
from video_generator.generator import VideoGenerator
from matching_engine.matcher import MatchingEngine
from tts.speech import TTSService

# Initialize services
cv_parser = CVParser()
video_generator = VideoGenerator()
matching_engine = MatchingEngine()
tts_service = TTSService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("🤖 Starting VidCV AI Service...")
    # Initialize models on startup
    await matching_engine.load_models()
    yield
    print("👋 Shutting down VidCV AI Service...")


app = FastAPI(
    title="VidCV AI Pipeline",
    description="AI services for CV parsing, video generation, and matching",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-pipeline"}


# ==================
# CV Parsing
# ==================
@app.post("/api/cv/parse")
async def parse_cv(file_url: str):
    """
    Parse CV from file URL and extract structured data
    """
    try:
        result = await cv_parser.parse(file_url)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cv/generate-script")
async def generate_script(cv_data: dict, language: str = "tr"):
    """
    Generate video script from parsed CV data
    """
    try:
        script = await cv_parser.generate_script(cv_data, language)
        return {"success": True, "script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================
# Video Generation
# ==================
@app.post("/api/video/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    avatar_url: str,
    script: str,
    language: str = "tr",
    provider: str = "heygen"  # heygen, d-id, self-hosted
):
    """
    Generate AI avatar video from script
    """
    try:
        job_id = await video_generator.create_job(
            avatar_url=avatar_url,
            script=script,
            language=language,
            provider=provider
        )
        # Process in background
        background_tasks.add_task(
            video_generator.process_job,
            job_id
        )
        return {"success": True, "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/status/{job_id}")
async def get_video_status(job_id: str):
    """
    Get video generation job status
    """
    try:
        status = await video_generator.get_status(job_id)
        return {"success": True, **status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================
# TTS (Text-to-Speech)
# ==================
@app.post("/api/tts/generate")
async def generate_speech(
    text: str,
    language: str = "tr",
    voice_id: str = None,
    provider: str = "elevenlabs"  # elevenlabs, azure, openai
):
    """
    Generate speech audio from text
    """
    try:
        audio_url = await tts_service.generate(
            text=text,
            language=language,
            voice_id=voice_id,
            provider=provider
        )
        return {"success": True, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================
# Matching Engine
# ==================
@app.post("/api/match/candidate-job")
async def match_candidate_to_job(candidate_data: dict, job_data: dict):
    """
    Calculate AI match score between candidate and job
    """
    try:
        score, details = await matching_engine.match(candidate_data, job_data)
        return {
            "success": True,
            "score": score,
            "details": details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/match/find-candidates")
async def find_matching_candidates(job_data: dict, limit: int = 20):
    """
    Find top matching candidates for a job
    """
    try:
        candidates = await matching_engine.find_candidates(job_data, limit)
        return {"success": True, "candidates": candidates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/match/find-jobs")
async def find_matching_jobs(candidate_data: dict, limit: int = 20):
    """
    Find top matching jobs for a candidate
    """
    try:
        jobs = await matching_engine.find_jobs(candidate_data, limit)
        return {"success": True, "jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
