"""
Video Generator Module
Creates AI avatar videos from scripts
"""
import os
import uuid
import asyncio
from typing import Dict, Any, Optional
from enum import Enum
import httpx
import redis.asyncio as redis


class VideoProvider(str, Enum):
    HEYGEN = "heygen"
    DID = "d-id"
    SELF_HOSTED = "self-hosted"


class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoGenerator:
    """Generate AI avatar videos"""

    def __init__(self):
        self.heygen_api_key = os.getenv("HEYGEN_API_KEY")
        self.did_api_key = os.getenv("DID_API_KEY")
        self.redis_client = None

    async def _get_redis(self):
        if not self.redis_client:
            self.redis_client = redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/2")
            )
        return self.redis_client

    async def create_job(
        self,
        avatar_url: str,
        script: str,
        language: str = "tr",
        provider: str = "heygen"
    ) -> str:
        """Create a new video generation job"""
        job_id = str(uuid.uuid4())

        redis_client = await self._get_redis()
        await redis_client.hset(f"video_job:{job_id}", mapping={
            "status": VideoStatus.PENDING.value,
            "avatar_url": avatar_url,
            "script": script,
            "language": language,
            "provider": provider,
            "progress": "0",
            "video_url": "",
            "error": ""
        })
        await redis_client.expire(f"video_job:{job_id}", 86400)  # 24h TTL

        return job_id

    async def process_job(self, job_id: str):
        """Process video generation job"""
        redis_client = await self._get_redis()

        # Get job data
        job_data = await redis_client.hgetall(f"video_job:{job_id}")
        if not job_data:
            return

        # Decode bytes to strings
        job_data = {k.decode(): v.decode() for k, v in job_data.items()}

        # Update status to processing
        await redis_client.hset(f"video_job:{job_id}", "status", VideoStatus.PROCESSING.value)

        try:
            provider = job_data.get("provider", "heygen")

            if provider == VideoProvider.HEYGEN.value:
                video_url = await self._generate_heygen(
                    avatar_url=job_data["avatar_url"],
                    script=job_data["script"],
                    language=job_data["language"],
                    job_id=job_id
                )
            elif provider == VideoProvider.DID.value:
                video_url = await self._generate_did(
                    avatar_url=job_data["avatar_url"],
                    script=job_data["script"],
                    language=job_data["language"],
                    job_id=job_id
                )
            else:
                video_url = await self._generate_self_hosted(
                    avatar_url=job_data["avatar_url"],
                    script=job_data["script"],
                    language=job_data["language"],
                    job_id=job_id
                )

            # Update job with result
            await redis_client.hset(f"video_job:{job_id}", mapping={
                "status": VideoStatus.COMPLETED.value,
                "video_url": video_url,
                "progress": "100"
            })

        except Exception as e:
            await redis_client.hset(f"video_job:{job_id}", mapping={
                "status": VideoStatus.FAILED.value,
                "error": str(e)
            })

    async def _generate_heygen(
        self,
        avatar_url: str,
        script: str,
        language: str,
        job_id: str
    ) -> str:
        """Generate video using HeyGen API"""
        async with httpx.AsyncClient() as client:
            # Create video
            response = await client.post(
                "https://api.heygen.com/v2/video/generate",
                headers={
                    "X-Api-Key": self.heygen_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "video_inputs": [{
                        "character": {
                            "type": "photo",
                            "photo_url": avatar_url
                        },
                        "voice": {
                            "type": "text",
                            "input_text": script,
                            "voice_id": self._get_heygen_voice(language)
                        }
                    }],
                    "dimension": {
                        "width": 1280,
                        "height": 720
                    }
                },
                timeout=60.0
            )

            if response.status_code != 200:
                raise Exception(f"HeyGen API error: {response.text}")

            result = response.json()
            video_id = result.get("data", {}).get("video_id")

            # Poll for completion
            return await self._poll_heygen_status(video_id, job_id)

    async def _poll_heygen_status(self, video_id: str, job_id: str) -> str:
        """Poll HeyGen for video completion"""
        redis_client = await self._get_redis()

        async with httpx.AsyncClient() as client:
            for _ in range(60):  # Max 5 minutes
                response = await client.get(
                    f"https://api.heygen.com/v1/video_status.get?video_id={video_id}",
                    headers={"X-Api-Key": self.heygen_api_key},
                    timeout=30.0
                )

                result = response.json()
                status = result.get("data", {}).get("status")

                if status == "completed":
                    return result.get("data", {}).get("video_url")
                elif status == "failed":
                    raise Exception("HeyGen video generation failed")

                # Update progress
                progress = result.get("data", {}).get("progress", 0)
                await redis_client.hset(f"video_job:{job_id}", "progress", str(progress))

                await asyncio.sleep(5)

        raise Exception("Video generation timeout")

    async def _generate_did(
        self,
        avatar_url: str,
        script: str,
        language: str,
        job_id: str
    ) -> str:
        """Generate video using D-ID API"""
        async with httpx.AsyncClient() as client:
            # Create talk
            response = await client.post(
                "https://api.d-id.com/talks",
                headers={
                    "Authorization": f"Basic {self.did_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "source_url": avatar_url,
                    "script": {
                        "type": "text",
                        "input": script,
                        "provider": {
                            "type": "microsoft",
                            "voice_id": self._get_did_voice(language)
                        }
                    }
                },
                timeout=60.0
            )

            if response.status_code not in [200, 201]:
                raise Exception(f"D-ID API error: {response.text}")

            result = response.json()
            talk_id = result.get("id")

            # Poll for completion
            return await self._poll_did_status(talk_id, job_id)

    async def _poll_did_status(self, talk_id: str, job_id: str) -> str:
        """Poll D-ID for video completion"""
        redis_client = await self._get_redis()

        async with httpx.AsyncClient() as client:
            for _ in range(60):  # Max 5 minutes
                response = await client.get(
                    f"https://api.d-id.com/talks/{talk_id}",
                    headers={"Authorization": f"Basic {self.did_api_key}"},
                    timeout=30.0
                )

                result = response.json()
                status = result.get("status")

                if status == "done":
                    return result.get("result_url")
                elif status == "error":
                    raise Exception(f"D-ID error: {result.get('error')}")

                await asyncio.sleep(5)

        raise Exception("Video generation timeout")

    async def _generate_self_hosted(
        self,
        avatar_url: str,
        script: str,
        language: str,
        job_id: str
    ) -> str:
        """Generate video using self-hosted models (SadTalker/LivePortrait)"""
        # TODO: Implement self-hosted video generation
        raise NotImplementedError("Self-hosted video generation not yet implemented")

    def _get_heygen_voice(self, language: str) -> str:
        """Get HeyGen voice ID for language"""
        voices = {
            "tr": "tr-TR-AhmetNeural",
            "en": "en-US-AriaNeural",
            "de": "de-DE-KatjaNeural",
            "ar": "ar-SA-HamedNeural"
        }
        return voices.get(language, voices["en"])

    def _get_did_voice(self, language: str) -> str:
        """Get D-ID voice ID for language"""
        voices = {
            "tr": "tr-TR-AhmetNeural",
            "en": "en-US-JennyNeural",
            "de": "de-DE-KatjaNeural",
            "ar": "ar-SA-HamedNeural"
        }
        return voices.get(language, voices["en"])

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        redis_client = await self._get_redis()
        job_data = await redis_client.hgetall(f"video_job:{job_id}")

        if not job_data:
            return {"status": "not_found"}

        return {k.decode(): v.decode() for k, v in job_data.items()}
