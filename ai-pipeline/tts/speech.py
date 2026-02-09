"""
Text-to-Speech Service
Generates audio from text using various providers
"""
import os
import uuid
from typing import Optional
import httpx
from openai import AsyncOpenAI


class TTSService:
    """Text-to-Speech service with multiple providers"""

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")

    async def generate(
        self,
        text: str,
        language: str = "tr",
        voice_id: Optional[str] = None,
        provider: str = "elevenlabs"
    ) -> str:
        """
        Generate speech audio from text
        Returns URL to audio file
        """
        if provider == "elevenlabs":
            return await self._generate_elevenlabs(text, language, voice_id)
        elif provider == "openai":
            return await self._generate_openai(text, language, voice_id)
        elif provider == "azure":
            return await self._generate_azure(text, language, voice_id)
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")

    async def _generate_elevenlabs(
        self,
        text: str,
        language: str,
        voice_id: Optional[str]
    ) -> str:
        """Generate audio using ElevenLabs"""
        if not voice_id:
            voice_id = self._get_elevenlabs_voice(language)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self.elevenlabs_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                },
                timeout=60.0
            )

            if response.status_code != 200:
                raise Exception(f"ElevenLabs error: {response.text}")

            # Save audio to MinIO and return URL
            audio_content = response.content
            return await self._save_audio(audio_content, "mp3")

    async def _generate_openai(
        self,
        text: str,
        language: str,
        voice_id: Optional[str]
    ) -> str:
        """Generate audio using OpenAI TTS"""
        voice = voice_id or self._get_openai_voice(language)

        response = await self.openai_client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text
        )

        audio_content = response.content
        return await self._save_audio(audio_content, "mp3")

    async def _generate_azure(
        self,
        text: str,
        language: str,
        voice_id: Optional[str]
    ) -> str:
        """Generate audio using Azure TTS"""
        # TODO: Implement Azure TTS
        raise NotImplementedError("Azure TTS not yet implemented")

    async def _save_audio(self, content: bytes, extension: str) -> str:
        """Save audio to MinIO and return URL"""
        # TODO: Implement MinIO upload
        # For now, return placeholder
        filename = f"audio/{uuid.uuid4()}.{extension}"
        return f"http://{self.minio_endpoint}/vidcv-audio/{filename}"

    def _get_elevenlabs_voice(self, language: str) -> str:
        """Get ElevenLabs voice ID for language"""
        voices = {
            "tr": "pMsXgVXv3BLzUgSXRplE",  # Turkish male
            "en": "EXAVITQu4vr4xnSDxMaL",  # English - Sarah
            "de": "pqHfZKP75CvOlQylNhV4",  # German
            "ar": "ODq5zmih8GrVes37Dizd"   # Arabic
        }
        return voices.get(language, voices["en"])

    def _get_openai_voice(self, language: str) -> str:
        """Get OpenAI voice for language"""
        # OpenAI voices are multilingual
        return "nova"  # or alloy, echo, fable, onyx, shimmer
