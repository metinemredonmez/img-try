"""
LangChain Chains for VidCV
"""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from config import get_llm


# ===================
# Output Schemas
# ===================
class CVData(BaseModel):
    """Structured CV data schema"""
    personal_info: Dict[str, Any] = Field(description="Personal information")
    summary: str = Field(description="Professional summary")
    experience: list = Field(description="Work experience list")
    education: list = Field(description="Education list")
    skills: Dict[str, list] = Field(description="Skills categorized")
    certifications: list = Field(description="Certifications list")


class VideoScript(BaseModel):
    """Video script schema"""
    script: str = Field(description="The video script text")
    duration_estimate: int = Field(description="Estimated duration in seconds")
    key_points: list = Field(description="Key points covered")


# ===================
# CV Parser Chain
# ===================
class CVParserChain:
    """Chain for parsing CV documents"""

    def __init__(self):
        self.llm = get_llm()
        self.parser = JsonOutputParser(pydantic_object=CVData)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert CV parser. Extract structured information from CV text.
Always respond with valid JSON matching the specified schema.
Be thorough but concise. Extract all relevant information."""),
            ("human", """Parse the following CV and extract structured data:

{cv_text}

{format_instructions}""")
        ])

        self.chain = (
            {"cv_text": RunnablePassthrough(), "format_instructions": lambda _: self.parser.get_format_instructions()}
            | self.prompt
            | self.llm
            | self.parser
        )

    async def parse(self, cv_text: str) -> Dict[str, Any]:
        """Parse CV text and return structured data"""
        result = await self.chain.ainvoke(cv_text)
        return result


# ===================
# Script Generator Chain
# ===================
class ScriptGeneratorChain:
    """Chain for generating video scripts from CV data"""

    LANGUAGE_INSTRUCTIONS = {
        "tr": "Türkçe olarak yaz. Profesyonel ve samimi bir ton kullan.",
        "en": "Write in English. Use a professional yet friendly tone.",
        "de": "Schreiben Sie auf Deutsch. Verwenden Sie einen professionellen, aber freundlichen Ton.",
        "ar": "اكتب باللغة العربية. استخدم نبرة مهنية ودودة.",
    }

    def __init__(self):
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating engaging video CV scripts.
Create natural, conversational scripts that sound authentic when spoken.
The person will speak directly to camera, so write in first person."""),
            ("human", """Create a video CV script based on the following data.
Target duration: {duration} seconds (approximately {word_count} words).

{language_instruction}

Structure:
1. Brief introduction (name, current role)
2. Key experience highlights (most relevant 2-3 points)
3. Key skills and strengths
4. What they're looking for / career goals
5. Closing statement

CV Data:
{cv_data}

Generate a natural, engaging script. No stage directions or brackets.""")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    async def generate(
        self,
        cv_data: Dict[str, Any],
        language: str = "tr",
        duration_seconds: int = 60
    ) -> str:
        """Generate video script from CV data"""
        word_count = duration_seconds * 2  # ~2 words per second

        result = await self.chain.ainvoke({
            "cv_data": str(cv_data),
            "duration": duration_seconds,
            "word_count": word_count,
            "language_instruction": self.LANGUAGE_INSTRUCTIONS.get(
                language,
                self.LANGUAGE_INSTRUCTIONS["en"]
            )
        })

        return result


# ===================
# Matching Chain
# ===================
class MatchingChain:
    """Chain for AI-powered job matching analysis"""

    def __init__(self):
        self.llm = get_llm()
        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert recruiter AI. Analyze candidate-job fit.
Provide detailed analysis and scoring."""),
            ("human", """Analyze the match between this candidate and job:

CANDIDATE:
{candidate_data}

JOB:
{job_data}

Provide:
1. Overall match score (0-100)
2. Skill match analysis
3. Experience relevance
4. Potential concerns
5. Recommendation

Respond in JSON format.""")
        ])

        self.chain = self.prompt | self.llm | self.parser

    async def analyze(
        self,
        candidate_data: Dict[str, Any],
        job_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze candidate-job match"""
        result = await self.chain.ainvoke({
            "candidate_data": str(candidate_data),
            "job_data": str(job_data)
        })
        return result


# ===================
# Hallucination Checker
# ===================
class HallucinationChecker:
    """Check for hallucinations in generated content"""

    def __init__(self):
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker. Verify that the generated content
accurately reflects the source data. Flag any hallucinations or fabricated information."""),
            ("human", """SOURCE DATA:
{source}

GENERATED CONTENT:
{generated}

Check for:
1. Factual accuracy - does the content match the source?
2. Fabricated information - anything not in source?
3. Exaggerations - claims beyond what source supports?

Respond with JSON:
{{
    "is_accurate": true/false,
    "issues": ["list of issues found"],
    "confidence": 0-100
}}""")
        ])

        self.chain = self.prompt | self.llm | JsonOutputParser()

    async def check(
        self,
        source_data: str,
        generated_content: str
    ) -> Dict[str, Any]:
        """Check generated content for hallucinations"""
        result = await self.chain.ainvoke({
            "source": source_data,
            "generated": generated_content
        })
        return result
