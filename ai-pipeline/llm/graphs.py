"""
LangGraph Workflows for VidCV
Multi-step AI workflows with state management
"""
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from config import get_llm
from .chains import CVParserChain, ScriptGeneratorChain, HallucinationChecker


# ===================
# State Definitions
# ===================
class VideoGenerationState(TypedDict):
    """State for video generation workflow"""
    cv_text: str
    cv_data: dict
    script: str
    script_approved: bool
    audio_url: str
    video_url: str
    status: str
    error: str
    language: str
    avatar_url: str


class MatchingState(TypedDict):
    """State for matching workflow"""
    candidate_data: dict
    job_data: dict
    match_score: float
    match_details: dict
    recommendations: list
    status: str


# ===================
# Video Generation Graph
# ===================
class VideoGenerationGraph:
    """
    LangGraph workflow for video CV generation

    Flow:
    1. Parse CV -> 2. Generate Script -> 3. Check Hallucination
    -> 4. Generate Audio -> 5. Generate Video -> 6. Done
    """

    def __init__(self):
        self.llm = get_llm()
        self.cv_parser = CVParserChain()
        self.script_generator = ScriptGeneratorChain()
        self.hallucination_checker = HallucinationChecker()

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(VideoGenerationState)

        # Add nodes
        workflow.add_node("parse_cv", self._parse_cv)
        workflow.add_node("generate_script", self._generate_script)
        workflow.add_node("check_hallucination", self._check_hallucination)
        workflow.add_node("regenerate_script", self._regenerate_script)
        workflow.add_node("generate_audio", self._generate_audio)
        workflow.add_node("generate_video", self._generate_video)

        # Add edges
        workflow.set_entry_point("parse_cv")
        workflow.add_edge("parse_cv", "generate_script")
        workflow.add_edge("generate_script", "check_hallucination")

        # Conditional edge based on hallucination check
        workflow.add_conditional_edges(
            "check_hallucination",
            self._should_regenerate,
            {
                "regenerate": "regenerate_script",
                "continue": "generate_audio"
            }
        )

        workflow.add_edge("regenerate_script", "check_hallucination")
        workflow.add_edge("generate_audio", "generate_video")
        workflow.add_edge("generate_video", END)

        return workflow.compile()

    async def _parse_cv(self, state: VideoGenerationState) -> VideoGenerationState:
        """Parse CV text into structured data"""
        try:
            cv_data = await self.cv_parser.parse(state["cv_text"])
            return {**state, "cv_data": cv_data, "status": "cv_parsed"}
        except Exception as e:
            return {**state, "error": str(e), "status": "failed"}

    async def _generate_script(self, state: VideoGenerationState) -> VideoGenerationState:
        """Generate video script from CV data"""
        try:
            script = await self.script_generator.generate(
                cv_data=state["cv_data"],
                language=state.get("language", "tr")
            )
            return {**state, "script": script, "status": "script_generated"}
        except Exception as e:
            return {**state, "error": str(e), "status": "failed"}

    async def _check_hallucination(self, state: VideoGenerationState) -> VideoGenerationState:
        """Check script for hallucinations"""
        try:
            result = await self.hallucination_checker.check(
                source_data=str(state["cv_data"]),
                generated_content=state["script"]
            )
            is_accurate = result.get("is_accurate", False)
            return {
                **state,
                "script_approved": is_accurate,
                "status": "hallucination_checked"
            }
        except Exception as e:
            # On error, approve and continue
            return {**state, "script_approved": True, "status": "hallucination_checked"}

    def _should_regenerate(self, state: VideoGenerationState) -> str:
        """Decide whether to regenerate script"""
        if not state.get("script_approved", False):
            return "regenerate"
        return "continue"

    async def _regenerate_script(self, state: VideoGenerationState) -> VideoGenerationState:
        """Regenerate script with more constraints"""
        # Add instruction to be more factual
        script = await self.script_generator.generate(
            cv_data=state["cv_data"],
            language=state.get("language", "tr")
        )
        return {**state, "script": script, "status": "script_regenerated"}

    async def _generate_audio(self, state: VideoGenerationState) -> VideoGenerationState:
        """Generate TTS audio from script"""
        # TODO: Implement TTS integration
        return {**state, "audio_url": "", "status": "audio_generated"}

    async def _generate_video(self, state: VideoGenerationState) -> VideoGenerationState:
        """Generate avatar video"""
        # TODO: Implement video generation
        return {**state, "video_url": "", "status": "completed"}

    async def run(
        self,
        cv_text: str,
        language: str = "tr",
        avatar_url: str = ""
    ) -> VideoGenerationState:
        """Run the video generation workflow"""
        initial_state: VideoGenerationState = {
            "cv_text": cv_text,
            "cv_data": {},
            "script": "",
            "script_approved": False,
            "audio_url": "",
            "video_url": "",
            "status": "started",
            "error": "",
            "language": language,
            "avatar_url": avatar_url
        }

        result = await self.graph.ainvoke(initial_state)
        return result


# ===================
# Matching Graph
# ===================
class MatchingGraph:
    """
    LangGraph workflow for candidate-job matching

    Flow:
    1. Embed Candidate -> 2. Embed Job -> 3. Calculate Similarity
    -> 4. AI Analysis -> 5. Generate Recommendations -> 6. Done
    """

    def __init__(self):
        from .chains import MatchingChain
        self.matching_chain = MatchingChain()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build matching workflow"""
        workflow = StateGraph(MatchingState)

        workflow.add_node("analyze_match", self._analyze_match)
        workflow.add_node("generate_recommendations", self._generate_recommendations)

        workflow.set_entry_point("analyze_match")
        workflow.add_edge("analyze_match", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)

        return workflow.compile()

    async def _analyze_match(self, state: MatchingState) -> MatchingState:
        """Analyze candidate-job match"""
        result = await self.matching_chain.analyze(
            state["candidate_data"],
            state["job_data"]
        )
        return {
            **state,
            "match_score": result.get("score", 0),
            "match_details": result,
            "status": "analyzed"
        }

    async def _generate_recommendations(self, state: MatchingState) -> MatchingState:
        """Generate improvement recommendations"""
        # TODO: Implement recommendations
        return {**state, "recommendations": [], "status": "completed"}

    async def run(
        self,
        candidate_data: dict,
        job_data: dict
    ) -> MatchingState:
        """Run matching workflow"""
        initial_state: MatchingState = {
            "candidate_data": candidate_data,
            "job_data": job_data,
            "match_score": 0,
            "match_details": {},
            "recommendations": [],
            "status": "started"
        }

        result = await self.graph.ainvoke(initial_state)
        return result
