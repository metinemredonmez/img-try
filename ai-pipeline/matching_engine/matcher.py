"""
AI Matching Engine
Matches candidates with jobs using embeddings and ML
"""
import os
from typing import Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class MatchingEngine:
    """AI-powered candidate-job matching"""

    def __init__(self):
        self.model = None
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    async def load_models(self):
        """Load ML models on startup"""
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully!")

    def _ensure_model(self):
        """Ensure model is loaded"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    async def match(
        self,
        candidate_data: Dict[str, Any],
        job_data: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate match score between candidate and job
        Returns score (0-100) and detailed breakdown
        """
        self._ensure_model()

        # Calculate component scores
        skill_score = self._calculate_skill_match(candidate_data, job_data)
        experience_score = self._calculate_experience_match(candidate_data, job_data)
        education_score = self._calculate_education_match(candidate_data, job_data)
        semantic_score = self._calculate_semantic_match(candidate_data, job_data)

        # Weighted average (based on PDF specs)
        weights = {
            "skills": 0.40,
            "experience": 0.25,
            "education": 0.10,
            "semantic": 0.25  # Combines culture fit + overall similarity
        }

        total_score = (
            skill_score * weights["skills"] +
            experience_score * weights["experience"] +
            education_score * weights["education"] +
            semantic_score * weights["semantic"]
        )

        details = {
            "skill_match": round(skill_score, 2),
            "experience_match": round(experience_score, 2),
            "education_match": round(education_score, 2),
            "semantic_match": round(semantic_score, 2),
            "weights_used": weights,
            "matched_skills": self._get_matched_skills(candidate_data, job_data),
            "missing_skills": self._get_missing_skills(candidate_data, job_data)
        }

        return round(total_score, 2), details

    def _calculate_skill_match(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any]
    ) -> float:
        """Calculate skill match percentage"""
        candidate_skills = set()
        if "skills" in candidate:
            skills_data = candidate["skills"]
            if isinstance(skills_data, dict):
                candidate_skills.update(s.lower() for s in skills_data.get("technical", []))
                candidate_skills.update(s.lower() for s in skills_data.get("soft", []))
            elif isinstance(skills_data, list):
                candidate_skills.update(s.lower() for s in skills_data)

        job_skills = set()
        required = job.get("required_skills", job.get("skills", []))
        if isinstance(required, list):
            job_skills.update(s.lower() for s in required)

        if not job_skills:
            return 100.0  # No requirements = perfect match

        matched = candidate_skills & job_skills
        return (len(matched) / len(job_skills)) * 100

    def _calculate_experience_match(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any]
    ) -> float:
        """Calculate experience match"""
        # Get candidate's total experience years
        candidate_years = 0
        for exp in candidate.get("experience", []):
            # Simple estimation: each job = 2 years average
            candidate_years += 2

        # Get job requirements
        min_years = job.get("experience_years_min", 0)
        max_years = job.get("experience_years_max", 20)

        if candidate_years >= min_years:
            if candidate_years <= max_years:
                return 100.0  # Perfect fit
            else:
                # Overqualified penalty
                return max(60, 100 - (candidate_years - max_years) * 5)
        else:
            # Underqualified
            gap = min_years - candidate_years
            return max(0, 100 - gap * 20)

    def _calculate_education_match(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any]
    ) -> float:
        """Calculate education match"""
        education_levels = {
            "high school": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "phd": 5,
            "doctorate": 5
        }

        # Get candidate's highest education
        candidate_level = 1
        for edu in candidate.get("education", []):
            degree = edu.get("degree", "").lower()
            for level_name, level_value in education_levels.items():
                if level_name in degree:
                    candidate_level = max(candidate_level, level_value)

        # Get required level
        required = job.get("education_level", "").lower()
        required_level = 1
        for level_name, level_value in education_levels.items():
            if level_name in required:
                required_level = level_value
                break

        if candidate_level >= required_level:
            return 100.0
        else:
            gap = required_level - candidate_level
            return max(0, 100 - gap * 25)

    def _calculate_semantic_match(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity using embeddings"""
        # Create text representations
        candidate_text = self._create_candidate_text(candidate)
        job_text = self._create_job_text(job)

        # Get embeddings
        embeddings = self.model.encode([candidate_text, job_text])

        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return similarity * 100

    def _create_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """Create text representation of candidate"""
        parts = []

        if "summary" in candidate:
            parts.append(candidate["summary"])

        if "skills" in candidate:
            skills = candidate["skills"]
            if isinstance(skills, dict):
                all_skills = skills.get("technical", []) + skills.get("soft", [])
                parts.append("Skills: " + ", ".join(all_skills))
            elif isinstance(skills, list):
                parts.append("Skills: " + ", ".join(skills))

        for exp in candidate.get("experience", [])[:3]:
            parts.append(f"{exp.get('title', '')} at {exp.get('company', '')}")

        return " ".join(parts)

    def _create_job_text(self, job: Dict[str, Any]) -> str:
        """Create text representation of job"""
        parts = []

        if "title" in job:
            parts.append(job["title"])

        if "description" in job:
            parts.append(job["description"])

        skills = job.get("required_skills", job.get("skills", []))
        if skills:
            parts.append("Required: " + ", ".join(skills))

        return " ".join(parts)

    def _get_matched_skills(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any]
    ) -> List[str]:
        """Get list of matched skills"""
        candidate_skills = set()
        if "skills" in candidate:
            skills_data = candidate["skills"]
            if isinstance(skills_data, dict):
                candidate_skills.update(s.lower() for s in skills_data.get("technical", []))
            elif isinstance(skills_data, list):
                candidate_skills.update(s.lower() for s in skills_data)

        job_skills = set(s.lower() for s in job.get("required_skills", []))
        return list(candidate_skills & job_skills)

    def _get_missing_skills(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any]
    ) -> List[str]:
        """Get list of missing skills"""
        candidate_skills = set()
        if "skills" in candidate:
            skills_data = candidate["skills"]
            if isinstance(skills_data, dict):
                candidate_skills.update(s.lower() for s in skills_data.get("technical", []))
            elif isinstance(skills_data, list):
                candidate_skills.update(s.lower() for s in skills_data)

        job_skills = set(s.lower() for s in job.get("required_skills", []))
        return list(job_skills - candidate_skills)

    async def find_candidates(
        self,
        job_data: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find matching candidates for a job"""
        # TODO: Implement with Elasticsearch or vector DB
        return []

    async def find_jobs(
        self,
        candidate_data: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find matching jobs for a candidate"""
        # TODO: Implement with Elasticsearch or vector DB
        return []
