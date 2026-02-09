-- VidCV PostgreSQL Initialization
-- Includes pgvector extension for AI embeddings

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "vector";    -- pgvector for AI embeddings

-- Create database (if not exists)
-- CREATE DATABASE vidcv;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE vidcv TO vidcv;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS ai;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ===================
-- Vector Tables for AI
-- ===================

-- CV Embeddings
CREATE TABLE IF NOT EXISTS ai.cv_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    candidate_id INTEGER NOT NULL,
    embedding vector(384),  -- sentence-transformers dimension
    content TEXT,
    embedding_model VARCHAR(100) DEFAULT 'paraphrase-multilingual-MiniLM-L12-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Job Embeddings
CREATE TABLE IF NOT EXISTS ai.job_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id INTEGER NOT NULL,
    embedding vector(384),
    content TEXT,
    embedding_model VARCHAR(100) DEFAULT 'paraphrase-multilingual-MiniLM-L12-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for vector similarity search
CREATE INDEX IF NOT EXISTS cv_embedding_idx ON ai.cv_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS job_embedding_idx ON ai.job_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ===================
-- Functions for similarity search
-- ===================

-- Find similar candidates for a job
CREATE OR REPLACE FUNCTION ai.find_similar_candidates(
    job_embedding vector(384),
    limit_count INTEGER DEFAULT 20
)
RETURNS TABLE (
    candidate_id INTEGER,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ce.candidate_id,
        1 - (ce.embedding <=> job_embedding) as similarity
    FROM ai.cv_embeddings ce
    ORDER BY ce.embedding <=> job_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Find similar jobs for a candidate
CREATE OR REPLACE FUNCTION ai.find_similar_jobs(
    cv_embedding vector(384),
    limit_count INTEGER DEFAULT 20
)
RETURNS TABLE (
    job_id INTEGER,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        je.job_id,
        1 - (je.embedding <=> cv_embedding) as similarity
    FROM ai.job_embeddings je
    ORDER BY je.embedding <=> cv_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- ===================
-- Analytics Tables
-- ===================

CREATE TABLE IF NOT EXISTS analytics.video_views (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_cv_id INTEGER NOT NULL,
    viewer_id INTEGER,
    viewer_type VARCHAR(50),  -- employer, headhunter, anonymous
    watch_duration_seconds INTEGER,
    completed BOOLEAN DEFAULT FALSE,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics.search_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER,
    search_query TEXT,
    filters JSONB,
    results_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS video_views_video_id_idx ON analytics.video_views(video_cv_id);
CREATE INDEX IF NOT EXISTS video_views_created_at_idx ON analytics.video_views(created_at);
CREATE INDEX IF NOT EXISTS search_logs_user_id_idx ON analytics.search_logs(user_id);

COMMENT ON SCHEMA ai IS 'AI-related tables including vector embeddings';
COMMENT ON SCHEMA analytics IS 'Analytics and tracking tables';
