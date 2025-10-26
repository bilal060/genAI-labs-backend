from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import sys
import time
import asyncio
from dotenv import load_dotenv
import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from motor.motor_asyncio import AsyncIOMotorClient
from cache import response_cache, performance_monitor
from anthropic import Anthropic
import httpx
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from logging_config import setup_logging

load_dotenv()

from contextlib import asynccontextmanager

# Setup logging
logger = setup_logging()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URL)
db = mongo_client.llm_lab
experiments_collection = db.experiments

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting application")
    await init_db()
    logger.info("Application startup completed")
    yield
    # Shutdown
    mongo_client.close()
    logger.info("Application shutdown")

app = FastAPI(title="LLM Lab API", version="1.0.0", docs_url="/docs", redoc_url="/redoc", lifespan=lifespan)

# Setup rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["50/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add GZip middleware for better performance
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://gen-ai-labs-frontend.vercel.app",
        "https://genai-labs-frontend.vercel.app",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# OpenRouter configuration (Primary)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Claude configuration (Fallback)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
claude_client = None

# Initialize APIs
if OPENROUTER_API_KEY:
    logger.info("OpenRouter API key found")
else:
    logger.warning("OpenRouter API key not found")

if ANTHROPIC_API_KEY:
    try:
        claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Claude API initialized successfully")
    except Exception as e:
        logger.error(f"Claude API initialization failed: {e}")
else:
    logger.warning("Claude API key not found")

def call_openrouter(prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 100) -> str:
    """Call OpenRouter API to generate text"""
    # Check cache first for speed
    cached_response = response_cache.get(prompt, temperature, top_p, max_tokens)
    if cached_response:
        logger.info("Using cached response")
        return cached_response
    
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API not initialized")
        raise Exception("OpenRouter API not initialized")
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://gen-ai-labs-frontend.vercel.app",
            "X-Title": "LLM Lab"
        }
        
        payload = {
            "model": "openai/gpt-4o-mini",  # Using GPT-4o-mini via OpenRouter
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(OPENROUTER_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()
            
            # Cache the response for future use
            response_cache.set(prompt, temperature, top_p, max_tokens, response_text)
            logger.info("Cached response for future use")
            
            return response_text
        else:
            error_detail = response.text
            if response.status_code == 401:
                logger.error("OpenRouter API key invalid or unauthorized")
                raise Exception("OpenRouter API key invalid or unauthorized")
            elif response.status_code == 429:
                logger.warning("OpenRouter API rate limit exceeded")
                raise Exception("OpenRouter API rate limit exceeded")
            else:
                logger.error(f"OpenRouter API error {response.status_code}: {error_detail}")
                raise Exception(f"OpenRouter API error {response.status_code}: {error_detail}")
                
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        raise Exception(f"Error calling OpenRouter: {str(e)}")

def call_claude(prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 100) -> str:
    """Call Claude API to generate text"""
    # Check cache first for speed
    cached_response = response_cache.get(prompt, temperature, top_p, max_tokens)
    if cached_response:
        logger.info("Using cached response")
        return cached_response
    
    if not claude_client:
        logger.error("Claude API not initialized")
        raise Exception("Claude API not initialized")
    
    try:
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text.strip()
        
        # Cache the response for future use
        response_cache.set(prompt, temperature, top_p, max_tokens, response_text)
        logger.info("Cached response for future use")
        
        return response_text
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        error_msg = str(e)
        if "credit balance" in error_msg.lower() or "too low" in error_msg.lower():
            raise Exception("Claude API credit balance too low. Please add credits to your account.")
        elif "invalid_request_error" in error_msg.lower():
            raise Exception("Invalid Claude API request. Please check your API key and model.")
        else:
            raise Exception(f"Error calling Claude: {str(e)}")

# Clear cache on startup to avoid old error responses
response_cache.clear()
logger.info("Cache cleared on startup")

# Database setup
async def init_db():
    """Initialize MongoDB database and collections"""
    try:
        # Test connection with timeout
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5.0)
        logger.info("MongoDB connection successful")
        
        # Create indexes for better performance
        await experiments_collection.create_index([("created_at", -1)])  # Descending order
        await experiments_collection.create_index([("name", 1)])
        await experiments_collection.create_index([("response_count", 1)])
        logger.info("MongoDB indexes created successfully")
        
    except asyncio.TimeoutError:
        logger.error("MongoDB connection timeout - skipping database initialization")
        return
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        logger.error("Please ensure MongoDB is running or set MONGODB_URL environment variable")
        return

# Database setup

# Pydantic models
class ParameterRange(BaseModel):
    temperature: List[float]
    top_p: List[float]
    max_tokens: int = 1000

class ExperimentRequest(BaseModel):
    prompt: str
    parameter_ranges: ParameterRange
    experiment_name: str

class Response(BaseModel):
    text: str
    parameters: dict
    metrics: dict

class ExperimentResponse(BaseModel):
    experiment_id: str  # Changed from int to str for MongoDB ObjectId
    name: str
    responses: List[Response]
    response_count: int
    created_at: str

# Quality Metrics Implementation
class QualityMetrics:
    @staticmethod
    def calculate_completeness(text: str) -> float:
        """Calculate completeness based on sentence structure and length"""
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check for complete sentences (have subject-verb structure)
        complete_sentences = 0
        for sentence in sentences:
            if len(sentence.split()) >= 3:  # Basic length check
                complete_sentences += 1
        
        return complete_sentences / len(sentences) if sentences else 0.0
    
    @staticmethod
    def calculate_coherence(text: str) -> float:
        """Calculate coherence using TF-IDF similarity between sentences"""
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0 if sentences else 0.0
        
        # Use TF-IDF to measure similarity between consecutive sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarities = []
            
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
        except:
            return 0.5  # Default value if TF-IDF fails
    
    @staticmethod
    def calculate_creativity(text: str) -> float:
        """Calculate creativity based on vocabulary diversity and unique phrases"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
        
        # Vocabulary diversity (unique words / total words)
        unique_words = len(set(words))
        total_words = len(words)
        diversity = unique_words / total_words if total_words > 0 else 0.0
        
        # Check for creative patterns (alliteration, metaphors, etc.)
        creative_score = 0.0
        
        # Alliteration detection
        words_by_first_letter = {}
        for word in words:
            if len(word) > 3:
                first_letter = word[0]
                if first_letter not in words_by_first_letter:
                    words_by_first_letter[first_letter] = 0
                words_by_first_letter[first_letter] += 1
        
        alliteration_score = max(words_by_first_letter.values()) / total_words if words_by_first_letter else 0.0
        creative_score += min(alliteration_score * 2, 0.3)  # Cap at 0.3
        
        # Length appropriateness (not too short, not too long)
        length_score = 1.0 - abs(len(text) - 500) / 1000  # Optimal around 500 chars
        length_score = max(0, min(1, length_score))
        
        return (diversity * 0.6 + creative_score * 0.2 + length_score * 0.2)
    
    @staticmethod
    def calculate_relevance(text: str, prompt: str) -> float:
        """Calculate relevance by checking keyword overlap and semantic similarity"""
        # Simple keyword overlap
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        if not prompt_words:
            return 1.0
        
        overlap = len(prompt_words.intersection(text_words))
        keyword_relevance = overlap / len(prompt_words)
        
        # Check for question answering (if prompt is a question)
        if '?' in prompt:
            question_words = ['what', 'how', 'why', 'when', 'where', 'who']
            has_answer = any(word in text.lower() for word in question_words)
            answer_bonus = 0.2 if has_answer else 0.0
        else:
            answer_bonus = 0.0
        
        return min(1.0, keyword_relevance + answer_bonus)
    
    @staticmethod
    def calculate_overall_score(text: str, prompt: str) -> dict:
        """Calculate all metrics and return overall score"""
        completeness = QualityMetrics.calculate_completeness(text)
        coherence = QualityMetrics.calculate_coherence(text)
        creativity = QualityMetrics.calculate_creativity(text)
        relevance = QualityMetrics.calculate_relevance(text, prompt)
        
        # Weighted overall score
        overall = (completeness * 0.25 + coherence * 0.25 + 
                  creativity * 0.25 + relevance * 0.25)
        
        return {
            "completeness": round(completeness, 3),
            "coherence": round(coherence, 3),
            "creativity": round(creativity, 3),
            "relevance": round(relevance, 3),
            "overall": round(overall, 3)
        }

# API Endpoints
@app.get("/")
async def root():
    return {"message": "LLM Lab API is running"}

@app.get("/api/performance")
async def get_performance_metrics():
    """Get current performance metrics"""
    return performance_monitor.get_metrics()

@app.post("/api/experiment", response_model=ExperimentResponse)
@limiter.limit("10/minute")
async def create_experiment(request: Request, experiment_request: ExperimentRequest):
    start_time = time.time()
    try:
        responses = []
        cache_hits = 0
        cache_misses = 0
        
        # Generate responses for each parameter combination
        for temp in experiment_request.parameter_ranges.temperature:
            for top_p in experiment_request.parameter_ranges.top_p:
                try:
                    # Check cache first
                    cached_response = response_cache.get(
                        experiment_request.prompt, temp, top_p, experiment_request.parameter_ranges.max_tokens
                    )
                    
                    if cached_response:
                        response_text = cached_response
                        cache_hits += 1
                        logger.info(f"Cache hit for temp={temp}, top_p={top_p}")
                    else:
                        cache_misses += 1
                        # Use OpenRouter first, then Claude as fallback
                        logger.info(f"Generating response with temp={temp}, top_p={top_p}")
                        
                        response_text = None
                        api_used = None
                        
                        # Try OpenRouter first
                        if OPENROUTER_API_KEY:
                            try:
                                response_text = call_openrouter(
                                    experiment_request.prompt,
                                    temperature=temp,
                                    top_p=top_p,
                                    max_tokens=experiment_request.parameter_ranges.max_tokens
                                )
                                api_used = "OpenRouter"
                                logger.info(f"Generated response with OpenRouter ({len(response_text)} chars)")
                            except Exception as e:
                                logger.warning(f"OpenRouter failed: {e}")
                                response_text = None
                        
                        # Fallback to Claude if OpenRouter failed
                        if not response_text and claude_client:
                            try:
                                response_text = call_claude(
                                    experiment_request.prompt,
                                    temperature=temp,
                                    top_p=top_p,
                                    max_tokens=experiment_request.parameter_ranges.max_tokens
                                )
                                api_used = "Claude"
                                logger.info(f"Generated response with Claude ({len(response_text)} chars)")
                            except Exception as e:
                                logger.warning(f"Claude failed: {e}")
                                response_text = None
                        
                        # If both failed
                        if not response_text:
                            raise Exception("Both OpenRouter and Claude APIs failed. Please check your API keys.")
                        
                        # Cache the response for future use
                        response_cache.set(
                            experiment_request.prompt, temp, top_p, experiment_request.parameter_ranges.max_tokens, response_text
                        )
                    
                    parameters = {
                        "temperature": temp,
                        "top_p": top_p,
                        "max_tokens": experiment_request.parameter_ranges.max_tokens
                    }
                    
                    # Calculate quality metrics
                    metrics = QualityMetrics.calculate_overall_score(response_text, experiment_request.prompt)
                    
                    responses.append(Response(
                        text=response_text,
                        parameters=parameters,
                        metrics=metrics
                    ))
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating response (temp={temp}, top_p={top_p}): {error_msg}")
                    
                    # Check for specific error types
                    if "both" in error_msg.lower() and "failed" in error_msg.lower():
                        logger.error("Both OpenRouter and Claude APIs failed")
                        raise HTTPException(
                            status_code=401, 
                            detail="Both OpenRouter and Claude APIs failed. Please check your API keys."
                        )
                    elif "openrouter" in error_msg.lower() and "not initialized" in error_msg.lower():
                        logger.error("OpenRouter API not initialized")
                        raise HTTPException(
                            status_code=401, 
                            detail="OpenRouter API not initialized. Please check your OPENROUTER_API_KEY."
                        )
                    elif "claude" in error_msg.lower() and "not initialized" in error_msg.lower():
                        logger.error("Claude API not initialized")
                        raise HTTPException(
                            status_code=401, 
                            detail="Claude API not initialized. Please check your ANTHROPIC_API_KEY."
                        )
                    
                    # Continue with other parameter combinations for other errors
                    continue
        
        if not responses:
            raise HTTPException(status_code=500, detail="Failed to generate any responses")
        
        # Save to MongoDB
        responses_data = [{"text": r.text, "parameters": r.parameters, "metrics": r.metrics} for r in responses]
        
        experiment_doc = {
            "name": experiment_request.experiment_name,
            "prompt": experiment_request.prompt,
            "parameters": experiment_request.parameter_ranges.model_dump(),
            "responses": responses_data,
            "metrics": [r.metrics for r in responses],
            "response_count": len(responses),
            "created_at": datetime.now()
        }
        
        result = await experiments_collection.insert_one(experiment_doc)
        experiment_id = result.inserted_id
        
        # Record performance metrics
        end_time = time.time()
        response_time = end_time - start_time
        performance_monitor.record_api_call(response_time, cache_hits > 0)
        
        logger.info(f"Performance: {response_time:.2f}s total, {cache_hits} cache hits, {cache_misses} cache misses")
        
        return ExperimentResponse(
            experiment_id=str(experiment_id),
            name=experiment_request.experiment_name,
            responses=responses,
            response_count=len(responses),
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/experiments")
async def get_experiments():
    try:
        logger.info("Fetching experiments from MongoDB")
        # Use to_list() to convert async cursor to list
        docs = await experiments_collection.find().sort("created_at", -1).to_list(length=None)
        logger.info(f"Found {len(docs)} documents")
        
        experiments = []
        for i, doc in enumerate(docs):
            logger.info(f"Processing experiment {i+1}: {doc.get('name', 'Unknown')}")
            experiments.append({
                "experiment_id": str(doc["_id"]),
                "id": str(doc["_id"]),  # Keep both for compatibility
                "name": doc["name"],
                "prompt": doc["prompt"],
                "parameters": doc["parameters"],
                "responses": doc["responses"],
                "metrics": doc["metrics"],
                "response_count": doc["response_count"],
                "created_at": doc["created_at"].isoformat()
            })
        
        logger.info(f"Processed {len(experiments)} experiments")
        return experiments
    except Exception as e:
        logger.error(f"Error fetching experiments: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching experiments: {str(e)}")

@app.get("/api/experiment/{experiment_id}")
async def get_experiment(experiment_id: str):
    from bson import ObjectId
    
    try:
        doc = await experiments_collection.find_one({"_id": ObjectId(experiment_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return {
        "experiment_id": str(doc["_id"]),
        "id": str(doc["_id"]),  # Keep both for compatibility
        "name": doc["name"],
        "prompt": doc["prompt"],
        "parameters": doc["parameters"],
        "responses": doc["responses"],
        "metrics": doc["metrics"],
        "response_count": doc["response_count"],
        "created_at": doc["created_at"].isoformat()
    }

@app.delete("/api/experiment/{experiment_id}")
async def delete_experiment(experiment_id: str):
    from bson import ObjectId
    
    try:
        result = await experiments_collection.delete_one({"_id": ObjectId(experiment_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return {"message": "Experiment deleted successfully"}

@app.get("/api/experiment/{experiment_id}/export")
async def export_experiment(experiment_id: str):
    """Export experiment data as CSV"""
    from bson import ObjectId
    
    try:
        doc = await experiments_collection.find_one({"_id": ObjectId(experiment_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Parse experiment data
    experiment = {
        "experiment_id": str(doc["_id"]),
        "name": doc["name"],
        "prompt": doc["prompt"],
        "parameters": doc["parameters"],
        "responses": doc["responses"],
        "metrics": doc["metrics"],
        "response_count": doc["response_count"],
        "created_at": doc["created_at"].isoformat()
    }
    
    # Generate CSV content
    headers = ['Temperature', 'Top-p', 'Max Tokens', 'Completeness', 'Coherence', 'Creativity', 'Relevance', 'Overall', 'Response Text']
    rows = []
    
    for response in experiment['responses']:
        rows.append([
            response['parameters']['temperature'],
            response['parameters']['top_p'],
            response['parameters']['max_tokens'],
            response['metrics']['completeness'],
            response['metrics']['coherence'],
            response['metrics']['creativity'],
            response['metrics']['relevance'],
            response['metrics']['overall'],
            response['text'].replace('\n', ' ').replace('\r', ' ')
        ])
    
    # Create CSV content
    csv_content = [headers] + rows
    csv_text = '\n'.join([','.join([str(cell) for cell in row]) for row in csv_content])
    
    from fastapi.responses import Response
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.csv"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
