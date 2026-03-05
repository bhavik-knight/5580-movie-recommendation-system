import sys
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Adjust path so src module can be imported
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ITEM_SIMILARITY_FILE, MOVIE_LOOKUP_FILE, OUTPUT_DIR, RATINGS_MATRIX_FILE, U_ITEM_PATH, U_ITEM_NAMES
from src.recommender import recommend

# Configure error logging
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
error_log = OUTPUT_DIR / "api_error.log"

logging.basicConfig(
    filename=error_log,
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MovieLens Recommender API",
    description="API for recommending movies based on item-item collaborative filtering.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# App-level state
class AppState:
    similarity_df = None
    title_to_id = None
    id_to_title = None
    u_item = None
    ratings_matrix = None
    average_ratings = None

state = AppState()

@app.on_event("startup")
def startup_event():
    """Load matrices and lookups into memory on startup."""
    try:
        print("Starting app...")
        
        if not ITEM_SIMILARITY_FILE.exists() or not MOVIE_LOOKUP_FILE.exists():
            raise FileNotFoundError("Precomputed matrices not found. Run src/client.py first to generate them.")
            
        print(f"Loading {ITEM_SIMILARITY_FILE}...")
        state.similarity_df = pd.read_csv(ITEM_SIMILARITY_FILE, index_col=0)
        state.similarity_df.columns = state.similarity_df.columns.astype(int)
        
        print(f"Loading {MOVIE_LOOKUP_FILE}...")
        lookup_df = pd.read_csv(MOVIE_LOOKUP_FILE)
        state.title_to_id = dict(zip(lookup_df['title'], lookup_df['movie_id']))
        state.id_to_title = dict(zip(lookup_df['movie_id'], lookup_df['title']))
        
        print(f"Loading {U_ITEM_PATH}...")
        state.u_item = pd.read_csv(U_ITEM_PATH, sep="|", names=U_ITEM_NAMES, encoding="ISO-8859-1")
        
        if RATINGS_MATRIX_FILE.exists():
            print(f"Loading {RATINGS_MATRIX_FILE}...")
            state.ratings_matrix = pd.read_csv(RATINGS_MATRIX_FILE, index_col=0)
            state.average_ratings = state.ratings_matrix.mean(axis=0)
        else:
            print(f"Warning: {RATINGS_MATRIX_FILE} not found. Average ratings will not be available.")
            state.average_ratings = pd.Series()
            
        print("All data loaded successfully.")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        print(f"Failed to load data on startup: {str(e)}")
        raise e

# Pydantic Models
class RecommendRequest(BaseModel):
    titles: List[str]
    top_n: int = 10

class MovieResult(BaseModel):
    rank: int
    title: str
    score: float
    reason: str

class RecommendResponse(BaseModel):
    input_titles: List[str]
    valid_titles: List[str]
    recommendations: List[MovieResult]
    message: str

@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        count = len(state.title_to_id) if state.title_to_id else 0
        return {"status": "ok", "movies_loaded": count}
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/movies")
def get_movies():
    """Returns full list of available movie titles (sorted alphabetically)."""
    try:
        if not state.title_to_id:
            raise HTTPException(status_code=503, detail="Service unavailable (data not loaded)")
            
        titles = sorted(list(state.title_to_id.keys()))
        return titles
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get movies error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: RecommendRequest):
    """Generates recommendations based on a list of input titles."""
    try:
        if not state.title_to_id:
            raise HTTPException(status_code=503, detail="Service unavailable (data not loaded)")
            
        if not (1 <= len(request.titles) <= 5):
            raise HTTPException(status_code=400, detail="Please provide between 1 and 5 movie titles.")
            
        valid_titles = [title for title in request.titles if title in state.title_to_id]
        
        if not valid_titles:
            raise HTTPException(status_code=400, detail=f"None of the provided titles were found in the database. Please provide valid movie titles from the 1997 MovieLens dataset.")
            
        results = recommend(valid_titles, top_n=request.top_n)
        
        recommendations = [MovieResult(**item) for item in results]
        message = "Recommendations generated successfully."
        
        return RecommendResponse(
            input_titles=request.titles,
            valid_titles=valid_titles,
            recommendations=recommendations,
            message=message
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error while generating recommendations.")

@app.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    """Returns title, genres, and average rating for a single movie."""
    try:
        if not state.u_item is not None:
            raise HTTPException(status_code=503, detail="Service unavailable (data not loaded)")
            
        movie_row = state.u_item[state.u_item['movie_id'] == movie_id]
        
        if movie_row.empty:
            raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found.")
            
        title = movie_row['title'].values[0]
        
        # Extract genres
        from src.config import GENRE_COLUMNS
        genres = [g for g in GENRE_COLUMNS if movie_row[g].values[0] == 1]
        
        avg_rating = None
        if not state.average_ratings.empty and str(movie_id) in state.average_ratings.index:
            avg_rating = float(state.average_ratings[str(movie_id)])
        elif not state.average_ratings.empty and movie_id in state.average_ratings.index:
            avg_rating = float(state.average_ratings[movie_id])
            
        return {
            "movie_id": movie_id,
            "title": title,
            "genres": genres,
            "average_rating": round(avg_rating, 2) if avg_rating is not None else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get movie details error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
