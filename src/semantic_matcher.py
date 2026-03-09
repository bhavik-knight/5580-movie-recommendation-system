import asyncio
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process

class SemanticMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.movie_embeddings = None
        self.movies_list = None

    async def initialize(self, movies_list):
        """Precompute embeddings for the entire movie dataset."""
        self.movies_list = movies_list
        self.movie_embeddings = await asyncio.to_thread(
            self.model.encode, movies_list, convert_to_tensor=True
        )

    async def find_matches(self, query_titles, threshold=0.7):
        """Match a list of raw titles to the dataset using semantic similarity."""
        if self.movie_embeddings is None:
            return []

        matched_titles = []
        for name in query_titles:
            # Encode extraction for semantic similarity check
            name_embedding = await asyncio.to_thread(self.model.encode, name, convert_to_tensor=True)
            
            # Semantic search against the entire dataset
            hits = util.semantic_search(name_embedding, self.movie_embeddings, top_k=1)
            
            score = hits[0][0]['score']
            if score > threshold:
                matched_titles.append(self.movies_list[hits[0][0]['corpus_id']])
            else:
                # Fallback to RapidFuzz for exact/typo matching
                match = process.extractOne(name, self.movies_list, score_cutoff=85)
                if match:
                    matched_titles.append(match[0])
        
        return list(dict.fromkeys(matched_titles))

    async def search_in_text(self, text, threshold=0.45):
        """Find movie titles directly within a conversational sentence (fallback)."""
        if self.movie_embeddings is None:
            return []
            
        query_embedding = await asyncio.to_thread(self.model.encode, text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.movie_embeddings, top_k=5)
        
        matched = []
        for hit in hits[0]:
            if hit['score'] > threshold:
                matched.append(self.movies_list[hit['corpus_id']])
        return matched
