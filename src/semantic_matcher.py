import asyncio

from rapidfuzz import process
from sentence_transformers import SentenceTransformer, util


class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the Semantic Matcher.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.movie_embeddings = None
        self.movies_list: list[str] | None = None

    async def initialize(self, movies_list: list[str]) -> None:
        """Precompute embeddings for the entire movie dataset.

        Args:
            movies_list (list[str]): List of all movie titles.
        """
        self.movies_list = movies_list
        self.movie_embeddings = await asyncio.to_thread(
            self.model.encode,  # type: ignore
            movies_list,
            convert_to_tensor=True,
        )

    async def find_matches(self, query_titles: list[str], threshold: float = 0.65) -> list[str]:
        """Match a list of raw titles to the dataset using semantic similarity.

        Args:
            query_titles (list[str]): List of query movie titles to find matches for.
            threshold (float): Similarity threshold above which to consider a match valid.

        Returns:
            list[str]: A list of matched movie titles.
        """
        if self.movie_embeddings is None or self.movies_list is None:
            return []

        matched_titles = []
        for name in query_titles:
            # Encode extraction for semantic similarity check
            name_embedding = await asyncio.to_thread(  # type: ignore
                self.model.encode, name, convert_to_tensor=True
            )

            # Semantic search against the entire dataset
            hits = util.semantic_search(name_embedding, self.movie_embeddings, top_k=1)

            score = hits[0][0]["score"]
            if score > threshold:
                matched_titles.append(self.movies_list[hits[0][0]["corpus_id"]])
            else:
                # Fallback to RapidFuzz for exact/typo matching
                match = process.extractOne(name, self.movies_list, score_cutoff=80)
                if match:
                    matched_titles.append(match[0])

        return list(dict.fromkeys(matched_titles))

    async def search_in_text(self, text: str, threshold: float = 0.45) -> list[str]:
        """Find movie titles directly within a conversational sentence (fallback).

        Args:
            text (str): The conversational text to search within.
            threshold (float): Similarity threshold for matches.

        Returns:
            list[str]: A list of matched movie titles from the text.
        """
        if self.movie_embeddings is None or self.movies_list is None:
            return []

        query_embedding = await asyncio.to_thread(  # type: ignore
            self.model.encode, text, convert_to_tensor=True
        )
        hits = util.semantic_search(query_embedding, self.movie_embeddings, top_k=5)

        matched = []
        for hit in hits[0]:
            if hit["score"] > threshold:
                matched.append(self.movies_list[hit["corpus_id"]])
        return matched
