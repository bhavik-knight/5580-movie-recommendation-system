import chainlit as cl
import httpx
import json
import asyncio
import os
from together import Together
from dotenv import load_dotenv
from rapidfuzz import process
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# Initialize Together Client and Sentence Transformer Globaly
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

@cl.on_chat_start
async def start():
    async with httpx.AsyncClient() as client:
        try:
            # Check if FastAPI server is running
            response = await client.get(f"{FASTAPI_BASE_URL}/api/v1/health")
            if response.status_code != 200:
                 await cl.Message(content="⚠️ API server is not running correctly. Please check api/app.py").send()
        except httpx.ConnectError:
            await cl.Message(content="⚠️ API server is not running. Please start it with: `uv run python api/app.py`").send()
            return

        # Fetch movies
        try:
            movies_response = await client.get(f"{FASTAPI_BASE_URL}/api/v1/movies")
            if movies_response.status_code == 200:
                movies_data = movies_response.json()
                movies_list = movies_data.get("movies", [])
                cl.user_session.set("movies", movies_list)
                
                # Precompute movie embeddings for semantic search
                count = movies_data.get("total", 0)
                await cl.Message(content=f"🧠 Learning about {count} movies...").send()
                embeddings = await asyncio.to_thread(semantic_model.encode, movies_list, convert_to_tensor=True)
                cl.user_session.set("movie_embeddings", embeddings)
            else:
                count = 0
                await cl.Message(content="⚠️ Could not load movies from API.").send()
        except Exception as e:
            count = 0
            await cl.Message(content=f"⚠️ Error fetching movies: {str(e)}").send()

    welcome_message = f"""🎬 Welcome to the Movie Recommendation Chatbot!

I can suggest movies based on ones you already love.
**Please name at least 3 movies that you like to get started.**

For example:
- "I loved Toy Story, Fargo, and Pulp Fiction"
- "What should I watch if I liked Star Wars, Raiders of the Lost Ark, and Aladdin?"

Loaded {count} movies. Let's find your next favourite!"""

    await cl.Message(content=welcome_message).send()

@cl.on_message
async def on_message(message: cl.Message):
    movies = cl.user_session.get("movies")
    if not movies:
        await cl.Message(content="⚠️ Could not find the available movies list. Did the startup fetch fail?").send()
        return

    # 1. Use Together AI to extract ONLY the movies the user LIKES/LOVES
    try:
        # Improved Prompt: Handle sentiment, conversational filler, and examples
        extraction_prompt = f"""You are a movie extraction specialist. Your job is to find raw movie titles in the message.
Rules:
- ONLY extract movies the user mentions LIKING, LOVING, or wanting recommendations SIMILAR TO.
- EXPLICITLY IGNORE movies the user mentions HATING, DISLIKING, or wanting to AVOID.
- Extract up to 5 titles.
- Return a JSON array of strings: ["Title 1", "Title 2"].
- If no liked movies are found, return an empty array [].
- Strictly return ONLY the JSON data.

User Message: '{message.content}'"""

        response = await asyncio.to_thread(
            together_client.chat.completions.create,
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": extraction_prompt}]
        )
        
        # Parse the JSON response
        data = json.loads(response.choices[0].message.content)
        
        # Handle different potential JSON structures
        if isinstance(data, list):
            raw_names = data
        elif isinstance(data, dict):
            raw_names = []
            for k in data.keys():
                if "," in k and len(data.keys()) == 1:
                    raw_names.extend([s.strip() for s in k.split(",")])
                else:
                    raw_names.append(k)
        else:
            raw_names = []
            
        # Match extracted names using Semantic Embeddings
        matched_titles = []
        movie_embeddings = cl.user_session.get("movie_embeddings")

        for name in raw_names:
            # Encode extraction for semantic similarity check
            name_embedding = await asyncio.to_thread(semantic_model.encode, name, convert_to_tensor=True)
            
            # Semantic search against the 1682 movies
            hits = util.semantic_search(name_embedding, movie_embeddings, top_k=1)
            
            # Use threshold for semantic similarity (0.7+)
            score = hits[0][0]['score']
            if score > 0.7:
                corpus_id = hits[0][0]['corpus_id']
                found_title = movies[corpus_id]
                matched_titles.append(found_title)
            else:
                # Fallback to RapidFuzz if semantic match is weak
                match = process.extractOne(name, movies, score_cutoff=85)
                if match:
                    matched_titles.append(match[0])
        
        # Deduplicate matches
        matched_titles = list(dict.fromkeys(matched_titles))

    except Exception as e:
        # Fallback to direct fuzzy search if LLM fails
        results = process.extract(message.content, movies, limit=5)
        matched_titles = [res[0] for res in results if res[1] > 90]

    if not matched_titles:
        await cl.Message(content="I couldn't quite catch those movie titles. Keep in mind my database only contains movies from **1922 to 1998**. Could you try typing them exactly, or maybe mention classics from that era?").send()
        return

    if len(matched_titles) < 3:
        recognized = ", ".join([f"**{t}**" for t in matched_titles])
        await cl.Message(content=f"I recognized: {recognized}. For the best results, please name at least 3 movies you like! What else do you enjoy?").send()
        return

    # Limit to 5 movies to satisfy API constraints
    input_titles = matched_titles[:5]

    # Loading Indicator
    msg = cl.Message(content="Finding the perfect movies for you... 🍿")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/api/v1/recommend",
                json={"titles": input_titles, "top_n": 10}
            )
            response.raise_for_status()
            results = response.json()
            recommendations = results.get("recommendations", [])

        if not recommendations:
            msg.content = f"I found some movies ({', '.join(input_titles)}), but I don't have enough data to make recommendations for them right now. Remember, my knowledge is limited to movies released between **1922 and 1998**. Try different ones from that period?"
            await msg.update()
            return

        # Format & Display
        final_input_text = ", ".join([f"**{t}**" for t in input_titles])
        formatted_string = f"Great choices! Based on {final_input_text}, here are my top picks for you:\n\n"
        
        for rec in recommendations:
            rank = rec.get("rank")
            title = rec.get("title")
            score = rec.get("score")
            reason = rec.get("reason")
            formatted_string += f"🎬 **{rank}. {title}** — Score: {score:.2f}\n💡 Because: {reason}\n\n"

        formatted_string += "Would you like to explore any of these further, or try different movies?"
        
        msg.content = formatted_string
        await msg.update()

    except Exception as e:
        msg.content = f"Sorry, I ran into an issue connecting to the recommendation engine. Is the FastAPI server running on port 8000?\n\n*Error: {str(e)}*"
        await msg.update()
