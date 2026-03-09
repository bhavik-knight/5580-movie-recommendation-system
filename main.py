import chainlit as cl
import httpx
import json
import ollama
import asyncio
import os
from dotenv import load_dotenv
from rapidfuzz import process

load_dotenv()

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
                count = movies_data.get("total", 0)
            else:
                count = 0
                await cl.Message(content="⚠️ Could not load movies from API.").send()
        except Exception as e:
            count = 0
            await cl.Message(content=f"⚠️ Error fetching movies: {str(e)}").send()

    welcome_message = f"""🎬 Welcome to the Movie Recommendation Chatbot!

I can suggest movies based on ones you already love.
Just tell me naturally what you enjoy, for example:
- "I loved Star Wars and Raiders of the Lost Ark"
- "Suggest something like Fargo or Pulp Fiction"
- "What should I watch if I liked Toy Story?"

Loaded {count} movies. Let's find your next favourite!"""

    await cl.Message(content=welcome_message).send()

@cl.on_message
async def on_message(message: cl.Message):
    movies = cl.user_session.get("movies")
    if not movies:
        await cl.Message(content="⚠️ Could not find the available movies list. Did the startup fetch fail?").send()
        return

    # 1. Use Ollama to extract the movie names from the message
    try:
        response = await asyncio.to_thread(
            ollama.chat,
            model="llama3.1:latest",
            messages=[
                {"role": "system", "content": "You are a movie title extractor. Your job is to extract raw movie names mentioned in the user message. Return a JSON array of strings: [\"Movie 1\", \"Movie 2\"]. If it's more convenient to return a JSON object, put the titles as keys. ONLY the JSON data."},
                {"role": "user", "content": f"Extract movie titles from: '{message.content}'"}
            ],
            format="json"
        )
        
        # Parse the JSON response
        data = json.loads(response['message']['content'])
        
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
            
        # Match extracted names with the exact titles in our dataset
        matched_titles = []
        for name in raw_names:
            match = process.extractOne(name, movies, score_cutoff=75)
            if match:
                matched_titles.append(match[0])
        
        # Deduplicate matches
        matched_titles = list(dict.fromkeys(matched_titles))

    except Exception as e:
        # Fallback to direct fuzzy search if LLM fails
        results = process.extract(message.content, movies, limit=5)
        matched_titles = [res[0] for res in results if res[1] > 90]

    if not matched_titles:
        await cl.Message(content="I couldn't quite catch those movie titles. Could you try typing them exactly, or maybe mention others?").send()
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
            msg.content = f"I found some movies ({', '.join(input_titles)}), but I don't have enough data to make recommendations for them right now. Try different ones?"
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
