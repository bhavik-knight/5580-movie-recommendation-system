import chainlit as cl
import httpx
import json
import ollama
import asyncio
import os
from dotenv import load_dotenv
from src.semantic_matcher import SemanticMatcher

load_dotenv()

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
matcher = SemanticMatcher()

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
                
                # Initialize Semantic Matcher
                await cl.Message(content="🧠 Learning about movies...").send()
                await matcher.initialize(movies_list)
                
                count = movies_data.get("total", 0)
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

    # 1. Use Ollama to extract LIKED and HATED movies
    try:
        extraction_prompt = f"""You are a movie extraction specialist. Your job is to find movie titles in the message.
Categorize them into 'likes' and 'hates'.
Rules:
- 'likes': movies the user loves, enjoys, or mentions as positive examples.
- 'hates': movies the user dislikes, hates, or wants to avoid.
- Return ONLY a JSON object: {{"likes": ["Title 1"], "hates": ["Title 2"]}}.
- Strictly return ONLY the JSON data.

User Message: '{message.content}'"""

        response = await asyncio.to_thread(
            ollama.chat,
            model="llama3.1:latest",
            messages=[{"role": "user", "content": extraction_prompt}],
            format="json"
        )
        
        data = json.loads(response['message']['content'])
        raw_likes = data.get("likes", [])
        raw_hates = data.get("hates", [])

        # Match using Semantic Matcher
        liked_matches = await matcher.find_matches(raw_likes)
        hated_matches = await matcher.find_matches(raw_hates)
        
        # Deduplicate and ensure likes don't overlap with hates (user might be contradictory)
        hated_matches = [m for m in hated_matches if m not in liked_matches]
        
    except Exception as e:
        # Fallback to direct semantic search in text
        liked_matches = await matcher.search_in_text(message.content)
        hated_matches = []

    if not liked_matches:
        await cl.Message(content="I couldn't quite catch those movie titles. Keep in mind my database only contains movies from **1922 to 1998**. Could you try typing them exactly, or maybe mention classics from that era?").send()
        return

    # Check if we have at least 3 LIKED movies
    if len(liked_matches) < 3:
        liked_text = ", ".join([f"**{t}**" for t in liked_matches])
        hated_text = ""
        if hated_matches:
            hated_text = f" and noted that you don't like {', '.join([f'**{t}**' for t in hated_matches])}"
            
        needed = 3 - len(liked_matches)
        await cl.Message(content=f"I recognized that you like {liked_text}{hated_text}. For the best results, I need at least 3 movies you like! Could you name {needed} more?").send()
        return

    # Limit to 5 movies to satisfy API constraints
    input_titles = liked_matches[:5]

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
