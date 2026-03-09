# Recommendation Engine - Main Branch Capabilities
**What it DOES and What it DOESN'T do**
*(Current state of the `main` branch codebase)*

---

## ✅ What It DOES:

1. **Local LLM Sentiment Extraction**
   Uses `Ollama` running `llama3.1:latest` completely locally to intelligently parse conversational user inputs. It successfully partitions mentioned movies into structured `"likes"` and `"hates"` objects.

2. **Cross-Turn Conversational Memory**
   Maintains a persistent `cl.user_session` that progressively tracks a user's liked and hated movies across multiple chat interactions. It actively mutates state (e.g., if a user changes their mind about a movie) and prompts the user until a minimum threshold of exactly 3 *liked* movies is reached.

3. **Advanced Semantic & Fuzzy Title Matching**
   Utilizes a local `SentenceTransformer` (`all-MiniLM-L6-v2`) to perform powerful semantic searches of the extracted titles against the strictly structured MovieLens 100k database. It supplements this with `RapidFuzz` to accurately catch partial strings and typos (e.g., effortlessly mapping short strings like "Psycho" to the exact canonical database title).

4. **Graceful Fallback Mechanism**
   If the local Ollama LLM crashes, takes too long, or fails to return structured JSON, the system dynamically degrades gracefully. It directly encodes the user's raw conversational text and scans the embedding matrix for potential matches within the text itself.

5. **Real-Time API Collaborative Filtering**
   Once titles are confidently matched, it sends up to 5 validated movies to the backend `FastAPI` instance, where an Item-Item Cosine Similarity matrix handles the mathematical predictive modeling.

6. **Actionable Explanations (Assignment Bonus)**
   Generates transparent, textual reasons alongside every one of the 10 recommended movies, explaining exactly *why* a movie was suggested (e.g., shared user-interaction data or genre crossover).

---

## ❌ What It DOESN'T Do:

1. **Rely on Cloud Service Billing (No Together AI)**
   Unlike experimental branches, the `main` branch does *not* require the Together AI cloud API or internet-dependent LLMs for instruction analysis. It handles everything offline using Ollama, avoiding credit limits (Error 402) entirely.

2. **Recognize Modern Movies**
   Due to the rigid scope of the MovieLens 100K dataset, the bot fundamentally cannot process or recommend any movie released after 1998 (e.g., *Avatar*, *Inception*). It is programmed to automatically warn users of this temporal boundary if matches fail.

3. **Generate "Hallucinated" AI Recommendations**
   The LLM (`llama3.1`) dictates absolutely zero logic regarding the actual recommendations. The frontend uses AI strictly as a "router" to understand the user's language and extract strings. True recommendations rely strictly on the mathematically structured Item-Item collaborative filtering matrix executed in the Python backend.

4. **Accept Unlimited Inputs**
   To respect API constraints and processing limits, it rigidly bounds user profiles, sending only a maximum of 5 of the most confidently verified "liked" movies to the backend engine for processing.
