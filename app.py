"""
Chat web app — Claude Sonnet via OpenRouter + Pinecone RAG (Gemini Embedding 2).
Run: python app.py  →  http://localhost:5000
"""

import urllib.parse
from flask import Flask, render_template, request, jsonify, send_file, abort
from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from embedder import embed_query
from pinecone_client import get_or_create_index, query_index

app = Flask(__name__)

llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
SONNET_MODEL = "anthropic/claude-sonnet-4-5"

# Keep index connection alive for the session
_index = None

def get_index():
    global _index
    if _index is None:
        _index = get_or_create_index()
    return _index


def retrieve_context(question: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Embed question, query Pinecone, return formatted context + raw sources.
    Always fetches top_k overall results + top 3 per non-text modality so
    images/videos are never buried by text docs.
    """
    q_vec = embed_query(question)
    index = get_index()

    # Main search
    results = query_index(index, q_vec, top_k=top_k)
    matches = results.get("matches", [])

    # Supplemental per-type searches so images/videos always surface
    seen_ids = {m["id"] for m in matches}
    for modality in ("image", "video"):
        extra = query_index(index, q_vec, top_k=3, filter={"type": {"$eq": modality}})
        for m in extra.get("matches", []):
            if m["id"] not in seen_ids:
                matches.append(m)
                seen_ids.add(m["id"])

    # Sort combined results by score descending
    matches.sort(key=lambda m: m["score"], reverse=True)

    sources = []
    parts = []
    for i, m in enumerate(matches, 1):
        meta = m["metadata"]
        score = round(m["score"], 4)
        ctype = meta.get("type", "unknown")
        title = meta.get("title", "untitled")
        source = meta.get("source", "")

        url = None
        if ctype in ("image", "video") and source:
            url = "/media/" + urllib.parse.quote(source.replace("\\", "/"), safe="/")
        sources.append({"rank": i, "type": ctype, "title": title, "source": source, "score": score, "url": url})

        if ctype == "text":
            snippet = meta.get("text", "")
            parts.append(f"[{i}] TEXT | {title} (relevance {score})\n{snippet}")
        elif ctype == "image":
            desc = meta.get("description", "No description available.")
            parts.append(f"[{i}] IMAGE | {title} | file: {source} (relevance {score})\nDescription: {desc}")
        elif ctype == "video":
            desc = meta.get("description", "No description available.")
            parts.append(f"[{i}] VIDEO | {title} | file: {source} (relevance {score})\nDescription: {desc}")
        else:
            parts.append(f"[{i}] {ctype.upper()} | {title} (relevance {score})")

    return "\n\n".join(parts), sources


@app.route("/media/<path:filepath>")
def media(filepath):
    """Serve local media files (images, videos) by absolute path."""
    from pathlib import Path
    # filepath comes in as forward-slash path, resolve to absolute
    p = Path(filepath)
    if not p.is_absolute():
        # Try relative to working directory
        p = Path.cwd() / p
    if not p.exists():
        abort(404)
    return send_file(str(p))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])   # full conversation history
        use_rag = data.get("use_rag", True)

        user_message = messages[-1]["content"] if messages else ""

        context_text = ""
        sources = []

        if use_rag and user_message:
            context_text, sources = retrieve_context(user_message)

        # Build system prompt
        if context_text:
            system = (
                "You are a helpful assistant with access to a knowledge base of text, images, and videos.\n"
                "Use the retrieved context below to answer the user's question. "
                "If the answer isn't in the context, answer from your own knowledge and say so.\n\n"
                f"--- RETRIEVED CONTEXT ---\n{context_text}\n--- END CONTEXT ---"
            )
        else:
            system = "You are a helpful assistant."

        # Call Claude Sonnet
        response = llm.chat.completions.create(
            model=SONNET_MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(debug=True, port=5000)
