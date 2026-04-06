"""
RAG query CLI — search Pinecone then answer with an OpenRouter LLM.

Usage:
  python query.py "what videos show a sunset?"
  python query.py "summarize the onboarding docs" --type text
  python query.py "find images of cats" --type image --top-k 3
"""

import argparse
import json

from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from embedder import embed_query
from pinecone_client import get_or_create_index, query_index

# OpenRouter client (OpenAI-compatible)
llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

LLM_MODEL = "google/gemini-2.0-flash-001"


def build_context(matches) -> str:
    parts = []
    for i, m in enumerate(matches, 1):
        meta = m["metadata"]
        content_type = meta.get("type", "unknown")
        title = meta.get("title", "untitled")
        source = meta.get("source", "")
        score = round(m["score"], 4)

        if content_type == "text":
            snippet = meta.get("text", "")
            parts.append(f"[{i}] TEXT | {title} (score {score})\n{snippet}")
        elif content_type == "image":
            parts.append(f"[{i}] IMAGE | {title} | file: {source} (score {score})")
        elif content_type == "video":
            parts.append(f"[{i}] VIDEO | {title} | file: {source} (score {score})")
        else:
            parts.append(f"[{i}] {content_type.upper()} | {title} (score {score})")

    return "\n\n".join(parts)


def rag_query(question: str, top_k: int = 5, type_filter: str = None) -> str:
    embedding = embed_query(question)
    index = get_or_create_index()

    filter_dict = {"type": {"$eq": type_filter}} if type_filter else None
    results = query_index(index, embedding, top_k=top_k, filter=filter_dict)
    matches = results.get("matches", [])

    if not matches:
        return "No relevant results found in the knowledge base."

    context = build_context(matches)

    prompt = f"""You are a helpful assistant. Use the retrieved context below to answer the user's question.
If the answer isn't in the context, say so honestly.

--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}
Answer:"""

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="RAG query against the Pinecone index")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--type", choices=["text", "image", "video"], default=None,
                        help="Filter results by content type")
    args = parser.parse_args()

    print(f"\nSearching for: {args.question}\n")
    answer = rag_query(args.question, top_k=args.top_k, type_filter=args.type)
    print("Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
