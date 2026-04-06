"""
Ingestion CLI — embed and upsert text, images, and videos into Pinecone.

Usage examples:
  python ingest.py text  --source "path/to/docs/" --chunk-size 500
  python ingest.py image --source "path/to/images/"
  python ingest.py video --source "path/to/videos/"
  python ingest.py text  --source "single_file.txt"
  python ingest.py image --source "photo.jpg"
  python ingest.py video --source "clip.mp4"
"""

import argparse
import hashlib
import os
from pathlib import Path

from embedder import embed_text, embed_image, embed_video
from pinecone_client import get_or_create_index, upsert_vector

# ── helpers ──────────────────────────────────────────────────────────────────

def file_id(path: str, extra: str = "") -> str:
    """Stable ID from file path + optional chunk suffix."""
    return hashlib.md5(f"{path}{extra}".encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping token-approximate chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── ingestors ────────────────────────────────────────────────────────────────

def ingest_text(source: str, chunk_size: int, index):
    path = Path(source)
    files = list(path.rglob("*.txt")) + list(path.rglob("*.md")) if path.is_dir() else [path]

    for fp in files:
        print(f"[TEXT] {fp.name}")
        text = fp.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            vec_id = file_id(str(fp), str(i))
            embedding = embed_text(chunk)
            upsert_vector(index, vec_id, embedding, {
                "type": "text",
                "source": str(fp),
                "title": fp.stem,
                "chunk": i,
                "text": chunk[:500],
            })
            print(f"  chunk {i+1}/{len(chunks)} upserted")


def ingest_images(source: str, index):
    path = Path(source)
    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in path.rglob("*") if f.suffix.lower() in exts] if path.is_dir() else [path]

    for fp in files:
        print(f"[IMAGE] {fp.name}")
        embedding = embed_image(str(fp))
        upsert_vector(index, file_id(str(fp)), embedding, {
            "type": "image",
            "source": str(fp),
            "title": fp.stem,
            "filename": fp.name,
        })
        print(f"  upserted")


def ingest_videos(source: str, index):
    path = Path(source)
    exts = {".mp4", ".mov"}
    files = [f for f in path.rglob("*") if f.suffix.lower() in exts] if path.is_dir() else [path]

    for fp in files:
        print(f"[VIDEO] {fp.name}")
        embedding = embed_video(str(fp))
        upsert_vector(index, file_id(str(fp)), embedding, {
            "type": "video",
            "source": str(fp),
            "title": fp.stem,
            "filename": fp.name,
        })
        print(f"  upserted")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest content into Pinecone")
    parser.add_argument("modality", choices=["text", "image", "video"])
    parser.add_argument("--source", required=True, help="File or directory path")
    parser.add_argument("--chunk-size", type=int, default=500, help="Words per text chunk")
    args = parser.parse_args()

    index = get_or_create_index()

    if args.modality == "text":
        ingest_text(args.source, args.chunk_size, index)
    elif args.modality == "image":
        ingest_images(args.source, index)
    elif args.modality == "video":
        ingest_videos(args.source, index)

    print("\nDone.")


if __name__ == "__main__":
    main()
