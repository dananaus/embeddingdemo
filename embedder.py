"""
Wrapper around Gemini Embedding 2 Preview (multimodal).
Supports: text, image files (PNG/JPEG), video files (MP4/MOV via File API).
"""

import base64
import mimetypes
import time
from pathlib import Path

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

client = genai.Client(api_key=GEMINI_API_KEY)


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a plain text string."""
    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def embed_image(image_path: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a local image file (PNG or JPEG)."""
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in ("image/png", "image/jpeg"):
        raise ValueError(f"Unsupported image type: {mime}. Use PNG or JPEG.")

    with open(path, "rb") as f:
        image_bytes = f.read()

    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=types.Content(
            parts=[types.Part(inline_data=types.Blob(mime_type=mime, data=image_bytes))]
        ),
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def embed_video(video_path: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Embed a local video file (MP4/MOV, max 120 seconds).
    Uses the Gemini File API to upload first, then embeds.
    """
    path = Path(video_path)
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in ("video/mp4", "video/quicktime"):
        raise ValueError(f"Unsupported video type: {mime}. Use MP4 or MOV.")

    print(f"  Uploading {path.name} to Gemini File API...")
    uploaded = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(mime_type=mime),
    )

    # Wait for processing
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"File upload failed with state: {uploaded.state.name}")

    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=types.Content(
            parts=[types.Part(file_data=types.FileData(
                mime_type=mime, file_uri=uploaded.uri
            ))]
        ),
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    # Clean up uploaded file
    client.files.delete(name=uploaded.name)

    return result.embeddings[0].values


def embed_query(query: str) -> list[float]:
    """Embed a user search query (uses RETRIEVAL_QUERY task type)."""
    return embed_text(query, task_type="RETRIEVAL_QUERY")
