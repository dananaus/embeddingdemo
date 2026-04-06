Ingest video files into the Pinecone vector database using Gemini Embedding 2 (via the Gemini File API for upload).

Run the following command using the Bash tool:

```
cd "c:/ai/claudecode/embedingdemo" && "C:/Users/oliva/AppData/Local/Programs/Python/Python312/python.exe" ingest.py video --source "$ARGUMENTS"
```

If no argument is provided, ask the user for a file or folder path. Supported formats: MP4, MOV (max 120 seconds each).

Note: Videos are uploaded to the Gemini File API for processing before embedding — this may take a moment per file.

After running, report the videos processed and confirm they were upserted to Pinecone.
