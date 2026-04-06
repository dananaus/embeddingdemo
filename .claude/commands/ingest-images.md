Ingest images into the Pinecone vector database. Each image is automatically captioned using a vision model via OpenRouter, then embedded with Gemini Embedding 2.

Run the following command using the Bash tool:

```
cd "c:/ai/claudecode/embedingdemo" && "C:/Users/oliva/AppData/Local/Programs/Python/Python312/python.exe" ingest.py image --source "$ARGUMENTS"
```

If no argument is provided, ask the user for a file or folder path. Supported formats: JPG, JPEG, PNG.

After running, report the images processed, show a snippet of each generated description, and confirm they were upserted to Pinecone.
