Start the embedding demo chat web app on http://localhost:5000.

The app uses Claude Sonnet via OpenRouter for chat, Gemini Embedding 2 for semantic search, and Pinecone as the vector store. Images are displayed inline in the chat UI.

First, check if the server is already running:

```
curl -sk http://localhost:5000 -o /dev/null -w "%{http_code}"
```

If it returns 200, tell the user the server is already running at http://localhost:5000.

If not running, start it with the Bash tool using run_in_background=true:

```
cd "c:/ai/claudecode/embedingdemo" && "C:/Users/oliva/AppData/Local/Programs/Python/Python312/python.exe" app.py
```

Wait 3 seconds, then verify it started by hitting http://localhost:5000 again.

Tell the user the server is live at http://localhost:5000 and remind them:
- RAG toggle in the top-right to enable/disable Pinecone search
- Click any retrieved image to open it fullscreen
- Shift+Enter for a new line, Enter to send
