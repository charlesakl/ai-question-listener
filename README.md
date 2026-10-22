# AI Question Listener (Windows)

A local Windows desktop prototype: click **Start Listening**, speak a question into your microphone, click **Stop & Answer**, then click **Copy Answer**.

## Setup (PowerShell)
1. Install Python 3.11 or 3.12.
2. Open PowerShell in this folder.
3. Run:
   ```
   py -m venv .venv
   .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`, and set your OpenAI API key (requires an API account with billing enabled):
   ```
   Copy-Item .env.example .env
   notepad .env
   ```
5. Launch:
   ```
   .\\.venv\\Scripts\\python.exe app.py
   ```

This prototype records the default **microphone** while listening; it does not capture system audio from Teams or Zoom, and it does not detect questions automatically. Recording is sent to OpenAI for transcription and answering after you press Stop. API usage may incur charges. Obtain consent before recording others. Keep `.env` private.
