import os
import queue
import tempfile
import threading
import wave
import tkinter as tk
from tkinter import messagebox, scrolledtext

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
RATE = 16000
CHANNELS = 1
audio_queue = queue.Queue()
recording = False
stream = None

root = tk.Tk()
root.title("AI Question Listener")
root.geometry("720x570")
status = tk.StringVar(value="Ready — click Start Listening")
question = tk.StringVar(value="Question will appear here")
tk.Label(root, text="AI Question Listener", font=("Segoe UI", 18, "bold")).pack(pady=(14, 6))
tk.Label(root, textvariable=status, fg="#225588").pack()
tk.Label(root, textvariable=question, wraplength=670, justify="left", font=("Segoe UI", 11)).pack(padx=20, pady=15)
answer = scrolledtext.ScrolledText(root, wrap="word", font=("Segoe UI", 11), height=16)
answer.pack(fill="both", expand=True, padx=20, pady=8)

def audio_callback(indata, frames, time_info, status_info):
    if recording:
        audio_queue.put(indata.copy())

def set_status(msg):
    root.after(0, lambda: status.set(msg))

def display_result(q, a):
    def update():
        question.set("Question: " + q)
        answer.delete("1.0", tk.END)
        answer.insert(tk.END, a)
        status.set("Ready — click Start Listening")
        start_btn.config(state="normal")
        stop_btn.config(state="disabled")
    root.after(0, update)

def process_audio(chunks):
    path = None
    try:
        set_status("Transcribing and generating answer...")
        samples = np.concatenate(chunks, axis=0)
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        with wave.open(path, "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(pcm.tobytes())
        client = OpenAI()
        with open(path, "rb") as f:
            transcript = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
        q = transcript.text.strip()
        if not q:
            raise ValueError("No speech detected. Try again.")
        response = client.responses.create(
            model="gpt-4.1-mini", max_output_tokens=120,
            instructions="""
You are an expert technical interview preparation assistant.
Answer questions accurately in 30-50 words.
Be direct, practical, and conversational.

For troubleshooting, give the top 2-3 diagnostic steps.
For comparisons, state the key difference first.
Include commands or metrics when useful.
Avoid introductions, repetition, and unnecessary details.
Never invent facts or personal experience.
EXPERTISE:
- Site Reliability Engineering (SRE)
- Kubernetes and OpenShift
- AWS and cloud infrastructure
- Linux and production troubleshooting
- Observability, monitoring, and incident management
- Event-driven remediation and Infrastructure as Code
- AIOps and intelligent automation
- Cybersecurity and operational resilience
- APIs, databases, and distributed systems
- Engineering leadership

RULES:
1. Answer the exact question directly.
2. Keep answers between 30 and 60 words.
3. Use natural, confident, professional language.
4. Prioritize technical accuracy and practical knowledge.
5. Avoid unnecessary introductions and conclusions.
6. For comparisons, explain the key difference first.
7. For troubleshooting, prioritize 2-3 diagnostic steps.
8. Include relevant commands and metrics when useful.
9. For behavioral questions, use a concise STAR approach.
10. Never invent facts or personal experience.
11. If the question is unclear, ask for clarification.

Make answers sound natural when spoken during an interview.
""",
            input=q,
        )
        display_result(q, response.output_text)   
    except Exception as exc:
        def show_error():
            status.set("Error — please try again")
            start_btn.config(state="normal")
            stop_btn.config(state="disabled")
            messagebox.showerror("AI Listener", str(exc))
        root.after(0, show_error)
    finally:
        if path and os.path.exists(path):
            os.remove(path)

def start():
    global recording, stream
    if not os.getenv("OPENAI_API_KEY"):
        messagebox.showerror("Missing API key", "Add OPENAI_API_KEY to your .env file.")
        return
    while not audio_queue.empty():
        audio_queue.get_nowait()
    try:
        stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="float32", callback=audio_callback)
        stream.start()
    except Exception as exc:
        messagebox.showerror("Microphone error", str(exc))
        return
    recording = True
    status.set("Listening... ask a question, then click Stop & Answer")
    start_btn.config(state="disabled")
    stop_btn.config(state="normal")

def stop():
    global recording, stream
    recording = False
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get_nowait())
    stop_btn.config(state="disabled")
    if not chunks:
        status.set("No audio recorded")
        start_btn.config(state="normal")
        return
    threading.Thread(target=process_audio, args=(chunks,), daemon=True).start()

def copy_answer():
    content = answer.get("1.0", tk.END).strip()
    if content:
        root.clipboard_clear()
        root.clipboard_append(content)
        status.set("Answer copied to clipboard")

buttons = tk.Frame(root)
buttons.pack(pady=12)
start_btn = tk.Button(buttons, text="Start Listening", command=start, width=18)
start_btn.pack(side="left", padx=6)
stop_btn = tk.Button(buttons, text="Stop & Answer", command=stop, state="disabled", width=18)
stop_btn.pack(side="left", padx=6)
tk.Button(buttons, text="Copy Answer", command=copy_answer, width=18).pack(side="left", padx=6)
root.mainloop()
