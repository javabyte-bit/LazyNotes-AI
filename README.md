# LazyNotes AI

LazyNotes AI is a local study assistant built with Python and Streamlit. It lets you paste notes and instantly turn them into a summary, key points, revision questions, and a beginner-friendly explanation of a concept using a local Ollama model on your own computer.

This project keeps everything on-device. Your notes are processed locally instead of being sent to a paid cloud API, which makes it a good fit for privacy-conscious students and laptop users with limited memory.

## Features
- Summarize notes into a short, student-friendly overview
- Extract the most important ideas
- Generate revision questions and answer prompts
- Explain a concept in simple language with examples
- Run completely local using Ollama and a lightweight model

## Why this project
Many study tools rely on cloud AI services, which can be expensive or not ideal for privacy. LazyNotes AI is designed to be simple, local, and beginner-friendly, so students can use it without needing a paid subscription or external API key.

## Tech stack
- Python
- Streamlit
- Ollama
- Local LLM inference with llama3.2:3b

## Local setup

### 1. Install requirements
- Python 3.11+
- Git
- Ollama

### 2. Open PowerShell in the project folder

```powershell
cd "C:\Users\tanya\OneDrive\Desktop\slow and steady"
```

### 3. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Pull the local model

```powershell
ollama pull llama3.2:3b
```

### 6. Start the app

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## Recommended hardware note
This project was designed for systems with 8-16 GB of RAM. The default model is intentionally small and lightweight so it runs locally without requiring a large GPU or paid AI service.

## Example usage
Paste study notes like this:

```text
Photosynthesis is the process plants use to make food from sunlight, water, and carbon dioxide. It happens mainly in the leaves. Chlorophyll absorbs light energy. The plant produces glucose and releases oxygen.
```

Then click:
- Summarize
- Key points
- Revision questions
- Explain concept

## Screenshots
Capture screenshots of:
1. The app landing page with the note box
2. A successful summary result from pasted notes
3. The concept explanation output

These screenshots can be added to the repository or used in the project submission and demo.

## GitHub and challenge workflow
1. Initialize a Git repository
2. Commit the project files
3. Create a public GitHub repository
4. Push the code
5. Add a clear README and screenshots
6. Submit the project for the Hacktoberfest challenge

## License
This project is intended for personal and educational use.
