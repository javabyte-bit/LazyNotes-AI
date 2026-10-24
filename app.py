import ipaddress
import os
from urllib.parse import urlparse

import requests
import streamlit as st
from streamlit.components.v1 import html as components_html

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LIST_MODELS_URL = os.getenv("OLLAMA_MODELS_URL", "http://localhost:11434/api/tags")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

DARK_THEME = {
    "bg": "#151515",
    "bg_alt": "#1E1E1E",
    "card": "#242424",
    "card_alt": "#2A2A2A",
    "border": "#454545",
    "primary": "#E7BC52",
    "primary_soft": "rgba(231, 188, 82, 0.14)",
    "text": "#F6F1E6",
    "muted": "#D3CBB8",
    "shadow": "rgba(0, 0, 0, 0.35)",
    "input_bg": "#1A1A1A",
    "mascot_fill": "#FFF8FC",
    "mascot_accent": "#F7A8D8",
    "mascot_detail": "#E9DDE7",
    "mascot_outline": "#19151D",
    "mascot_face": "#19151D",
    "mascot_cheek": "#F7A8D8",
    "mascot_focus": "#F5C542",
    "rookie_fill": "#F5C542",
    "rookie_detail": "#FFF1C7",
    "rookie_outline": "#30271D",
    "fashionista_fill": "#F7A8D8",
    "fashionista_detail": "#FFF9FC",
    "fashionista_outline": "#17131A",
    "golden_fill": "#E4AD38",
    "golden_detail": "#FFF0B5",
    "golden_outline": "#342817",
    "bot_fill": "#E9E5EB",
    "bot_detail": "#FFF9FC",
    "bot_outline": "#302B35",
    "bot_glow": "#FF72BB",
    "clumsy_fill": "#FFF8FC",
    "clumsy_detail": "#FFD5EA",
    "clumsy_outline": "#17131A",
}

LIGHT_THEME = {
    "bg": "#FFF7FB",
    "bg_alt": "#FFF1F8",
    "card": "#FFFCFE",
    "card_alt": "#FFF4FA",
    "border": "#EFC7DD",
    "primary": "#D94B98",
    "primary_soft": "rgba(247, 168, 216, 0.18)",
    "text": "#17131A",
    "muted": "#574A56",
    "shadow": "rgba(64, 28, 51, 0.1)",
    "input_bg": "#FFFFFF",
    "mascot_fill": "#F7A8D8",
    "mascot_accent": "#FF72BB",
    "mascot_detail": "#FFF9FC",
    "mascot_outline": "#17131A",
    "mascot_face": "#17131A",
    "mascot_cheek": "#FFD5EA",
    "mascot_focus": "#D94B98",
    "rookie_fill": "#F7A8D8",
    "rookie_detail": "#FFF1F8",
    "rookie_outline": "#17131A",
    "fashionista_fill": "#F7A8D8",
    "fashionista_detail": "#FFF9FC",
    "fashionista_outline": "#17131A",
    "golden_fill": "#E4AD38",
    "golden_detail": "#FFF0B5",
    "golden_outline": "#342817",
    "bot_fill": "#E9E5EB",
    "bot_detail": "#FFF9FC",
    "bot_outline": "#302B35",
    "bot_glow": "#D94B98",
    "clumsy_fill": "#FFF8FC",
    "clumsy_detail": "#FFD5EA",
    "clumsy_outline": "#17131A",
}


def get_theme() -> str:
    query_theme = st.query_params.get("theme", "dark")
    if isinstance(query_theme, list):
        selected = query_theme[0] if query_theme else "dark"
    else:
        selected = query_theme
    return selected if selected in {"light", "dark"} else "dark"


def set_theme(theme: str) -> None:
    st.session_state["theme"] = theme
    st.query_params["theme"] = theme


def persist_theme_choice() -> None:
    selected = st.session_state.get("appearance_theme")
    if selected in {"light", "dark"}:
        st.query_params["theme"] = selected


def get_ollama_status() -> tuple[bool, list[str]]:
    try:
        response = requests.get(LIST_MODELS_URL, timeout=20)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        return True, [model.get("name", "") for model in models if model.get("name")]
    except (requests.RequestException, ValueError):
        return False, []


def get_available_models() -> list[str]:
    return get_ollama_status()[1]


def ollama_is_local() -> bool:
    hostname = urlparse(OLLAMA_URL).hostname
    if hostname == "localhost":
        return True
    try:
        return bool(hostname and ipaddress.ip_address(hostname).is_loopback)
    except ValueError:
        return False


def ensure_model_available(model_name: str) -> str:
    if not model_name.strip():
        raise ValueError("Please choose an Ollama model before generating output.")

    available_models = get_available_models()
    if available_models and model_name not in available_models:
        raise ValueError(
            f"Model '{model_name}' is not installed locally. "
            "Open Ollama and run: ollama pull " + model_name
        )

    if not available_models:
        raise ValueError(
            "Could not reach Ollama. Check that Ollama is running locally on your computer."
        )

    return model_name.strip()


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    model = ensure_model_available(model)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 400,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(
            "The local Ollama server is not responding. Start Ollama and try again."
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("The model returned an invalid response. Please try again.") from exc

    return data.get("response", "").strip()


def summarize_notes(notes: str, model: str = DEFAULT_MODEL) -> str:
    prompt = f"""You are a helpful student study assistant.

Read the notes below and produce a clear, student-friendly summary.
Keep it simple, accurate, and focused on the main ideas.

Notes:
{notes}

Return:
1. A short summary in 4-6 sentences.
2. A heading called 'Key ideas:' followed by 3-5 bullet points.
3. A heading called 'Important terms:' followed by 3-5 important terms.
"""
    return call_ollama(prompt, model=model, temperature=0.2)


def key_points(notes: str, model: str = DEFAULT_MODEL) -> str:
    prompt = f"""List the most important facts, ideas, and concepts from these notes.
Focus on what a student should remember for an exam or revision.

Notes:
{notes}

Output only bullet points.
"""
    return call_ollama(prompt, model=model, temperature=0.2)


def revision_questions(notes: str, model: str = DEFAULT_MODEL) -> str:
    prompt = f"""Create 5 short revision questions based on the notes below.
Each question should test understanding, not just memory.
Then add a short answer key.

Notes:
{notes}
"""
    return call_ollama(prompt, model=model, temperature=0.4)


def explain_concept(notes: str, concept: str, model: str = DEFAULT_MODEL) -> str:
    prompt = f"""The student is learning about the concept '{concept}'.
Use the notes below to explain this concept in a clear, beginner-friendly way.
Include:
- a simple explanation
- a real-world example
- why it matters
- 2 short questions the student can ask themselves

Notes:
{notes}
"""
    return call_ollama(prompt, model=model, temperature=0.3)


def reset_workflow_outputs() -> None:
    for key in ("study_summary", "study_key_points", "study_revision_questions"):
        st.session_state.pop(key, None)
    st.session_state["workflow_reviewed"] = False


def render_mascot(theme_name: str) -> None:
    palette = DARK_THEME if theme_name == "dark" else LIGHT_THEME
    mascot_html = """
        <div class="mascot-stage" aria-label="LazyNotes mascots">
            <div class="mascot-bubble" id="mascot-bubble" role="status" aria-live="polite">You got this!</div>
            <div class="mascot-unit rookie-unit">
                <button class="mascot-button rookie" type="button" data-name="Rookie" aria-label="Rookie, the playful golden mascot. Click for encouragement.">
                    <svg class="bean-art" viewBox="0 0 150 170" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
                        <ellipse class="bean-shadow" cx="75" cy="160" rx="33" ry="5"/>
                        <path class="bean-leg bean-leg-left" d="M48 137c-5 7-7 15-2 19 4 3 13 3 18 0 4-3 2-10 0-16Z"/>
                        <path class="bean-leg bean-leg-right" d="M102 137c5 7 7 15 2 19-4 3-13 3-18 0-4-3-2-10 0-16Z"/>
                        <path class="bean-arm bean-arm-left" d="M42 76c-13 2-20 10-18 18 1 6 7 9 13 6l11-11Z"/>
                        <path class="bean-body" d="M75 22c-19 0-30 13-31 30-13 6-20 19-19 36l5 43c2 17 19 25 45 25s43-8 45-25l5-43c1-17-6-30-19-36-1-17-12-30-31-30Z"/>
                        <path class="bean-belly rookie-belly" d="M53 112c1 22 7 33 22 33s21-11 22-33c-13 5-31 5-44 0Z"/>
                        <path class="bean-belly-shine" d="M59 124c3 9 8 13 16 13"/>
                        <ellipse class="bean-eye-white" cx="57" cy="79" rx="12" ry="15"/>
                        <ellipse class="bean-eye-white" cx="92" cy="79" rx="12" ry="15"/>
                        <ellipse class="bean-pupil" cx="60" cy="82" rx="7" ry="10"/>
                        <ellipse class="bean-pupil" cx="95" cy="82" rx="7" ry="10"/>
                        <circle class="bean-glint" cx="57" cy="76" r="3.5"/>
                        <circle class="bean-glint" cx="92" cy="76" r="3.5"/>
                        <circle class="bean-glint-small" cx="64" cy="86" r="1.5"/>
                        <circle class="bean-glint-small" cx="99" cy="86" r="1.5"/>
                        <path class="bean-eye-blink" d="M46 81c6 8 16 8 22 0m13 0c6 8 16 8 22 0"/>
                        <path class="bean-brow rookie-brow" d="M47 61c5-4 10-4 15-1m23-1c5-2 10-1 14 3"/>
                        <ellipse class="bean-cheek rookie-cheek" cx="43" cy="99" rx="7" ry="4"/>
                        <ellipse class="bean-cheek rookie-cheek" cx="107" cy="99" rx="7" ry="4"/>
                        <path class="bean-mouth" d="M67 103c3 4 6 5 8 5s5-1 8-5"/>
                        <g class="bean-arm-wave">
                            <path class="bean-arm rookie-arm" d="M107 78c11 1 17 7 17 15l-1 8c-1 5-6 7-10 4-5-3-8-12-13-19Z"/>
                            <circle class="bean-hand rookie-hand" cx="122" cy="91" r="8"/>
                            <path class="bean-hand-shine" d="M119 87c2-2 4-2 6 0"/>
                        </g>
                        <path class="rookie-star" d="m35 47 3 6 7 1-5 5 1 7-6-3-6 3 1-7-5-5 7-1Z"/>
                        <path class="bean-sparkle sparkle-a" d="m121 39 2 4 5 1-4 3 1 5-4-2-4 2 1-5-4-3 5-1Z"/>
                    </svg>
                    <span class="mascot-name">Rookie</span>
                </button>
            </div>
            <div class="mascot-unit fashionista-unit">
                <button class="mascot-button fashionista" type="button" data-name="Fashionista" aria-label="Fashionista, the stylish pink mascot. Click for encouragement.">
                    <svg class="bean-art" viewBox="0 0 150 170" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
                        <ellipse class="bean-shadow" cx="75" cy="160" rx="33" ry="5"/>
                        <path class="bean-leg bean-leg-left" d="M48 137c-5 7-7 15-2 19 4 3 13 3 18 0 4-3 2-10 0-16Z"/>
                        <path class="bean-leg bean-leg-right" d="M102 137c5 7 7 15 2 19-4 3-13 3-18 0-4-3-2-10 0-16Z"/>
                        <path class="bean-arm bean-arm-left" d="M42 76c-13 2-20 10-18 18 1 6 7 9 13 6l11-11Z"/>
                        <path class="bean-body" d="M75 22c-19 0-30 13-31 30-13 6-20 19-19 36l5 43c2 17 19 25 45 25s43-8 45-25l5-43c1-17-6-30-19-36-1-17-12-30-31-30Z"/>
                        <path class="bean-belly fashionista-belly" d="M53 112c1 22 7 33 22 33s21-11 22-33c-13 5-31 5-44 0Z"/>
                        <path class="bean-belly-shine" d="M59 124c3 9 8 13 16 13"/>
                        <path class="fashionista-mask" d="M43 71c8-13 19-19 32-19s24 6 32 19c-8 12-18 18-32 18S51 83 43 71Z"/>
                        <ellipse class="bean-eye-white" cx="57" cy="78" rx="12" ry="15"/>
                        <ellipse class="bean-eye-white" cx="92" cy="78" rx="12" ry="15"/>
                        <ellipse class="bean-pupil" cx="60" cy="81" rx="7" ry="10"/>
                        <ellipse class="bean-pupil" cx="95" cy="81" rx="7" ry="10"/>
                        <circle class="bean-glint" cx="57" cy="75" r="3.5"/>
                        <circle class="bean-glint" cx="92" cy="75" r="3.5"/>
                        <circle class="bean-glint-small" cx="64" cy="85" r="1.5"/>
                        <circle class="bean-glint-small" cx="99" cy="85" r="1.5"/>
                        <path class="bean-eye-blink" d="M46 80c6 8 16 8 22 0m13 0c6 8 16 8 22 0"/>
                        <ellipse class="bean-cheek fashionista-cheek" cx="43" cy="99" rx="7" ry="4"/>
                        <ellipse class="bean-cheek fashionista-cheek" cx="107" cy="99" rx="7" ry="4"/>
                        <path class="bean-mouth" d="M67 103c3 4 6 5 8 5s5-1 8-5"/>
                        <g class="bean-bow">
                            <path d="M75 35c-13-12-19-2-12 7 4 4 9 2 12-1Zm0 0c13-12 19-2 12 7-4 4-9 2-12-1Z"/>
                            <circle cx="75" cy="36" r="3.5"/>
                        </g>
                        <path class="fashionista-star" d="m109 111 3 6 7 1-5 5 1 7-6-3-6 3 1-7-5-5 7-1Z"/>
                        <g class="bean-arm-wave">
                            <path class="bean-arm fashionista-arm" d="M107 78c11 1 17 7 17 15l-1 8c-1 5-6 7-10 4-5-3-8-12-13-19Z"/>
                            <circle class="bean-hand fashionista-hand" cx="122" cy="91" r="8"/>
                            <path class="bean-hand-shine" d="M119 87c2-2 4-2 6 0"/>
                        </g>
                        <path class="bean-sparkle sparkle-b" d="m31 45 2 4 5 1-4 3 1 5-4-2-4 2 1-5-4-3 5-1Z"/>
                    </svg>
                    <span class="mascot-name">Fashionista</span>
                </button>
            </div>
            <div class="mascot-unit golden-guy-unit">
                <button class="mascot-button golden-guy" type="button" data-name="Golden Guy" aria-label="Golden Guy, the confident golden mascot. Click for encouragement.">
                    <svg class="bean-art" viewBox="0 0 150 170" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
                        <ellipse class="bean-shadow" cx="75" cy="160" rx="33" ry="5"/>
                        <path class="bean-leg bean-leg-left" d="M48 137c-5 7-7 15-2 19 4 3 13 3 18 0 4-3 2-10 0-16Z"/>
                        <path class="bean-leg bean-leg-right" d="M102 137c5 7 7 15 2 19-4 3-13 3-18 0-4-3-2-10 0-16Z"/>
                        <path class="bean-arm bean-arm-left" d="M42 76c-13 2-20 10-18 18 1 6 7 9 13 6l11-11Z"/>
                        <path class="bean-body" d="M75 22c-19 0-30 13-31 30-13 6-20 19-19 36l5 43c2 17 19 25 45 25s43-8 45-25l5-43c1-17-6-30-19-36-1-17-12-30-31-30Z"/>
                        <path class="bean-belly golden-belly" d="M53 112c1 22 7 33 22 33s21-11 22-33c-13 5-31 5-44 0Z"/>
                        <path class="bean-belly-shine" d="M59 124c3 9 8 13 16 13"/>
                        <ellipse class="bean-eye-white" cx="57" cy="79" rx="12" ry="15"/>
                        <ellipse class="bean-eye-white" cx="92" cy="79" rx="12" ry="15"/>
                        <ellipse class="bean-pupil" cx="60" cy="82" rx="7" ry="10"/>
                        <ellipse class="bean-pupil" cx="95" cy="82" rx="7" ry="10"/>
                        <circle class="bean-glint" cx="57" cy="76" r="3.5"/>
                        <circle class="bean-glint" cx="92" cy="76" r="3.5"/>
                        <path class="bean-eye-blink" d="M46 81c6 8 16 8 22 0m13 0c6 8 16 8 22 0"/>
                        <path class="bean-brow" d="M47 61c5-3 10-3 15 0m23 0c5-3 10-3 14 0"/>
                        <ellipse class="bean-cheek" cx="43" cy="99" rx="7" ry="4"/>
                        <ellipse class="bean-cheek" cx="107" cy="99" rx="7" ry="4"/>
                        <path class="bean-mouth" d="M67 103c3 4 6 5 8 5s5-1 8-5"/>
                        <g class="golden-crown">
                            <path d="m55 45 4-14 12 10 4-17 5 17 12-10 4 14Z"/>
                            <path class="crown-band" d="M56 45h38v6H56z"/>
                            <circle cx="60" cy="32" r="2"/><circle cx="75" cy="27" r="2"/><circle cx="91" cy="32" r="2"/>
                        </g>
                        <g class="bean-arm-wave">
                            <path class="bean-arm" d="M107 78c11 1 17 7 17 15l-1 8c-1 5-6 7-10 4-5-3-8-12-13-19Z"/>
                            <circle class="bean-hand" cx="122" cy="91" r="8"/>
                            <path class="bean-hand-shine" d="M119 87c2-2 4-2 6 0"/>
                        </g>
                        <path class="bean-sparkle golden-sparkle-one" d="m119 53 2 5 5 1-4 4 1 5-4-2-4 2 1-5-4-4 5-1Z"/>
                        <path class="bean-sparkle golden-sparkle-two" d="m31 54 2 4 4 1-3 3 1 4-4-2-3 2 1-4-3-3 4-1Z"/>
                    </svg>
                    <span class="mascot-name">Golden Guy</span>
                </button>
            </div>
            <div class="mascot-unit bean-bot-unit">
                <button class="mascot-button bean-bot" type="button" data-name="Bean Bot" aria-label="Bean Bot, the curious helper robot. Click for encouragement.">
                    <svg class="bean-art" viewBox="0 0 150 170" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
                        <ellipse class="bean-shadow" cx="75" cy="160" rx="33" ry="5"/>
                        <path class="bot-leg" d="M51 138v13c0 5 5 7 10 7s9-2 9-7v-13Zm29 0v13c0 5 4 7 9 7s10-2 10-7v-13Z"/>
                        <path class="bot-arm" d="M37 83c-10 2-15 9-13 16l3 7c2 4 7 5 10 1l8-13Zm76 0c10 2 15 9 13 16l-3 7c-2 4-7 5-10 1l-8-13Z"/>
                        <rect class="bot-body" x="39" y="42" width="72" height="101" rx="31"/>
                        <path class="bot-belly" d="M53 112h44v19H53z"/>
                        <path class="bot-antenna" d="M75 42V29"/>
                        <circle class="bot-antenna-tip" cx="75" cy="25" r="7"/>
                        <rect class="bot-screen" x="46" y="57" width="58" height="45" rx="17"/>
                        <ellipse class="bot-eye" cx="63" cy="78" rx="5" ry="8"/>
                        <ellipse class="bot-eye" cx="87" cy="78" rx="5" ry="8"/>
                        <circle class="bot-eye-glint" cx="61" cy="75" r="2"/>
                        <circle class="bot-eye-glint" cx="85" cy="75" r="2"/>
                        <path class="bot-smile" d="M68 90c4 4 10 4 14 0"/>
                        <circle class="bot-led led-one" cx="58" cy="120" r="3"/>
                        <circle class="bot-led led-two" cx="70" cy="120" r="3"/>
                        <circle class="bot-led led-three" cx="82" cy="120" r="3"/>
                        <rect class="bot-side-light" x="33" y="76" width="6" height="20" rx="3"/>
                        <rect class="bot-side-light" x="111" y="76" width="6" height="20" rx="3"/>
                        <path class="bean-sparkle bot-sparkle" d="m114 48 2 5 5 1-4 3 1 5-4-2-4 2 1-5-4-3 5-1Z"/>
                    </svg>
                    <span class="mascot-name">Bean Bot</span>
                </button>
            </div>
            <div class="mascot-unit clumsy-babe-unit">
                <button class="mascot-button clumsy-babe" type="button" data-name="Clumsy Babe" aria-label="Clumsy Babe, the shy and sweet mascot. Click for encouragement.">
                    <svg class="bean-art" viewBox="0 0 150 170" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
                        <ellipse class="bean-shadow" cx="75" cy="160" rx="33" ry="5"/>
                        <path class="bean-leg bean-leg-left" d="M48 137c-5 7-7 15-2 19 4 3 13 3 18 0 4-3 2-10 0-16Z"/>
                        <path class="bean-leg bean-leg-right" d="M102 137c5 7 7 15 2 19-4 3-13 3-18 0-4-3-2-10 0-16Z"/>
                        <path class="bean-arm bean-arm-left" d="M42 76c-13 2-20 10-18 18 1 6 7 9 13 6l11-11Z"/>
                        <path class="bean-body" d="M75 22c-19 0-30 13-31 30-13 6-20 19-19 36l5 43c2 17 19 25 45 25s43-8 45-25l5-43c1-17-6-30-19-36-1-17-12-30-31-30Z"/>
                        <path class="bean-belly clumsy-belly" d="M53 112c1 22 7 33 22 33s21-11 22-33c-13 5-31 5-44 0Z"/>
                        <path class="bean-belly-shine" d="M59 124c3 9 8 13 16 13"/>
                        <ellipse class="bean-eye-white" cx="57" cy="79" rx="12" ry="15"/>
                        <ellipse class="bean-eye-white" cx="92" cy="79" rx="12" ry="15"/>
                        <ellipse class="bean-pupil clumsy-pupil" cx="60" cy="83" rx="6" ry="8"/>
                        <ellipse class="bean-pupil clumsy-pupil" cx="95" cy="83" rx="6" ry="8"/>
                        <circle class="bean-glint" cx="57" cy="76" r="3.5"/>
                        <circle class="bean-glint" cx="92" cy="76" r="3.5"/>
                        <path class="bean-eye-blink" d="M46 81c6 8 16 8 22 0m13 0c6 8 16 8 22 0"/>
                        <ellipse class="bean-cheek clumsy-cheek" cx="43" cy="99" rx="8" ry="5"/>
                        <ellipse class="bean-cheek clumsy-cheek" cx="107" cy="99" rx="8" ry="5"/>
                        <path class="bean-mouth clumsy-mouth" d="M68 105c2 2 4 3 6 2 2-1 4-1 6 1"/>
                        <path class="clumsy-tuft" d="M68 28c2-8 8-10 13-7 4 2 5 6 4 10"/>
                        <g class="bean-arm-wave">
                            <path class="bean-arm" d="M107 78c11 1 17 7 17 15l-1 8c-1 5-6 7-10 4-5-3-8-12-13-19Z"/>
                            <circle class="bean-hand" cx="122" cy="91" r="8"/>
                            <path class="bean-hand-shine" d="M119 87c2-2 4-2 6 0"/>
                        </g>
                        <path class="bean-sparkle clumsy-heart" d="M117 47c-5-6-12 2 0 12 12-10 5-18 0-12Z"/>
                    </svg>
                    <span class="mascot-name">Clumsy Babe</span>
                </button>
            </div>
        </div>

        <script>
            const stage = document.querySelector('.mascot-stage');
            const bubble = document.getElementById('mascot-bubble');
            const mascots = document.querySelectorAll('.mascot-button');
            const jumpTimers = new WeakMap();
            const lastMessages = new Map();
            let bubbleTimer = null;
            const messages = {
                'Rookie': [
                    'You can do it! 💪', 'One step at a time!',
                    "You're doing better than you think!", "Keep going, you've got this!",
                    'Small progress is still progress!'
                ],
                'Fashionista': [
                    "You're absolutely iconic! ✨", 'Look at you, making progress!',
                    'Main character energy!', 'Your future self is proud of you!',
                    'Confidence looks good on you!'
                ],
                'Golden Guy': [
                    'Think big. Start small. Win daily. 👑', 'Your future is worth the effort!',
                    'Success starts with showing up!', 'Boss mindset activated!',
                    'Invest in yourself today!'
                ],
                'Bean Bot': [
                    'Brainpower: activated! 🤖', 'Error 404: giving up not found!',
                    'One problem at a time, human!', "You've got a strong mindset!",
                    'Processing progress... 100% awesome!'
                ],
                'Clumsy Babe': [
                    'Um... I believe in you! 💗', "It's okay to take your time.",
                    "You don't have to be perfect.", "You're stronger than you realise!",
                    'Tiny steps count, okay?'
                ]
            };

            function chooseMessage(name) {
                const options = messages[name] || messages.Rookie;
                const previous = lastMessages.get(name);
                const freshOptions = options.filter((message) => message !== previous);
                const pool = freshOptions.length ? freshOptions : options;
                const choice = pool[Math.floor(Math.random() * pool.length)];
                lastMessages.set(name, choice);
                return choice;
            }

            function showBubbleFor(mascot) {
                const mascotRect = mascot.getBoundingClientRect();
                const stageRect = stage.getBoundingClientRect();
                bubble.textContent = chooseMessage(mascot.dataset.name);
                bubble.classList.add('show');
                const width = bubble.offsetWidth;
                const center = mascotRect.left - stageRect.left + mascotRect.width / 2;
                const safeCenter = Math.max(width / 2 + 8, Math.min(stage.clientWidth - width / 2 - 8, center));
                bubble.style.left = safeCenter + 'px';
                bubble.style.top = Math.max(2, mascotRect.top - stageRect.top - bubble.offsetHeight - 3) + 'px';
                clearTimeout(bubbleTimer);
                bubbleTimer = setTimeout(() => bubble.classList.remove('show'), 2400);
            }

            mascots.forEach((mascot) => {
                let blinkTimer = null;
                mascot.addEventListener('pointerenter', () => {
                    mascot.classList.add('is-hovering', 'is-blinking');
                    clearTimeout(blinkTimer);
                    blinkTimer = setTimeout(() => mascot.classList.remove('is-blinking'), 240);
                });
                mascot.addEventListener('pointerleave', () => mascot.classList.remove('is-hovering'));
                mascot.addEventListener('click', () => {
                    mascot.classList.remove('is-jumping', 'is-sparkling');
                    void mascot.offsetWidth;
                    mascot.classList.add('is-jumping', 'is-sparkling');
                    showBubbleFor(mascot);
                    clearTimeout(jumpTimers.get(mascot));
                    jumpTimers.set(mascot, setTimeout(() => mascot.classList.remove('is-jumping', 'is-sparkling'), 780));
                });
            });

        </script>
        """

    components_html(
        f"""
        <style>
        :root {{
            --bg: {palette['bg']};
            --bg-alt: {palette['bg_alt']};
            --card: {palette['card']};
            --card-alt: {palette['card_alt']};
            --border: {palette['border']};
            --primary: {palette['primary']};
            --primary-soft: {palette['primary_soft']};
            --text: {palette['text']};
            --muted: {palette['muted']};
            --shadow: {palette['shadow']};
            --input-bg: {palette['input_bg']};
            --mascot-fill: {palette['mascot_fill']};
            --mascot-accent: {palette['mascot_accent']};
            --mascot-detail: {palette['mascot_detail']};
            --mascot-outline: {palette['mascot_outline']};
            --mascot-face: {palette['mascot_face']};
            --mascot-cheek: {palette['mascot_cheek']};
            --rookie-fill: {palette['rookie_fill']};
            --rookie-detail: {palette['rookie_detail']};
            --rookie-outline: {palette['rookie_outline']};
            --fashionista-fill: {palette['fashionista_fill']};
            --fashionista-detail: {palette['fashionista_detail']};
            --fashionista-outline: {palette['fashionista_outline']};
            --mascot-focus: {palette['mascot_focus']};
            --golden-fill: {palette['golden_fill']};
            --golden-detail: {palette['golden_detail']};
            --golden-outline: {palette['golden_outline']};
            --bot-fill: {palette['bot_fill']};
            --bot-detail: {palette['bot_detail']};
            --bot-outline: {palette['bot_outline']};
            --bot-glow: {palette['bot_glow']};
            --clumsy-fill: {palette['clumsy_fill']};
            --clumsy-detail: {palette['clumsy_detail']};
            --clumsy-outline: {palette['clumsy_outline']};
        }}

        .mascot-stage {{
            position: relative;
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 146px));
            grid-auto-rows: 150px;
            align-content: center;
            justify-content: center;
            gap: 0 10px;
            width: 100%;
            height: 212px;
            padding: 4px 0;
            box-sizing: border-box;
            pointer-events: none;
        }}

        .mascot-bubble {{
            position: absolute;
            z-index: 2;
            top: 3px;
            left: 50%;
            transform: translate(-50%, -5px);
            padding: 6px 12px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--card);
            color: var(--text);
            box-shadow: 0 8px 18px var(--shadow);
            font: 700 12px/1.2 "Segoe UI", sans-serif;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: opacity .18s ease, transform .18s ease, visibility .18s;
        }}
        .mascot-bubble.show {{
            opacity: 1;
            visibility: visible;
            transform: translate(-50%, 0);
        }}

        .mascot-unit {{
            position: relative;
            z-index: 1;
            width: 146px;
            height: 146px;
            pointer-events: none;
        }}
        .mascot-button {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            width: 146px;
            height: 146px;
            padding: 0;
            border: 0;
            background: transparent;
            cursor: pointer;
            pointer-events: auto;
            -webkit-tap-highlight-color: transparent;
        }}

        .mascot-button:focus-visible {{
            outline: 3px solid var(--mascot-focus);
            outline-offset: 2px;
            border-radius: 46%;
        }}

        .bean-art {{
            display: block;
            width: 112px;
            height: 124px;
            overflow: visible;
            transform-origin: 50% 86%;
            filter: drop-shadow(0 4px 2px rgba(0, 0, 0, .14));
            animation: bean-idle 3.2s ease-in-out infinite;
        }}

        .fashionista-unit .bean-art {{ animation-delay: .32s; }}
        .bean-shadow {{ fill: var(--text); opacity: .12; }}
        .rookie {{ --bean-fill: var(--rookie-fill); --bean-detail: var(--rookie-detail); --bean-outline: var(--rookie-outline); }}
        .fashionista {{ --bean-fill: var(--fashionista-fill); --bean-detail: var(--fashionista-detail); --bean-outline: var(--fashionista-outline); }}
        .golden-guy {{ --bean-fill: var(--golden-fill); --bean-detail: var(--golden-detail); --bean-outline: var(--golden-outline); }}
        .bean-bot {{ --bean-fill: var(--bot-fill); --bean-detail: var(--bot-detail); --bean-outline: var(--bot-outline); }}
        .clumsy-babe {{ --bean-fill: var(--clumsy-fill); --bean-detail: var(--clumsy-detail); --bean-outline: var(--clumsy-outline); }}
        .bean-body {{
            fill: var(--bean-fill);
            stroke: var(--bean-outline);
            stroke-width: 4;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}
        .bean-leg, .bean-arm {{
            fill: var(--bean-fill);
            stroke: var(--bean-outline);
            stroke-width: 3.5;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}
        .bean-belly {{ stroke: var(--bean-outline); stroke-width: 1.5; }}
        .rookie-belly {{ fill: #FFF1C7; }}
        .fashionista-belly {{ fill: #FFF9FC; }}
        .golden-belly {{ fill: var(--golden-detail); }}
        .clumsy-belly {{ fill: var(--clumsy-detail); }}
        .bean-belly-shine {{ fill: none; stroke: #FFFFFF; stroke-width: 2.5; stroke-linecap: round; opacity: .75; }}
        .bean-eye-white {{ fill: #FFFFFF; stroke: var(--bean-outline); stroke-width: 1.4; }}
        .bean-pupil {{ fill: #17131A; }}
        .bean-glint, .bean-glint-small {{ fill: #FFFFFF; }}
        .bean-eye-blink {{ fill: none; stroke: #17131A; stroke-width: 4; stroke-linecap: round; opacity: 0; }}
        .is-blinking .bean-eye-white, .is-blinking .bean-pupil, .is-blinking .bean-glint, .is-blinking .bean-glint-small {{ opacity: 0; }}
        .is-blinking .bean-eye-blink {{ opacity: 1; }}
        .bean-brow, .bean-mouth {{ fill: none; stroke: #17131A; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
        .rookie-brow {{ stroke-width: 2.5; }}
        .bean-cheek {{ fill: #F7A8D8; opacity: .85; }}
        .rookie-cheek {{ fill: #F29E9F; }}
        .fashionista-mask {{ fill: #FFF9FC; stroke: #17131A; stroke-width: 2; }}
        .bean-bow {{ fill: #FF72BB; stroke: #17131A; stroke-width: 2.5; stroke-linejoin: round; }}
        .fashionista-star {{ fill: #FFF9FC; stroke: #17131A; stroke-width: 2; stroke-linejoin: round; }}
        .rookie-star {{ fill: #FFF1C7; stroke: #30271D; stroke-width: 1.5; stroke-linejoin: round; }}
        .golden-crown {{ fill: #F5C542; stroke: var(--golden-outline); stroke-width: 2.5; stroke-linejoin: round; }}
        .golden-crown .crown-band {{ fill: #D99420; }}
        .golden-guy .bean-sparkle {{ fill: #FFF0B5; stroke: var(--golden-outline); stroke-width: 1.5; animation: gold-sparkle 4.5s ease-in-out infinite; }}
        .golden-sparkle-two {{ animation-delay: 2.25s !important; }}
        .bot-leg, .bot-arm {{ fill: #D9D3DE; stroke: var(--bot-outline); stroke-width: 3.5; stroke-linecap: round; stroke-linejoin: round; }}
        .bot-body {{ fill: var(--bot-fill); stroke: var(--bot-outline); stroke-width: 4; }}
        .bot-belly {{ fill: var(--bot-detail); stroke: var(--bot-outline); stroke-width: 2; }}
        .bot-antenna {{ stroke: var(--bot-outline); stroke-width: 4; stroke-linecap: round; }}
        .bot-antenna-tip {{ fill: var(--bot-glow); stroke: var(--bot-outline); stroke-width: 2.5; animation: bot-led 1.8s ease-in-out infinite; }}
        .bot-screen {{ fill: #302B35; stroke: var(--bot-outline); stroke-width: 3; }}
        .bot-eye {{ fill: var(--bot-glow); animation: bot-led 2.6s ease-in-out infinite; }}
        .bot-eye-glint {{ fill: #FFFFFF; }}
        .bot-smile {{ fill: none; stroke: #FFF9FC; stroke-width: 2.5; stroke-linecap: round; }}
        .bot-led {{ fill: var(--bot-glow); animation: bot-led 1.4s ease-in-out infinite; }}
        .led-two {{ animation-delay: .22s; }}
        .led-three {{ animation-delay: .44s; }}
        .bot-side-light {{ fill: var(--bot-glow); }}
        .bot-sparkle {{ fill: var(--bot-glow); stroke: var(--bot-outline); stroke-width: 1.5; }}
        .clumsy-cheek {{ fill: var(--clumsy-detail); opacity: 1; }}
        .clumsy-pupil {{ transform: translateY(2px); }}
        .clumsy-mouth {{ transform: rotate(-8deg); transform-origin: 75px 105px; }}
        .clumsy-tuft {{ fill: none; stroke: var(--clumsy-detail); stroke-width: 4; stroke-linecap: round; }}
        .fashionista.is-hovering .bean-bow {{ animation: fashion-bow .5s ease-in-out; transform-origin: 75px 36px; }}
        .clumsy-babe .bean-art {{ animation-name: clumsy-idle; }}
        .mascot-button.clumsy-babe.is-hovering .bean-art {{ animation: clumsy-tilt .55s ease-in-out; }}
        .bean-bot .mascot-name {{ color: var(--bot-glow); }}
        .mascot-button.is-sparkling .bean-sparkle {{ animation: bean-sparkle .65s ease-out both; }}
        .bean-arm-wave {{ transform-box: view-box; transform-origin: 73% 53%; }}
        .bean-hand {{ fill: var(--bean-detail); stroke: var(--bean-outline); stroke-width: 3; }}
        .bean-hand-shine {{ fill: none; stroke: #FFFFFF; stroke-width: 2; stroke-linecap: round; }}
        .bean-sparkle {{ opacity: 0; transform-box: fill-box; transform-origin: center; }}
        .rookie .bean-sparkle {{ fill: #F5C542; stroke: #30271D; stroke-width: 1.5; }}
        .fashionista .bean-sparkle {{ fill: #FF72BB; stroke: #17131A; stroke-width: 1.5; }}
        .is-sparkling .bean-sparkle {{ animation: bean-sparkle .65s ease-out both; }}
        .sparkle-b {{ animation-delay: .12s !important; }}
        .mascot-name {{
            display: block;
            padding-top: 1px;
            color: var(--muted);
            font: 800 11px/1.2 "Segoe UI", sans-serif;
            letter-spacing: .08em;
            text-transform: uppercase;
        }}
        .mascot-button:hover .bean-arm-wave, .mascot-button.is-hovering .bean-arm-wave {{ animation: bean-wave .62s ease-in-out infinite; }}
        .mascot-button.is-hovering .bean-art {{ animation: bean-tilt .38s ease-in-out; }}
        .mascot-button.is-jumping .bean-art {{ animation: bean-jump .72s cubic-bezier(.2,.75,.25,1); }}

        @keyframes bean-idle {{
            0%, 100% {{ transform: translateY(0) scale(1); }}
            50% {{ transform: translateY(-3px) scale(1, 1.015); }}
        }}
        @keyframes bean-wave {{
            0%, 100% {{ transform: rotate(0); }}
            35% {{ transform: rotate(-38deg); }}
            68% {{ transform: rotate(-8deg); }}
        }}
        @keyframes bean-tilt {{
            0%, 100% {{ transform: rotate(0); }}
            35% {{ transform: rotate(-4deg); }}
            70% {{ transform: rotate(3deg); }}
        }}
        @keyframes bean-jump {{
            0% {{ transform: translateY(0) scale(1, 1); }}
            16% {{ transform: translateY(5px) scale(1.12, .82); }}
            42% {{ transform: translateY(-20px) scale(.94, 1.08); }}
            68% {{ transform: translateY(-11px) scale(1.02, .98); }}
            86% {{ transform: translateY(-2px) scale(1.08, .9); }}
            100% {{ transform: translateY(0) scale(1, 1); }}
        }}
        @keyframes bean-sparkle {{
            0% {{ opacity: 0; transform: scale(.3) rotate(-20deg); }}
            30% {{ opacity: 1; transform: scale(1.15) rotate(10deg); }}
            100% {{ opacity: 0; transform: scale(.7) rotate(30deg); }}
        }}
        @keyframes gold-sparkle {{
            0%, 72%, 100% {{ opacity: 0; transform: scale(.5) rotate(-12deg); }}
            78%, 90% {{ opacity: .95; transform: scale(1) rotate(8deg); }}
        }}
        @keyframes bot-led {{
            0%, 100% {{ opacity: .55; }}
            50% {{ opacity: 1; }}
        }}
        @keyframes clumsy-idle {{
            0%, 100% {{ transform: rotate(-2deg) translateY(0); }}
            50% {{ transform: rotate(-1deg) translateY(-2px); }}
        }}
        @keyframes clumsy-tilt {{
            0%, 100% {{ transform: rotate(-2deg); }}
            45% {{ transform: rotate(4deg) translateY(-3px); }}
        }}
        @keyframes fashion-bow {{
            0%, 100% {{ transform: rotate(0deg) scale(1); }}
            50% {{ transform: rotate(12deg) scale(1.08); }}
        }}

        @media (max-width: 640px) {{
            .mascot-stage {{
                grid-template-columns: repeat(6, minmax(0, 1fr));
                grid-auto-rows: 100px;
                height: 212px;
                row-gap: 4px;
            }}
            .mascot-unit, .mascot-button {{ width: 100%; height: 100px; }}
            .rookie-unit, .fashionista-unit, .golden-guy-unit {{ grid-column: span 2; }}
            .bean-bot-unit {{ grid-column: 2 / span 2; }}
            .clumsy-babe-unit {{ grid-column: 4 / span 2; }}
            .bean-art {{ width: min(88px, 100%); height: 90px; }}
            .mascot-name {{ font-size: 8px; letter-spacing: .02em; }}
            .mascot-bubble {{ top: 2px; font-size: 10px; }}
        }}

        @media (min-width: 641px) and (max-width: 900px) {{
            .mascot-stage {{ grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0; }}
            .mascot-unit, .mascot-button {{ width: 100%; }}
            .bean-art {{ width: min(108px, 100%); height: 120px; }}
            .mascot-name {{ font-size: 9px; letter-spacing: .03em; }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            .bean-art, .bean-arm-wave, .bean-sparkle, .mascot-bubble, .bean-bow, .golden-crown, .bot-eye, .bot-led, .bot-antenna-tip {{
                animation: none !important;
                transition: none !important;
            }}
        }}
        </style>
        {mascot_html}
        """,
        height=220,
        scrolling=False,
    )


st.set_page_config(page_title="LazyNotes AI", page_icon="📝", layout="wide")

theme = get_theme()

selected_theme = DARK_THEME if theme == "dark" else LIGHT_THEME
brand_accent = "#F5C542" if theme == "dark" else selected_theme["text"]

st.markdown(
    f"""
    <style>
    :root {{
        --bg: {selected_theme['bg']};
        --bg-alt: {selected_theme['bg_alt']};
        --card: {selected_theme['card']};
        --card-alt: {selected_theme['card_alt']};
        --border: {selected_theme['border']};
        --primary: {selected_theme['primary']};
        --primary-soft: {selected_theme['primary_soft']};
        --text: {selected_theme['text']};
        --muted: {selected_theme['muted']};
        --shadow: {selected_theme['shadow']};
        --input-bg: {selected_theme['input_bg']};
        --brand-accent: {brand_accent};
        --workflow-accent: #F5C542;
        --workflow-glow: rgba(245, 197, 66, 0.18);
    }}

    .stApp {{
        background: var(--bg);
        color: var(--text);
        transition: background 0.35s ease, color 0.25s ease;
    }}

    .stAppHeader {{
        background: transparent;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
        box-shadow: none;
    }}

    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    .app-shell {{
        padding: 0.4rem 0 2rem;
        animation: fadeInUp 0.6s ease;
    }}

    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding: 0.5rem 0.25rem 0.8rem;
        border-bottom: 1px solid var(--border);
    }}

    .brand {{
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0;
        color: var(--brand-accent);
        text-shadow: 0 0 20px rgba(245, 197, 66, 0.12);
    }}

    .brand-mark {{
        display: inline-flex;
        width: 2.2rem;
        height: 2.2rem;
        align-items: center;
        justify-content: center;
        border-radius: 0.8rem;
        background: var(--primary-soft);
        color: var(--brand-accent);
        box-shadow: inset 0 0 0 1px var(--border);
    }}

    [data-testid="stPopoverButton"] {{
        width: 2.65rem !important;
        height: 2.65rem !important;
        min-height: 2.65rem !important;
        padding: 0 !important;
        border-radius: 0.8rem !important;
        border: 1px solid var(--border) !important;
        background: var(--card) !important;
        color: var(--text) !important;
        font-size: 1.15rem !important;
        transition: transform 0.2s ease, color 0.2s ease, border-color 0.2s ease;
    }}

    [data-testid="stPopoverButton"] p {{
        margin: 0 !important;
        color: inherit !important;
        font-size: 1.15rem !important;
    }}

    [data-testid="stPopoverButton"]:hover {{
        transform: rotate(18deg);
        color: var(--primary) !important;
        border-color: var(--primary) !important;
    }}

    [data-testid="stPopoverButton"]:focus-visible {{
        outline: 3px solid var(--primary) !important;
        outline-offset: 3px;
    }}

    [data-testid="stHorizontalBlock"]:has([data-testid="stPopoverButton"]) {{
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }}

    [data-testid="stHorizontalBlock"]:has([data-testid="stPopoverButton"]) > [data-testid="column"]:first-child {{
        flex: 1 1 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }}

    [data-testid="stHorizontalBlock"]:has([data-testid="stPopoverButton"]) > [data-testid="column"]:last-child {{
        flex: 0 0 2.75rem !important;
        width: 2.75rem !important;
        min-width: 2.75rem !important;
    }}

    .theme-toggle {{
        justify-content: center;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.08);
        color: var(--text);
        border-radius: 999px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .theme-toggle:hover {{
        transform: translateY(-1px);
        border-color: var(--primary);
        box-shadow: 0 8px 20px var(--shadow);
    }}

    .hero {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 1.2rem;
        padding: 1.3rem 0 2rem;
        align-items: start;
    }}

    .hero-card, .panel, .result-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 1.2rem;
        box-shadow: 0 14px 28px var(--shadow);
    }}

    .hero-card {{
        padding: 1.7rem;
    }}

    .eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.42rem 0.7rem;
        border-radius: 999px;
        background: var(--primary-soft);
        border: 1px solid var(--border);
        color: var(--primary);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }}

    h1 {{
        font-size: clamp(2.1rem, 4vw, 3.4rem);
        line-height: 1.05;
        margin: 0 0 0.8rem;
        letter-spacing: -0.05em;
        color: var(--text);
    }}

    .lead {{
        font-size: 1rem;
        line-height: 1.6;
        color: var(--muted);
        max-width: 46rem;
        margin: 0;
    }}

    .hero-stats {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1.6rem;
    }}

    .feature-card {{
        display: flex;
        min-width: 0;
        min-height: 78px;
        align-items: center;
        gap: 0.75rem;
        padding: 0.85rem 0.9rem;
        border: 1px solid rgba(247, 168, 216, 0.48);
        border-radius: 0.8rem;
        background: var(--card-alt);
        color: var(--text);
        box-shadow: 0 5px 14px var(--shadow);
    }}

    .feature-icon {{
        display: grid;
        width: 2.35rem;
        height: 2.35rem;
        flex: 0 0 auto;
        place-items: center;
        border-radius: 0.7rem;
        background: rgba(247, 168, 216, 0.18);
        color: var(--mascot-accent);
        font-size: 1.15rem;
        font-weight: 800;
    }}

    .feature-label {{
        display: block;
        margin-bottom: 0.2rem;
        color: var(--muted);
        font-size: 0.64rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }}

    .feature-description {{
        display: block;
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 700;
        line-height: 1.35;
    }}

    .panel {{
        padding: 1.2rem;
        min-height: 100%;
    }}

    .workflow-section {{
        padding: 1.1rem 0 1.6rem;
    }}

    .workflow-heading {{
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 1rem;
    }}

    .workflow-heading h2 {{
        margin: 0;
        color: var(--text);
        font-size: 1.35rem;
    }}

    .workflow-count {{
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 700;
        white-space: nowrap;
    }}

    .workflow-track {{
        height: 5px;
        margin: 0.8rem 0 1rem;
        overflow: hidden;
        border-radius: 999px;
        background: var(--border);
    }}

    .workflow-track span {{
        display: block;
        height: 100%;
        border-radius: inherit;
        background: var(--workflow-accent);
        box-shadow: 0 0 12px var(--workflow-glow);
        transition: width 0.35s ease;
    }}

    .workflow-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
    }}

    .workflow-card {{
        display: flex;
        min-width: 0;
        min-height: 150px;
        flex-direction: column;
        align-items: flex-start;
        padding: 0.95rem 1rem;
        border: 1px solid var(--border);
        border-radius: 0.65rem;
        background: var(--card);
        color: var(--text) !important;
        text-decoration: none !important;
        box-shadow: 0 8px 18px var(--shadow);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .workflow-card:hover {{
        transform: translateY(-3px);
        border-color: var(--workflow-accent);
        box-shadow: 0 12px 24px var(--shadow);
    }}

    .workflow-card:visited {{
        color: var(--text) !important;
    }}

    .workflow-card:focus-visible {{
        outline: 3px solid var(--workflow-accent);
        outline-offset: 3px;
    }}

    .workflow-card.is-active {{
        border-color: var(--workflow-accent);
        box-shadow: 0 0 0 1px var(--workflow-accent), 0 0 22px var(--workflow-glow);
    }}

    .workflow-card.is-complete .workflow-index {{
        background: var(--workflow-accent);
        color: #19151D;
    }}

    .workflow-card-top {{
        display: flex;
        width: 100%;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.65rem;
    }}

    .workflow-index {{
        display: inline-grid;
        width: 1.8rem;
        height: 1.8rem;
        place-items: center;
        border: 1px solid var(--border);
        border-radius: 50%;
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 800;
    }}

    .workflow-icon {{
        color: var(--workflow-accent);
        font-size: 1.35rem;
        line-height: 1;
        transform-origin: center;
        transition: transform 0.2s ease;
    }}

    .workflow-card:hover .workflow-icon {{
        transform: translateY(-2px) rotate(-5deg);
    }}

    .workflow-status {{
        margin-bottom: 0.22rem;
        color: var(--workflow-accent);
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    .workflow-title {{
        display: block;
        margin-bottom: 0.3rem;
        color: var(--text);
        font-size: 0.95rem;
        font-weight: 800;
        line-height: 1.25;
    }}

    .workflow-description {{
        display: block;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.45;
    }}

    .panel h3 {{
        margin: 0 0 0.75rem;
        font-size: 1.15rem;
        color: var(--text);
    }}

    .panel-list {{
        list-style: none;
        margin: 0;
        padding: 0;
        display: grid;
        gap: 0.6rem;
        color: var(--muted);
    }}

    .panel-list li {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: var(--card-alt);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 0.7rem 0.8rem;
    }}

    .step-dot {{
        width: 0.7rem;
        height: 0.7rem;
        border-radius: 50%;
        background: var(--primary);
        box-shadow: 0 0 0 5px var(--primary-soft);
    }}

    .notes-panel {{
        margin-top: 1rem;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 1.3rem;
        box-shadow: 0 12px 24px var(--shadow);
        padding: 1.2rem;
        animation: fadeInUp 0.7s ease;
    }}

    .section-heading {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.8rem;
    }}

    .section-heading h2 {{
        margin: 0;
        font-size: 1.15rem;
        color: var(--text);
    }}

    .pill {{
        display: inline-flex;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: var(--primary-soft);
        border: 1px solid var(--border);
        color: var(--primary);
        font-weight: 600;
        font-size: 0.75rem;
    }}

    .stTextArea textarea, .stTextInput input {{
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 1rem !important;
        padding: 0.9rem 1rem !important;
        box-shadow: none !important;
    }}

    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }}

    .stButton > button {{
        border-radius: 0.9rem !important;
        border: 1px solid var(--border) !important;
        background: var(--card-alt) !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        padding: 0.7rem 1rem !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        border-color: var(--primary) !important;
        box-shadow: 0 8px 20px var(--shadow);
    }}

    .stButton > button:focus {{
        box-shadow: 0 0 0 2px var(--primary-soft) !important;
    }}

    .action-row {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }}

    .result-card {{
        margin-top: 1rem;
        padding: 1.2rem 1.1rem;
        animation: fadeInUp 0.45s ease;
    }}

    .result-card h3 {{
        margin: 0 0 0.8rem;
        color: var(--text);
        font-size: 1.1rem;
    }}

    .result-card p, .result-card li, .result-card ul, .result-card ol {{
        color: var(--text);
        line-height: 1.7;
    }}

    .result-card strong {{
        color: var(--primary);
    }}

    .example-card {{
        background: var(--primary-soft);
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 1rem;
        padding: 1rem;
        margin-top: 1rem;
    }}

    .stAlert {{
        background: var(--card-alt) !important;
        border: 1px solid var(--border) !important;
        border-radius: 1rem !important;
        color: var(--text) !important;
    }}

    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(12px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @media (max-width: 900px) {{
        .hero {{
            grid-template-columns: 1fr;
        }}
        .workflow-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
        .action-row {{
            grid-template-columns: 1fr;
        }}
    }}

    @media (max-width: 640px) {{
        .brand {{ font-size: 1.4rem; gap: 0.55rem; }}
        .brand-mark {{ width: 1.9rem; height: 1.9rem; }}
        .hero-card {{ padding: 1.2rem; }}
        .hero-stats {{ grid-template-columns: 1fr; gap: 0.55rem; margin-top: 1.25rem; }}
        .feature-card {{ min-height: 66px; padding: 0.7rem 0.8rem; }}
        .feature-description {{ font-size: 0.84rem; }}
        .workflow-heading h2 {{ font-size: 1.15rem; }}
        .workflow-card {{ min-height: 154px; padding: 0.8rem; }}
        .workflow-title {{ font-size: 0.86rem; }}
        .workflow-description {{ font-size: 0.73rem; }}
    }}

    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation: none !important;
            transition: none !important;
            scroll-behavior: auto !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

ollama_connected, available_models = get_ollama_status()
model_name = st.session_state.get("model_name", DEFAULT_MODEL)

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

col_brand, col_settings = st.columns([6, 0.65])
with col_brand:
    st.markdown(
        '<div class="topbar"><div class="brand"><span class="brand-mark">📝</span><span>LazyNotes AI</span></div></div>',
        unsafe_allow_html=True,
    )
with col_settings:
    with st.popover("⚙", help="Open settings", use_container_width=False):
        st.markdown("**Appearance**")
        selected_theme = st.radio(
            "Theme mode",
            options=("light", "dark"),
            format_func=lambda mode: "☀️ Light Mode" if mode == "light" else "🌙 Dark Mode",
            index=0 if theme == "light" else 1,
            key="appearance_theme",
            on_change=persist_theme_choice,
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**AI Setup**")
        if ollama_connected and available_models:
            st.success(f"Connected · {len(available_models)} model(s) available")
            current_index = available_models.index(model_name) if model_name in available_models else 0
            model_name = st.selectbox(
                "Installed model",
                options=available_models,
                index=current_index,
                key="ai_model_picker",
                on_change=reset_workflow_outputs,
            )
            st.session_state["model_name"] = model_name
        else:
            model_name = st.text_input(
                "Model name",
                value=model_name,
                key="model_name",
                on_change=reset_workflow_outputs,
                help="Enter a model installed in Ollama.",
            )
            if ollama_connected:
                st.warning("Ollama is connected, but no models are installed yet.")
            else:
                st.error("Ollama is unavailable. Start Ollama to generate study materials.")

render_mascot(theme)

privacy_description = (
    "Your notes stay on your device"
    if ollama_is_local()
    else "Your notes go to your configured AI server"
)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-card">
            <div class="eyebrow">AI study assistant</div>
            <h1>Turn your notes into study-ready insight.</h1>
            <p class="lead">Turn class notes into clear summaries, key ideas, and revision questions with your friendly study buddy.</p>
            <div class="hero-stats" aria-label="Study app benefits">
                <div class="feature-card">
                    <span class="feature-icon" aria-hidden="true">✦</span>
                    <span>
                        <span class="feature-label">Your study buddy</span>
                        <span class="feature-description">AI-powered revision</span>
                    </span>
                </div>
                <div class="feature-card">
                    <span class="feature-icon" aria-hidden="true">🔒</span>
                    <span>
                        <span class="feature-label">Your privacy</span>
                        <span class="feature-description">{privacy_description}</span>
                    </span>
                </div>
                <div class="feature-card">
                    <span class="feature-icon" aria-hidden="true">♡</span>
                    <span>
                        <span class="feature-label">Less study stress</span>
                        <span class="feature-description">Summaries in a few clicks</span>
                    </span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

notes_ready = bool(st.session_state.get("study_notes", "").strip())
model_ready = bool(model_name.strip() and model_name.strip() in available_models)
summary_ready = bool(st.session_state.get("study_summary"))
review_outputs_ready = bool(
    st.session_state.get("study_key_points")
    and st.session_state.get("study_revision_questions")
)
review_ready = review_outputs_ready and st.session_state.get("workflow_reviewed", False)
step_completion = [notes_ready, model_ready, summary_ready, review_ready]
progress_count = sum(step_completion)

workflow_steps = [
    ("Paste your notes", "Add your class notes to the workspace.", "✎", "#notes-workspace"),
    ("Choose your AI model", "Pick an installed model from Settings.", "◉", "#model-selection"),
    ("Generate your summary", "Create a summary from your notes.", "✦", "#study-actions"),
    ("Review ideas and questions", "Generate both review tools, then confirm you reviewed them.", "▤", "#study-results"),
]
workflow_cards = []
active_step = next((index for index, done in enumerate(step_completion) if not done), 3)
for index, (title, description, icon, target) in enumerate(workflow_steps):
    done = step_completion[index]
    status = "DONE" if done else "CURRENT" if index == active_step else "UP NEXT"
    classes = "workflow-card is-complete" if done else "workflow-card"
    if not done and index == active_step:
        classes += " is-active"
    current_attr = ' aria-current="step"' if not done and index == active_step else ""
    workflow_cards.append(
        f'<a class="{classes}" href="{target}"{current_attr}>'
        f'<span class="workflow-card-top"><span class="workflow-index">{index + 1:02}</span>'
        f'<span class="workflow-icon" aria-hidden="true">{icon}</span></span>'
        f'<span class="workflow-status">{status}</span>'
        f'<span class="workflow-title">{title}</span>'
        f'<span class="workflow-description">{description}</span></a>'
    )

st.markdown(
    f"""
    <section class="workflow-section" aria-labelledby="workflow-title">
      <div class="workflow-heading">
        <h2 id="workflow-title">How it works</h2>
        <span class="workflow-count">{progress_count} / 4 steps complete</span>
      </div>
      <div class="workflow-track" role="progressbar" aria-label="Study workflow progress" aria-valuemin="0" aria-valuemax="4" aria-valuenow="{progress_count}">
        <span style="width: {progress_count * 25}%"></span>
      </div>
      <div class="workflow-grid">{''.join(workflow_cards)}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notes-panel" id="notes-workspace">
        <div class="section-heading">
            <h2>Notes workspace</h2>
            <span class="pill">Study tools</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

notes = st.text_area(
    "Paste your notes here",
    height=260,
    placeholder="Type or paste your class notes here...",
    key="study_notes",
    on_change=reset_workflow_outputs,
)

if not notes.strip():
    st.warning("Add your notes first, then choose an action.")

st.markdown('<div id="study-actions"></div>', unsafe_allow_html=True)
st.markdown('<div class="action-row">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Summarize", disabled=not notes.strip()):
        try:
            with st.spinner("Generating summary..."):
                result = summarize_notes(notes, model=model_name)
            st.session_state["study_summary"] = result
            st.session_state["workflow_reviewed"] = False
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

with col2:
    if st.button("Key points", disabled=not notes.strip()):
        try:
            with st.spinner("Finding the important ideas..."):
                result = key_points(notes, model=model_name)
            st.session_state["study_key_points"] = result
            st.session_state["workflow_reviewed"] = False
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

with col3:
    if st.button("Revision questions", disabled=not notes.strip()):
        try:
            with st.spinner("Creating revision questions..."):
                result = revision_questions(notes, model=model_name)
            st.session_state["study_revision_questions"] = result
            st.session_state["workflow_reviewed"] = False
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div id="study-results"></div>', unsafe_allow_html=True)

for result_key, result_title in (
    ("study_summary", "Summary"),
    ("study_key_points", "Key points"),
    ("study_revision_questions", "Revision questions"),
):
    saved_result = st.session_state.get(result_key)
    if saved_result:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### {result_title}")
        st.markdown(saved_result)
        st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get("study_key_points") and st.session_state.get("study_revision_questions"):
    st.checkbox("I reviewed the key ideas and revision questions", key="workflow_reviewed")

st.markdown(
    """
    <div class="example-card">
        <strong>Concept explain</strong><br>
        Use the input below to understand one idea from your notes in plain language.
    </div>
    """,
    unsafe_allow_html=True,
)

concept = st.text_input("Explain a concept from your notes", placeholder="Example: photosynthesis")

if st.button("Explain concept") and notes.strip() and concept.strip():
    try:
        with st.spinner("Explaining the concept..."):
            result = explain_concept(notes, concept, model=model_name)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### Understanding: {concept}")
        st.markdown(result)
        st.markdown('</div>', unsafe_allow_html=True)
    except ValueError as exc:
        st.error(str(exc))

st.caption("Need help generating study materials? Open Settings to check your AI setup.")

st.markdown('</div>', unsafe_allow_html=True)

