"""Zyra plant-care assistant: inference, formatting, and the Gradio app.

The running application lives here as plain functions. Pieces worth keeping
separate live in sibling modules:
    zyra.models       — heavy model loading (ZyraModels / load_models)
    zyra.prompts      — system prompts
    zyra.tools.trefle — Trefle botanical API (tool calling)
    zyra.utils        — image helpers
"""

import json
import re
from functools import partial
from typing import Any

import gradio as gr
import numpy as np
from dotenv import load_dotenv

from zyra.models import ZyraModels
from zyra.prompts import DEFAULT_USER_PROMPT, IDENTIFY_PROMPT, REFINE_PROMPT
from zyra.tools.trefle import lookup_plant, summarize_for_llm
from zyra.utils import load_image_from_url, numpy_to_pil, pil_to_data_uri

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MAX_NEW_TOKENS = 512
GENERATION_TEMPERATURE = 0.2

TTS_VOICE = "af_heart"
TTS_SAMPLE_RATE = 24000

# Confidence floor applied once a result is verified against Trefle.
VERIFIED_CONFIDENCE = 90

# Baseline record: parse fallback and the "nothing identified yet" state.
EMPTY_PLANT_DATA: dict[str, Any] = {
    "common_name": None,
    "scientific_name": None,
    "confidence_percent": 0,
    "response": "",
    "needs_more_info": False,
    "more_info_request": None,
}


# --------------------------------------------------------------------------- #
# Audio: speech-to-text and text-to-speech
# --------------------------------------------------------------------------- #


def transcribe_audio(asr_pipeline, audio) -> str:
    """Transcribe a Gradio ``(sample_rate, samples)`` tuple to text."""
    if audio is None:
        return ""
    sample_rate, audio_array = audio
    result = asr_pipeline({"array": audio_array, "sampling_rate": sample_rate})
    return result["text"].strip()


def synthesize_speech(tts_pipeline, text: str) -> tuple[int, np.ndarray] | None:
    """Render ``text`` to a ``(sample_rate, samples)`` tuple, or None if empty."""
    if not text or not text.strip():
        return None

    chunks = [
        result.audio.cpu().numpy()
        for result in tts_pipeline(text, voice=TTS_VOICE)
        if result.audio is not None
    ]
    if not chunks:
        return None

    return TTS_SAMPLE_RATE, np.concatenate(chunks)


# --------------------------------------------------------------------------- #
# Inference: chat content, generation, and parsing
# --------------------------------------------------------------------------- #


def build_user_content(image: np.ndarray | None, user_text: str):
    """Build the multimodal user message content for a chat completion."""
    text = user_text or DEFAULT_USER_PROMPT
    if image is None:
        return text

    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": pil_to_data_uri(numpy_to_pil(image))},
        },
    ]


def generate(model, messages: list[dict]) -> str:
    """Run a chat completion against the processing model and return its text."""
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=GENERATION_TEMPERATURE,
    )
    return (response["choices"][0]["message"]["content"] or "").strip()


def parse_plant_response(raw: str) -> dict[str, Any]:
    """Parse the model's JSON reply, tolerating markdown fences and surrounding prose."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {**EMPTY_PLANT_DATA, "response": raw}


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def format_plant_identification(plant_data: dict[str, Any], trefle=None) -> str:
    """Render the identification panel, preferring verified Trefle fields."""
    if trefle:
        common = trefle.common_name or plant_data.get("common_name") or "Unknown"
        scientific = trefle.scientific_name or "Unknown"
        confidence = max(int(plant_data.get("confidence_percent") or 0), VERIFIED_CONFIDENCE)
        source = "Verified via Trefle"
    else:
        common = plant_data.get("common_name") or "Unknown"
        scientific = plant_data.get("scientific_name") or "Unknown"
        confidence = int(plant_data.get("confidence_percent") or 0)
        source = "AI estimate"

    return (
        f"Common name: {common}\n"
        f"Scientific name: {scientific}\n"
        f"Confidence: {confidence}%\n"
        f"Source: {source}"
    )


def format_response_text(data: dict[str, Any]) -> str:
    """Combine the response with any follow-up request for more information."""
    response = (data.get("response") or "").strip()
    if data.get("needs_more_info") and data.get("more_info_request"):
        response = f"{response}\n\n{data['more_info_request'].strip()}"
    return response


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

_UNKNOWN_INFO = format_plant_identification({})

# Gradio output shape: (audio, response_text, plant_info, reference_image, history).
Response = tuple[
    tuple[int, np.ndarray] | None, str, str, np.ndarray | None, list[dict]
]


def _identify_plant(
    models: ZyraModels, image, user_text: str, history: list[dict]
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": IDENTIFY_PROMPT},
        *history,
        {"role": "user", "content": build_user_content(image, user_text)},
    ]
    return parse_plant_response(generate(models.processing, messages))


def _lookup_trefle(plant_data: dict[str, Any]):
    query = plant_data.get("scientific_name") or plant_data.get("common_name")
    if not query:
        return None
    try:
        return lookup_plant(query)
    except Exception:
        return None


def _refine_response(
    models: ZyraModels, plant_data: dict[str, Any], trefle_result, user_text: str
) -> dict[str, Any]:
    """Refine the response using Trefle data; fall back to the original on failure."""
    messages = [
        {"role": "system", "content": REFINE_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_question": user_text,
                    "initial_assessment": plant_data,
                    "trefle_data": summarize_for_llm(trefle_result),
                }
            ),
        },
    ]

    try:
        refined = parse_plant_response(generate(models.processing, messages))
    except Exception:
        return plant_data

    refined.setdefault("common_name", plant_data.get("common_name"))
    refined.setdefault("scientific_name", plant_data.get("scientific_name"))
    refined.setdefault(
        "confidence_percent",
        max(int(plant_data.get("confidence_percent") or 0), VERIFIED_CONFIDENCE),
    )
    return refined


def respond(models: ZyraModels, image, text, audio, history: list[dict] | None) -> Response:
    """Handle one turn of conversation and produce all UI outputs."""
    history = history or []

    user_text = (text or "").strip()
    spoken_text = transcribe_audio(models.asr, audio)
    if spoken_text:
        user_text = f"{user_text} {spoken_text}".strip()

    if image is None and not user_text:
        message = (
            "I couldn't understand the audio. Please try again or type your question."
            if audio is not None
            else "Please provide a plant photo, a question, or a voice message."
        )
        return None, message, _UNKNOWN_INFO, None, history

    try:
        plant_data = _identify_plant(models, image, user_text, history)
    except Exception as exc:
        error = f"Sorry, I couldn't reach the plant care model: {exc}"
        return None, error, _UNKNOWN_INFO, None, history

    trefle_result = _lookup_trefle(plant_data)
    reference_image = None
    if trefle_result:
        plant_data = _refine_response(models, plant_data, trefle_result, user_text)
        if trefle_result.image_url:
            reference_image = load_image_from_url(trefle_result.image_url)

    response_text = format_response_text(plant_data)
    plant_info = format_plant_identification(plant_data, trefle_result)
    audio_output = synthesize_speech(models.tts, response_text)

    new_history = [
        *history,
        {"role": "user", "content": user_text or DEFAULT_USER_PROMPT},
        {"role": "assistant", "content": response_text},
    ]

    return audio_output, response_text, plant_info, reference_image, new_history


# --------------------------------------------------------------------------- #
# Gradio app
# --------------------------------------------------------------------------- #


def build_demo(models: ZyraModels) -> gr.Blocks:
    """Build the Gradio app wired to ``respond`` with ``models`` bound in."""
    handle_request = partial(respond, models)

    with gr.Blocks(title="Zyra — Plant Care Assistant") as demo:
        gr.Markdown(
            "# Zyra\n"
            "Your multi-modal plant care assistant. Upload a photo, type a question, "
            "or record a voice message."
        )

        chat_history = gr.State([])

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Plant Photo", type="numpy")
                text_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. Why are the leaves turning yellow?",
                )
                audio_input = gr.Audio(label="Voice Input", type="numpy")
                submit_btn = gr.Button("Ask Zyra", variant="primary")
            with gr.Column():
                plant_info_output = gr.Textbox(label="Plant Identification", lines=4)
                text_output = gr.Textbox(label="Zyra's Response", lines=4)
                reference_image_output = gr.Image(label="Reference Image (Trefle)")
                audio_output = gr.Audio(label="Listen", autoplay=True)

        submit_btn.click(
            fn=handle_request,
            inputs=[image_input, text_input, audio_input, chat_history],
            outputs=[
                audio_output,
                text_output,
                plant_info_output,
                reference_image_output,
                chat_history,
            ],
        )

    return demo
