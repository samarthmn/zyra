# Zyra

Zyra is a small Gradio app for plant ID and plant-care questions. Upload a photo, type a question, or record your voice. It uses a local GGUF vision-language model, Whisper for speech input, Kokoro for speech output, and Trefle when a botanical match is available.

## Demo

Demo video: _coming soon_

## Tech

- Python 3.14
- Gradio for the app UI
- llama.cpp for local GGUF inference
- Quantized Gemma 4 for plant ID, image understanding, and care answers
- Whisper Large v3 Turbo for speech-to-text
- Kokoro 82M for text-to-speech
- Trefle API for botanical lookup and tool calling

The plant reasoning model is local. Use quantized `gemma-4-12B` when you want better answers and can wait a bit longer. Switch to quantized `gemma-4-E4B` when you want faster responses and can accept lower accuracy.

## Run

```sh
cp env.example .env
uv run python main.py
```

After it starts, open `http://127.0.0.1:7860/`.

Set `TREFLE_TOKEN` in `.env` if you want verified plant data. Without it, Zyra still answers from the model, but skips the Trefle lookup.

## Notebook

Open `main.ipynb` if you prefer running the app cell by cell.
