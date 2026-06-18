# Zyra

Zyra is a small Gradio app for plant ID and plant-care questions. Upload a photo, type a question, or record your voice. It uses a local GGUF vision-language model, Whisper for speech input, Kokoro for speech output, and Trefle when a botanical match is available.

## Run

```sh
cp env.example .env
uv run python main.py
```

Set `TREFLE_TOKEN` in `.env` if you want verified plant data. Without it, Zyra still answers from the model, but skips the Trefle lookup.

## Notebook

Open `main.ipynb` if you prefer running the app cell by cell.
