"""Loading of the heavy ML models used by Zyra.

Loading is expensive (downloads + GPU placement), so it lives here, separate from the
app logic, and produces a single :class:`ZyraModels` bundle.
"""

from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from kokoro import KPipeline
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Processing model: multimodal vision-language model (llama.cpp / GGUF via mmproj).
# 4 bit quantized gemma 4 E4B
# PROCESSING_MODEL_REPO = "google/gemma-4-E4B-it-qat-q4_0-gguf"
# PROCESSING_MODEL_FILE = "gemma-4-E4B_q4_0-it.gguf"
# PROCESSING_MMPROJ_FILE = "gemma-4-E4B-it-mmproj.gguf"
# 4 bit quantized gemma 4 12B
PROCESSING_MODEL_REPO = "google/gemma-4-12B-it-qat-q4_0-gguf"
PROCESSING_MODEL_FILE = "gemma-4-12b-it-qat-q4_0.gguf"
PROCESSING_MMPROJ_FILE = "mmproj-gemma-4-12b-it-qat-q4_0.gguf"
# whisper-large-v3-turbo
ASR_MODEL_REPO = "openai/whisper-large-v3-turbo"
# Kokoro-82M
TTS_MODEL_REPO = "hexgrad/Kokoro-82M"
TTS_LANG_CODE = "a"
# 4096
PROCESSING_N_CTX = 4096


@dataclass
class ZyraModels:
    """Bundle of the loaded models the app relies on."""

    processing: Llama
    asr: Any  # transformers automatic-speech-recognition pipeline
    tts: KPipeline


def load_models() -> ZyraModels:
    """Download and initialise every model. Call once at startup."""
    mmproj_path = hf_hub_download(
        repo_id=PROCESSING_MODEL_REPO, filename=PROCESSING_MMPROJ_FILE
    )
    processing = Llama.from_pretrained(
        repo_id=PROCESSING_MODEL_REPO,
        filename=PROCESSING_MODEL_FILE,
        chat_handler=Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False),
        n_ctx=PROCESSING_N_CTX,
        verbose=False,
    )

    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_REPO, dtype=DTYPE, low_cpu_mem_usage=True, use_safetensors=True
    )
    asr_model.to(DEVICE)
    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_REPO)
    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=asr_processor.tokenizer,
        feature_extractor=asr_processor.feature_extractor,
        dtype=DTYPE,
        device=DEVICE,
    )

    tts = KPipeline(lang_code=TTS_LANG_CODE, repo_id=TTS_MODEL_REPO, device=DEVICE)

    return ZyraModels(processing=processing, asr=asr, tts=tts)
