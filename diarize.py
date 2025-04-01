import argparse
import logging
import os
import re
import json
import torch
import torchaudio
from faster_whisper import WhisperModel, BatchedInferencePipeline, decode_audio
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--audio", help="name of the target audio file", required=True)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation.",
)
parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses numerical digits.",
)
parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)
parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference (0 for longform inference)",
)
parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio (None for detection)",
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use ('cuda' or 'cpu')",
)
args = parser.parse_args()
language = process_language_arg(args.language, args.model_name)

# Ensure output directory exists
output_dir = "/app/output"
os.makedirs(output_dir, exist_ok=True)

# Process audio
if args.stemming:
    logger.info("Performing source separation...")
    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o temp_outputs --device "{args.device}"'
    )
    if return_code != 0:
        logger.warning("Source splitting failed, using original audio.")
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio

# Transcribe audio
logger.info(f"Loading Whisper model: {args.model_name}")
whisper_model = WhisperModel(args.model_name, device=args.device, compute_type=mtypes[args.device])
whisper_pipeline = BatchedInferencePipeline(whisper_model)
audio_waveform = decode_audio(vocal_target)
suppress_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer) if args.suppress_numerals else [-1]

logger.info("Transcribing audio...")
if args.batch_size > 0:
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform, language, suppress_tokens=suppress_tokens, batch_size=args.batch_size
    )
else:
    transcript_segments, info = whisper_model.transcribe(
        audio_waveform, language, suppress_tokens=suppress_tokens, vad_filter=True
    )
full_transcript = "".join(segment.text for segment in transcript_segments)
logger.info(f"Detected language: {info.language}")

# Clear GPU memory
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()

# Load alignment model
try:
    alignment_model, alignment_tokenizer = load_alignment_model(
        args.device, dtype=torch.float16 if args.device == "cuda" else torch.float32
    )
    logger.info("Alignment model loaded.")
except Exception as e:
    logger.error(f"Failed to load alignment model: {e}")
    cleanup("temp_outputs")
    raise

# Align transcript
emissions, stride = generate_emissions(
    alignment_model, torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device),
    batch_size=args.batch_size
)
del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(full_transcript, romanize=True, language=langs_to_iso[info.language])
segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
spans = get_spans(tokens_starred, segments, blank_token)
word_timestamps = postprocess_results(text_starred, spans, stride, scores)

# Convert to mono for NeMo
temp_path = os.path.join(os.getcwd(), "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
mono_file = os.path.join(temp_path, "mono_file.wav")
torchaudio.save(mono_file, torch.from_numpy(audio_waveform).unsqueeze(0).float(), 16000, channels_first=True)

# Diarize with NeMo
logger.info("Performing diarization with NeMo...")
diarizer_config = create_config(temp_path)
diarizer_config.diarizer.clustering.parameters.max_num_speakers = 3  # Allow up to 3 speakers
msdd_model = NeuralDiarizer(cfg=diarizer_config).to(args.device)
msdd_model.diarize()
del msdd_model
torch.cuda.empty_cache()

# Read speaker timestamps
speaker_ts = []
rttm_file = os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
with open(rttm_file, "r") as f:
    for line in f:
        parts = line.split()
        s = int(float(parts[5]) * 1000)  # Start time in ms
        e = s + int(float(parts[8]) * 1000)  # End time in ms
        speaker = int(parts[11].split("_")[-1])  # Extract speaker number
        speaker_ts.append([s, e, speaker])

# Map words to speakers
wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

# Add punctuation if supported
if info.language in punct_model_langs:
    logger.info("Restoring punctuation...")
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    words_list = [x["word"] for x in wsm]
    labeled_words = punct_model.predict(words_list, chunk_size=230)
    ending_puncts = ".?!"
    model_puncts = ".,;:!?"
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
    for word_dict, (word, punct) in zip(wsm, labeled_words):
        if word and punct in ending_puncts and (word[-1] not in model_puncts or is_acronym(word)):
            word_dict["word"] = word + (punct if not word.endswith("..") else punct.rstrip("."))
else:
    logger.warning(f"Punctuation not available for {info.language}. Using original.")

# Finalize mappings
wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

# Adjust JSON structure for process_speakers.py
adjusted_ssm = [
    {
        "text": seg["text"],
        "start_time": seg["start_time"],  # Keep in milliseconds
        "end_time": seg["end_time"],      # Keep in milliseconds
        "speaker": f"Speaker {seg['speaker']}"  # Match process_speakers.py format
    }
    for seg in ssm
]

# Write to fixed output file
output_json = os.path.join(output_dir, "video_audio.json")
with open(output_json, "w", encoding="utf-8") as json_file:
    json.dump(adjusted_ssm, json_file, ensure_ascii=False, indent=4)
logger.info(f"JSON output saved to {output_json}")

# Cleanup
cleanup(temp_path)
logger.info("Processing complete.")
