"""Whisper API adapter using bounded PCM uploads and source-relative timings."""

import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

from openai import APIConnectionError, APIError, APIStatusError, AuthenticationError, OpenAI, RateLimitError

from app.config import load_settings

LANGUAGES = frozenset({
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs",
    "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu",
    "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka",
    "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml",
    "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
    "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo",
    "yue", "zh",
})

CHUNK_SECONDS = 600  # 19.2 MB at 16 kHz mono PCM, below the API's 25 MB limit.


def transcribe_api(audio, language, config_path=None):
    settings = load_settings(config_path)
    if not settings.llm.api_key:
        raise ValueError("Add your OpenAI API key to [llm] api_key in config.toml, then retry.")
    info = SimpleNamespace(language=language or "unknown", language_probability=None)

    def segments():
        try:
            with (
                OpenAI(api_key=settings.llm.api_key, timeout=600, max_retries=2) as client,
                wave.open(str(audio), "rb") as source,
                tempfile.TemporaryDirectory(prefix="whisper-", dir=Path(audio).parent) as directory,
            ):
                frames_per_chunk = source.getframerate() * CHUNK_SECONDS
                frame_offset = 0
                while frame_offset < source.getnframes():
                    frames = source.readframes(frames_per_chunk)
                    chunk = Path(directory) / "chunk.wav"
                    with wave.open(str(chunk), "wb") as output:
                        output.setparams(source.getparams())
                        output.writeframes(frames)
                    if chunk.stat().st_size >= 25_000_000:
                        raise ValueError("Audio chunk exceeds the Whisper API upload limit.")
                    with chunk.open("rb") as upload:
                        result = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=upload,
                            response_format="verbose_json",
                            timestamp_granularities=["segment", "word"],
                            **({"language": language} if language else {}),
                        )
                    if result.language:
                        info.language = result.language
                    offset = frame_offset / source.getframerate()
                    words = result.words or []
                    word_index = 0
                    for segment in result.segments or []:
                        segment_words = []
                        while word_index < len(words):
                            word = words[word_index]
                            if (word.start + word.end) / 2 >= segment.end:
                                break
                            word_index += 1
                            segment_words.append(
                                SimpleNamespace(
                                    word=word.word,
                                    start=word.start + offset,
                                    end=word.end + offset,
                                    probability=None,
                                )
                            )
                        yield SimpleNamespace(
                            start=segment.start + offset,
                            end=segment.end + offset,
                            text=segment.text,
                            avg_logprob=segment.avg_logprob,
                            no_speech_prob=segment.no_speech_prob,
                            words=segment_words,
                        )
                    if result.text.strip() and not result.segments:
                        raise ValueError("Whisper returned text without timestamps. Retry transcription.")
                    frame_offset += len(frames) // (source.getsampwidth() * source.getnchannels())
        except AuthenticationError:
            raise ValueError("OpenAI rejected the API key. Check [llm] api_key in config.toml.") from None
        except RateLimitError:
            raise ValueError(
                "OpenAI quota or rate limit reached. Check API billing and retry later."
            ) from None
        except APIConnectionError:
            raise ValueError("Could not reach OpenAI. Check your connection and retry.") from None
        except APIStatusError:
            raise ValueError("OpenAI could not transcribe this audio. Retry later.") from None
        except ValueError:
            raise
        except (APIError, OSError, TypeError, AttributeError):
            raise ValueError(
                "Whisper API transcription failed. Check your configuration and retry."
            ) from None

    return segments(), info
