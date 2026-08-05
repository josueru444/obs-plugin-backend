from pydantic import BaseModel, field_validator

# Supported ISO 639-1 language codes for Whisper
SUPPORTED_LANGUAGES = {
    "auto", "original",
    "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr",
    "cs", "da", "nl", "en", "et", "fi", "fr", "gl", "de", "el",
    "he", "hi", "hu", "is", "id", "it", "ja", "kn", "kk", "ko",
    "lv", "lt", "mk", "ms", "mr", "mi", "ne", "no", "fa", "pl",
    "pt", "ro", "ru", "sr", "sk", "sl", "es", "sw", "sv", "tl",
    "ta", "th", "tr", "uk", "ur", "vi", "cy",
}


class WSResponse(BaseModel):
    """Response sent back to the OBS plugin over WebSocket."""

    text: str
    sentence_id: int
    is_final: bool

    model_config = {"frozen": True}


class AudioQueryParams(BaseModel):
    """Query parameters received from the OBS plugin on WebSocket connect."""

    lang_in: str = "es"
    lang_out: str = "original"
    show_partial: bool = True
    partial_strategy: str = "buffered"

    @field_validator("partial_strategy")
    @classmethod
    def validate_partial_strategy(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"buffered", "streaming"}:
            raise ValueError("partial_strategy must be 'buffered' or 'streaming'")
        return v

    @field_validator("lang_in")
    @classmethod
    def validate_lang_in(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported lang_in '{v}'. "
                f"Must be one of: {sorted(SUPPORTED_LANGUAGES)}"
            )
        return v

    @field_validator("lang_out")
    @classmethod
    def validate_lang_out(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported lang_out '{v}'. "
                f"Must be one of: {sorted(SUPPORTED_LANGUAGES)}"
            )
        return v


class TextTranslateRequest(BaseModel):
    text: str
    is_final: bool
    sentence_id: int
