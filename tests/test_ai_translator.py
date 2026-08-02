import pytest
from core.ai_translator import NullTranslator, GenericAITranslator


@pytest.mark.asyncio
async def test_null_translator():
    translator = NullTranslator()
    assert translator.is_enabled is False

    tokens = []
    async for token in translator.translate_stream("Hello world", "en", "es"):
        tokens.append(token)

    assert tokens == ["Hello world"]


@pytest.mark.asyncio
async def test_generic_ai_translator_init():
    # If openai module is available, test initialization
    try:
        translator = GenericAITranslator(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="gemma4"
        )
        assert translator.is_enabled is True
    except ImportError:
        pytest.skip("openai package not installed")
