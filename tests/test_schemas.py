import pytest
from pydantic import ValidationError
from schemas.schemas import WSResponse, AudioQueryParams


def test_ws_response_schema():
    res = WSResponse(text="Hola mundo", sentence_id=1, is_final=True)
    assert res.text == "Hola mundo"
    assert res.sentence_id == 1
    assert res.is_final is True

    json_data = res.model_dump_json()
    assert '"text":"Hola mundo"' in json_data
    assert '"sentence_id":1' in json_data
    assert '"is_final":true' in json_data


def test_audio_query_params_valid():
    params = AudioQueryParams(lang_in="es", lang_out="en")
    assert params.lang_in == "es"
    assert params.lang_out == "en"

    params_auto = AudioQueryParams(lang_in="auto", lang_out="original")
    assert params_auto.lang_in == "auto"
    assert params_auto.lang_out == "original"


def test_audio_query_params_invalid():
    with pytest.raises(ValidationError):
        AudioQueryParams(lang_in="invalid_lang")

    with pytest.raises(ValidationError):
        AudioQueryParams(lang_out="klingon")
