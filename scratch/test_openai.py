import asyncio
from openai import AsyncOpenAI
import httpx

async def test():
    client = AsyncOpenAI(api_key="sk-123", base_url="http://localhost:11434/v1")
    
    # Mock httpx client to intercept request
    request_json = None
    
    def handle_request(request):
        nonlocal request_json
        import json
        request_json = json.loads(request.read())
        return httpx.Response(200, json={"choices": [{"delta": {"content": "test"}}], "object": "chat.completion.chunk"})

    # Patch the transport
    client._client._transport = httpx.MockTransport(handle_request)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "es",
                    "target_lang_code": "en",
                    "text": "hola"
                }
            ]
        }
    ]
    
    try:
        await client.chat.completions.create(
            model="translategemma",
            messages=messages,
            stream=True
        )
    except Exception as e:
        print("Exception:", e)
    
    print("Serialized request:", request_json)

if __name__ == "__main__":
    asyncio.run(test())
