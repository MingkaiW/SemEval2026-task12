import os
import anthropic

print("KEY_SET", bool(os.getenv("ANTHROPIC_API_KEY")))

try:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    resp = client.messages.create(
        model=model_name,
        max_tokens=5,
        messages=[{"role": "user", "content": "Hello"}],
    )
    print("OK", resp.content[0].text)
except Exception as e:
    print("ERR", repr(e))
    print("ERR_TYPE", type(e))
    print("ERR_DICT", getattr(e, "__dict__", None))
    print("ERR_MESSAGE", getattr(e, "message", None))
    print("ERR_BODY", getattr(e, "body", None))
    print("ERR_RESPONSE", getattr(e, "response", None))
