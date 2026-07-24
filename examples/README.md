# Examples

Runnable scripts that talk to a live shrike server. Start one first:

```bash
shrike-serve --model models_cache/qwen2.5-0.5b-instruct   # serves on :8000
```

## `openai_client.py`

A dependency-free client for the OpenAI-compatible `/v1/chat/completions`
endpoint (standard library only — no `openai`, `httpx`, or `requests`).

```bash
python examples/openai_client.py "Explain paged attention in one sentence."
python examples/openai_client.py --stream "Count to five."
python examples/openai_client.py --base-url http://localhost:8000 --max-tokens 64 "Hi"
```

Because shrike speaks the OpenAI wire format, the official client works too:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="shrike",
    messages=[{"role": "user", "content": "Explain paged attention briefly."}],
)
print(resp.choices[0].message.content)
```
