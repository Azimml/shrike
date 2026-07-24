# shrike/cli.py — interactive terminal chat REPL (Claude-Code-style)

Run as `python -m shrike.cli [--model models_cache/qwen2.5-0.5b-instruct] [--spec-ngram 0]`.
Stdlib only (argparse, readline, sys, time) + the existing shrike package. No new deps.

## Behavior
- Loads `LLMEngine` in-process at startup (print a short banner while loading:
  model name, KV blocks, load time). Drives `engine.step()` directly.
- REPL loop with a colored prompt `shrike> ` (ANSI, detect `sys.stdout.isatty()`;
  plain text when not a TTY). Use `readline` for line editing + in-session history.
- Multi-turn chat: keep `messages` list of {role, content}. Each user turn:
  tokenize the FULL conversation with
  `engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]`
  and submit via `engine.add_request(token_ids, SamplingParams(...), chat=False)`
  (returns req_id). Then loop `engine.step()` while `engine.has_work`, printing
  each token's text incrementally (same dangling-"�" buffering trick as
  scripts/chat.py). Append the assistant reply to `messages`.
- After each reply print a dim one-line status: `⏱ 42 tok · 1.3s · 32 tok/s · TTFT 180ms`.
- Ctrl+C during generation: abort that request cleanly —
  `engine.scheduler.finish(req, "aborted")` (req is `engine.requests[req_id]`),
  also `del engine.requests[req_id]`, print `[aborted]`, drop the partial reply
  from history, return to prompt. Ctrl+C at the prompt: print hint to use /exit.
  Ctrl+D or /exit: quit.

## Slash commands (line starting with `/`)
- `/help` — list commands
- `/clear` — reset conversation history
- `/metrics` — pretty-print `engine.metrics()`
- `/temp <float>` — set temperature (0 = greedy, default 0.7 for chat)
- `/max <int>` — set max_new_tokens (default 512)
- `/spec on|off` — toggle speculation: set `engine.scheduler.spec_ngram = 2` or `0`
- `/system <text>` — set/replace a system message at messages[0]
- `/exit` — quit
Unknown /command: print help hint. Prompt-caching means multi-turn re-prefill of
the shared history is mostly cache hits — mention that in /help for /metrics.

## Sampling defaults for chat
temperature 0.7, top_p 0.9, max_new_tokens 512. Note: speculation only engages
at temperature 0 (engine constraint) — `/spec on` should also print a hint
saying "takes effect at /temp 0".

## Also
Append a "## Interactive CLI" section to README.md (3 lines: command, /help pointer,
one-sentence description). Keep the file under ~250 lines, plain and readable,
matching the repo's code style (type hints, no over-abstraction).

## Acceptance
- `printf 'hi\n/exit\n' | python -m shrike.cli` works end-to-end without a TTY (prints reply).
- /clear, /metrics, /temp verified manually.
- No changes outside shrike/cli.py and README.md.
