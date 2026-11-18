---
title: MaskWall Firewall API
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Prompt-injection and leak firewall in front of an LLM
---

# MaskWall firewall API

FastAPI service for [MaskWall](https://github.com/namyaa6/MaskWall): scans a message for prompt injection with a
fine-tuned int8 MiniLM, applies a block/warn/allow policy, calls the LLM, and scans the reply for leaks.

Interactive docs: `/docs`. Health: `/health`. The web app that uses this API is linked from the GitHub repository.
Message text is never stored (only verdicts, scores, counts and timings).
