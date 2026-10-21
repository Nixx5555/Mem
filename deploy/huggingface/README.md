---
title: MemGate
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: A memory layer that decides what an LLM keeps and forgets
---

# MemGate

Chat with a model that never sees the raw conversation. MemGate decides, message
by message, what to keep verbatim, what to compress and what to forget under a
fixed token budget, and the model answers only from what MemGate kept. The side
panel shows the three memory tiers, every routing decision and the exact context
sent with each question.

Capstone project, SVKM's NMIMS Indore: Simar Singh Khanuja and Yash Ramchandani.
Code: https://github.com/Nixx5555/Mem
