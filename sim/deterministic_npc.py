"""Deterministic NPC voice for fair, credential-free model comparisons.

Scenario rules and authored intents remain the source of facts. This adapter
turns the latest NPC context into concise text without inference or sampling,
so every evaluated PM model sees the same coworker evidence.
"""

import re


class _Usage:
    input_tokens = 0
    output_tokens = 0


class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Response:
    stop_reason = "end_turn"
    usage = _Usage()

    def __init__(self, text):
        self.content = [_TextBlock(text)]


def _text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict)
            else getattr(block, "text", "")
            for block in content)
    return str(content)


def deterministic_reply(messages):
    transcript = "\n".join(_text_content(m.get("content", ""))
                           for m in messages)
    latest = _text_content(messages[-1].get("content", "")) if messages else ""
    prior_speech = "\n".join(
        _text_content(m.get("content", ""))
        for m in messages if m.get("role") == "assistant")

    def once(text):
        # Room turns recur every five minutes. Unchanged evidence should produce
        # silence rather than a verbatim status loop.
        return "PASS" if text in prior_speech else text

    discoveries = re.findall(
        r"You just discovered:\s*(.*?)\s*You've decided", latest, re.DOTALL)
    if discoveries:
        return once("Important update: " + discoveries[-1].strip())

    motivations = re.findall(
        r"Motivation:\s*(.*?)\s*React naturally", latest, re.DOTALL)
    if motivations:
        return once("Heads up: " + motivations[-1].strip())

    realizations = re.findall(
        r"you've just realized about '([^']+)':\s*([^\n]+)", transcript)
    if realizations:
        task, update = realizations[-1]
        return once("Current status on %s: %s" % (task, update.strip()))

    pictures = re.findall(
        r"your current picture of your work:\s*([^\n]+)", transcript)
    if pictures:
        return once("Current status: " + pictures[-1].strip())

    if "It's your turn:" in latest:
        return "PASS"
    if "meeting transcript" in latest.lower():
        return "Deterministic meeting summary: attendees reviewed the stated agenda."
    return once("Acknowledged. I'll continue with my assigned work and flag blockers.")


class _Messages:
    @staticmethod
    def create(**kwargs):
        return _Response(deterministic_reply(kwargs.get("messages") or []))


class DeterministicNPCClient:
    messages = _Messages()
