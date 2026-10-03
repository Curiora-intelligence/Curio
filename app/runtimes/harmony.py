"""Official token-based Harmony rendering/parsing. Never show analysis to users."""
from functools import lru_cache
import json

from openai_harmony import (Author, Conversation, DeveloperContent, HarmonyEncodingName, Message,
                            ReasoningEffort, RenderConversationConfig, Role, SystemContent,
                            ToolDescription, load_harmony_encoding)

from app.runtimes.turn import ModelTurn, ToolRequest


@lru_cache(maxsize=1)
def encoding():
    # First use needs the official tokenizer vocabulary cached (see README).
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def literal(text: str) -> str:
    return text.replace("<|", "< |")


def render(messages: list[dict], tools: list[dict]) -> list[int]:
    system = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
    instructions = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    developer = DeveloperContent.new().with_instructions(instructions)
    if tools:
        developer.with_function_tools([ToolDescription.new(t["alias"], t["description"], t["parameters"]) for t in tools])
    conversation = [Message.from_role_and_content(Role.SYSTEM, system),
                    Message.from_role_and_content(Role.DEVELOPER, developer)]
    for item in messages:
        if "harmony" in item:
            conversation.append(Message.from_dict(item["harmony"]))
        elif item["role"] == "system":
            continue
        elif item["role"] == "tool":
            conversation.append(Message.from_author_and_content(Author.new(Role.TOOL, item["name"]),
                                                                 literal(item["content"])))
        else:
            message = Message.from_role_and_content(Role(item["role"]), literal(item["content"]))
            if item["role"] == "assistant":
                message.with_channel("final")
            conversation.append(message)
    return encoding().render_conversation_for_completion(Conversation.from_messages(conversation), Role.ASSISTANT,
                                                         RenderConversationConfig(auto_drop_analysis=False))


def parse(tokens: list[int], tools: list[dict]) -> ModelTurn:
    enc = encoding()
    # A length stop is not a completed final answer or a safe executable call.
    if not tokens or tokens[-1] not in enc.stop_tokens_for_assistant_actions():
        raise RuntimeError("GPT-OSS stopped before completing a final response or tool call.")
    entries = enc.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT)
    if any(message.author.role != Role.ASSISTANT for message in entries):
        raise RuntimeError("Model attempted to generate a non-assistant message.")
    handoff = tokens[-1] == enc.encode("<|call|>", allowed_special="all")[0]
    if handoff != bool(entries and entries[-1].recipient and entries[-1].recipient != "all"):
        raise RuntimeError("Harmony recipient does not match the action stop token.")
    result = ModelTurn(continuation=[{"harmony": message.to_dict()} for message in entries])
    aliases = {"functions." + tool["alias"]: tool["name"] for tool in tools}
    for message in entries:
        content = "".join(getattr(part, "text", "") for part in message.content)
        if message.recipient and message.recipient != "all":
            try:
                args = json.loads(content)
            except json.JSONDecodeError:
                args = content  # Registry turns malformed arguments into a tool error.
            result.tool_calls.append(ToolRequest(aliases.get(message.recipient, message.recipient), args, message.recipient))
        elif message.channel == "final" and content.strip():
            result.final = content.strip()
    if result.tool_calls:
        result.final = None
    elif not result.final:
        raise RuntimeError("GPT-OSS did not produce a final channel or a tool call.")
    return result
