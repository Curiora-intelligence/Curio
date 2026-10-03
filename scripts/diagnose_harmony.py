"""Print structural diagnostics only; never log generated reasoning or prompts."""
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from app.core.models import cached_model
os.environ['CURIO_MLX_TEXT_MODEL'] = cached_model('mlx-community/gpt-oss-20b-MXFP4-Q8')
from app.runtimes import harmony
from openai_harmony import Role
original = harmony.parse

def diagnose(tokens, tools):
    print({'token_count': len(tokens), 'last_token': tokens[-1], 'expected_stops': harmony.encoding().stop_tokens_for_assistant_actions()}, flush=True)
    try:
        entries = harmony.encoding().parse_messages_from_completion_tokens(tokens, Role.ASSISTANT)
        print([{'role': m.author.role, 'channel': m.channel, 'recipient': m.recipient, 'content_lengths': [len(getattr(c, 'text', '')) for c in m.content]} for m in entries], flush=True)
    except Exception as exc:
        print('Parser failure type: ' + type(exc).__name__, flush=True)
    try:
        return original(tokens, tools)
    except RuntimeError as exc:
        print('Runtime contract failure: ' + str(exc), flush=True)
        raise
harmony.parse = diagnose
import asyncio
from app.services.curio import CurioService, MAX_TEXT_TOKENS
service = CurioService()
print({"max_text_tokens": MAX_TEXT_TOKENS}, flush=True)
async def main():
    try:
        answer, cid = await service.respond('What is 2 + 2? Answer with only the number.', user_id='harmony-diagnostic')
        print({"answer": answer, "conversation_id": cid}, flush=True)
    finally:
        await service.close()
asyncio.run(main())
