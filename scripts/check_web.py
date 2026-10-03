"""Verify the existing Exa adapter without logging credentials."""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()
from app.core.settings import Settings
from app.services.ephemeral import EphemeralStore
from app.tools.browser import ExaBrowser

async def main():
    settings = Settings()
    if settings.browser_backend != 'exa' or not settings.exa_api_key:
        print('WEB UNAVAILABLE: set CURIO_BROWSER_BACKEND=exa and EXA_API_KEY in the backend .env. No real search was performed.')
        return 2
    browser = ExaBrowser(settings.exa_api_key, EphemeralStore())
    try:
        result = await browser.search('site:zomato.com spicy chicken biryani Hyderabad', topn=3)
        print(json.dumps({'configured': True, 'results': result['results']}, ensure_ascii=False))
        if not result['results']:
            return 1
        page = await browser.open(result['results'][0]['page_id'])
        print(json.dumps({'open_succeeded': bool(page.get('text')), 'url': page.get('url')}))
        return 0
    except Exception as exc:
        print(json.dumps({'verified': False, 'error_type': type(exc).__name__, 'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}))
        return 1

if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
