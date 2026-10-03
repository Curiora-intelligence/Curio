import httpx
import pytest
from app.tools.browser import ExaBrowser, UnavailableBrowser, public_url
from app.tools.discovery import DiscoveryRequest, search_services
from app.services.ephemeral import EphemeralStore


def test_service_ranking_deterministic_explainable_and_constraints():
    request = DiscoveryRequest(category="biryani", latitude=17.4401, longitude=78.3489, budget_max=300,
                               preferences=["spicy chicken biryani"], availability="today")
    first = search_services(request)
    assert first == search_services(request)
    assert first['recommendations'][0]['id'] == 'demo-food-1'
    assert first['weights'] == {'rating': .3, 'distance': .25, 'budget_fit': .2, 'availability': .15, 'preference_match': .1}
    for recommendation in first['recommendations']:
        assert recommendation['fictional'] and len(recommendation['reasons']) == 5
        assert recommendation['price'] <= 300
    assert search_services(DiscoveryRequest(category="food", budget_max=1))['recommendations'] == []
    assert search_services(DiscoveryRequest(category="plumber", availability="today"))['location_required']
    with pytest.raises(ValueError):
        DiscoveryRequest(category="food", latitude=17)


async def test_browser_unavailable():
    backend = UnavailableBrowser()
    assert (await backend.search('food'))['error'] == 'tool_unavailable'
    assert (await backend.open('https://example.com'))['error'] == 'tool_unavailable'
    assert (await backend.find('id', 'a'))['error'] == 'tool_unavailable'


async def test_exa_search_open_find_with_redis_disabled():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"results": [{"url": "https://example.com/item", "title": "Item", "text": "Visible information\nPrice: 100\nMaterial unknown"}]})
    browser = ExaBrowser("test-key", EphemeralStore(), httpx.MockTransport(handler))
    results = await browser.search("item")
    page = await browser.open(results['results'][0]['page_id'])
    found = await browser.find(page['page_id'], 'PRICE')
    assert found['matches'][0]['line'] == 2
    assert len(requests) == 1 and requests[0].url.host == 'api.exa.ai'
    assert (await browser.open('https://example.com/another'))['untrusted']
    assert (await browser.find('missing', 'x'))['error'] == 'page_expired'


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'http://localhost/x', 'http://127.0.0.1/x', 'http://169.254.169.254/', 'http://host.internal', 'https://user:secret@example.com/'])
def test_browser_blocks_private_urls(url):
    with pytest.raises(ValueError):
        public_url(url)
