from fastapi import APIRouter
from app.tools.discovery import DiscoveryRequest, search_services

discovery_router = APIRouter(prefix="/discovery", tags=["Discovery"])


@discovery_router.post("/")
async def discovery(request: DiscoveryRequest):
    return search_services(request)
