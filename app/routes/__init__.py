"""
Brain Disease AI - Routes Package
"""
from app.routes.user_routes import router as user_router
from app.routes.scan_routes import router as scan_router
from app.routes.info_routes import router as info_router

__all__ = ["user_router", "scan_router", "info_router"]
