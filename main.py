"""
Brain Disease Detection AI - Main Application Entry Point

This is the main FastAPI application that brings together all modules:
- Authentication & User Management
- Brain Scan Upload & AI Analysis
- Disease Information & Hospital Recommendations
- AI Chatbot for Health Queries
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

# Import configuration
from app.config import settings

# Import database
from app.database import engine, Base, get_db

# Import routers
from app.auth.routes import router as auth_router
from app.routes.user_routes import router as user_router
from app.routes.scan_routes import router as scan_router
from app.routes.info_routes import router as info_router
from app.chatbot.routes import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Brain Disease Detection AI Application...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
    
    # Create necessary directories
    directories = [
        settings.UPLOAD_DIR,
        settings.MODEL_PATH,
        "static/images",
        "logs"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Required directories created")
    
    # Initialize AI model
    try:
        from app.ai_models import predictor as _predictor
        info = _predictor.BrainDiseasePredictor().get_model_info()
        if info.get("model_loaded"):
            logger.info("AI models loaded and ready for inference")
        else:
            logger.warning("AI models not loaded - check TensorFlow install and .h5 files in app/ai_models/weights/")
    except Exception as e:
        logger.error(f"AI model initialization failed: {e}", exc_info=True)
    
    logger.info("Application startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Brain Disease Detection AI Application...")
    logger.info("Application shutdown complete!")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## Brain Disease Detection AI Platform
    
    An AI-powered platform for early detection of brain diseases using medical imaging.
    
    ### Features:
    * **AI Analysis**: Deep learning models for brain scan analysis
    * **Disease Detection**: Support for Stroke, Epilepsy, Alzheimer's, Parkinson's, Brain Tumor
    * **Chatbot**: AI-powered health assistant for medical queries
    * **Secure**: JWT-based authentication with role-based access
    
    ### Supported Scan Types:
    * MRI (Magnetic Resonance Imaging)
    * CT (Computed Tomography)
    * PET (Positron Emission Tomography)
    """,
    version="1.0.0",
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    openapi_url="/api/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
    )


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
else:
    # Create static directory if it doesn't exist
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Configure templates
templates_path = Path(__file__).parent / "templates"
if not templates_path.exists():
    templates_path.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(templates_path))


# Include API routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(user_router, prefix="/api/v1/users", tags=["Users"])
app.include_router(scan_router, prefix="/api/v1/scans", tags=["Brain Scans"])
app.include_router(info_router, prefix="/api/v1/info", tags=["Information"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chatbot"])


# ============== Frontend Routes ==============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    try:
        return templates.TemplateResponse(
            request,
            name="index.html",
            context={}
        )
    except Exception as e:
        logger.error(f"Error rendering home page: {e}", exc_info=True)
        # Fallback to simple HTML if template fails
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain Disease AI</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="bg-light">
            <nav class="navbar navbar-dark bg-primary">
                <div class="container"><span class="navbar-brand">Brain Disease AI</span></div>
            </nav>
            <div class="container mt-5">
                <h1>Welcome to Brain Disease Detection AI</h1>
                <p>Early disease detection using advanced AI</p>
                <a href="/login" class="btn btn-primary">Login</a>
                <a href="/register" class="btn btn-secondary">Register</a>
            </div>
        </body>
        </html>
        """


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    try:
        return templates.TemplateResponse(request, name="login.html", context={})
    except Exception as e:
        logger.error(f"Error rendering login page: {e}", exc_info=True)
        return RedirectResponse(url="/", status_code=302)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page"""
    try:
        return templates.TemplateResponse(request, name="register.html", context={})
    except Exception as e:
        logger.error(f"Error rendering register page: {e}", exc_info=True)
        return RedirectResponse(url="/", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """User dashboard page"""
    try:
        return templates.TemplateResponse(request, name="dashboard.html", context={})
    except Exception as e:
        logger.error(f"Error rendering dashboard page: {e}", exc_info=True)
        return RedirectResponse(url="/", status_code=302)


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Scan upload page"""
    try:
        return templates.TemplateResponse(request, name="upload.html", context={})
    except Exception as e:
        logger.error(f"Error rendering upload page: {e}", exc_info=True)
        return RedirectResponse(url="/", status_code=302)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chatbot page"""
    try:
        return templates.TemplateResponse(request, name="chat.html", context={})
    except Exception as e:
        logger.error(f"Error rendering chat page: {e}", exc_info=True)
        return RedirectResponse(url="/", status_code=302)


@app.get("/dashboard/scans/{scan_id}", response_class=HTMLResponse)
async def scan_detail_page(request: Request, scan_id: str):
    """Scan detail/results page"""
    try:
        return templates.TemplateResponse(request, name="scan_detail.html", context={"scan_id": scan_id})
    except Exception as e:
        logger.error(f"Error rendering scan detail page: {e}", exc_info=True)
        return RedirectResponse(url="/", status_code=302)


# ============== Health Check & Info Routes ==============

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "environment": "development" if settings.DEBUG else "production"
    }


@app.get("/api/v1/status", tags=["Health"])
async def api_status():
    """API status endpoint with more details"""
    return {
        "status": "operational",
        "services": {
            "api": "running",
            "database": "connected",
            "ai_model": "ready" if not settings.DEBUG else "simulation_mode"
        },
        "supported_diseases": [
            "Stroke",
            "Epilepsy",
            "Alzheimer's Disease",
            "Parkinson's Disease",
            "Brain Tumor"
        ],
        "supported_scan_types": ["MRI", "CT", "PET"]
    }


# ============== Error Handlers ==============

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler for debugging"""
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}", exc_info=True)
    import traceback
    print("=" * 80, flush=True)
    print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    print("=" * 80, flush=True)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": str(exc), "status_code": 500}, status_code=500)
    return JSONResponse({"detail": "Internal server error", "status_code": 500}, status_code=500)


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 error handler"""
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Endpoint not found", "status_code": 404}, status_code=404)
    try:
        return templates.TemplateResponse(
            request,
            name="index.html",
            context={"error": "Page not found"},
            status_code=404
        )
    except Exception as e:
        logger.error(f"Error rendering 404 template: {e}", exc_info=True)
        return JSONResponse({"detail": "Page not found", "status_code": 404}, status_code=404)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    """Custom 500 error handler"""
    logger.error(f"Server error: {exc}", exc_info=True)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Internal server error", "status_code": 500}, status_code=500)
    try:
        return templates.TemplateResponse(
            request,
            name="index.html",
            context={"error": "Something went wrong"},
            status_code=500
        )
    except Exception as e:
        logger.error(f"Error rendering 500 template: {e}", exc_info=True)
        return JSONResponse({"detail": "Internal server error", "status_code": 500}, status_code=500)


# ============== Development Server ==============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        reload_excludes=["app.log", "uploads/*"],
        log_level="info" if not settings.DEBUG else "debug"
    )
