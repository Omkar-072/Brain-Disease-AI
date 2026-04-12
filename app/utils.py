"""
Brain Disease Detection AI - Utility Functions

Common utility functions used across the application.
"""

import os
import re
import uuid
import hashlib
import secrets
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# ============== String Utilities ==============

def generate_uuid() -> str:
    """Generate a unique UUID string."""
    return str(uuid.uuid4())


def generate_short_id(length: int = 8) -> str:
    """Generate a short random ID."""
    return secrets.token_urlsafe(length)[:length]


def generate_otp(length: int = 6) -> str:
    """Generate a numeric OTP of specified length."""
    return ''.join([str(secrets.randbelow(10)) for _ in range(length)])


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage."""
    # Remove path separators
    filename = os.path.basename(filename)
    # Replace special characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    return filename


def mask_email(email: str) -> str:
    """Mask email for display (john@example.com -> j***@example.com)."""
    if '@' not in email:
        return email
    name, domain = email.split('@', 1)
    if len(name) <= 2:
        masked_name = name[0] + '*'
    else:
        masked_name = name[0] + '*' * (len(name) - 2) + name[-1]
    return f"{masked_name}@{domain}"


def mask_phone(phone: str) -> str:
    """Mask phone number for display (1234567890 -> ******7890)."""
    if len(phone) < 4:
        return phone
    return '*' * (len(phone) - 4) + phone[-4:]


# ============== Hash Utilities ==============

def hash_string(text: str, algorithm: str = 'sha256') -> str:
    """Hash a string using specified algorithm."""
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def hash_file(file_path: str, algorithm: str = 'sha256') -> str:
    """Calculate hash of a file."""
    hasher = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# ============== Date/Time Utilities ==============

def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.utcnow()


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string."""
    return dt.strftime(format_str)


def parse_datetime(date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Parse string to datetime."""
    return datetime.strptime(date_str, format_str)


def time_ago(dt: datetime) -> str:
    """Convert datetime to human-readable 'time ago' format."""
    now = datetime.utcnow()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif seconds < 2592000:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    else:
        return dt.strftime("%B %d, %Y")


def is_expired(dt: datetime, ttl_seconds: int) -> bool:
    """Check if a datetime has expired based on TTL."""
    return datetime.utcnow() > dt + timedelta(seconds=ttl_seconds)


# ============== File Utilities ==============

def ensure_directory(path: str) -> Path:
    """Ensure a directory exists, create if not."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_extension(filename: str) -> str:
    """Get file extension without dot."""
    return os.path.splitext(filename)[1].lstrip('.')


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def is_valid_image(filename: str) -> bool:
    """Check if file is a valid image based on extension."""
    valid_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
    ext = get_file_extension(filename).lower()
    return ext in valid_extensions


def is_valid_medical_image(filename: str) -> bool:
    """Check if file is a valid medical image."""
    valid_extensions = {'jpg', 'jpeg', 'png', 'dcm', 'nii', 'nii.gz'}
    ext = get_file_extension(filename).lower()
    # Handle .nii.gz
    if filename.lower().endswith('.nii.gz'):
        return True
    return ext in valid_extensions


# ============== Validation Utilities ==============

def is_valid_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_phone(phone: str) -> bool:
    """Validate phone number format."""
    pattern = r'^\+?[\d\s-]{10,15}$'
    return bool(re.match(pattern, phone))


def is_strong_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength.
    Returns (is_valid, message).
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Password is strong"


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input by removing potentially harmful content."""
    if not text:
        return ""
    # Truncate
    text = text[:max_length]
    # Remove null bytes
    text = text.replace('\x00', '')
    # Basic XSS prevention
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    return text.strip()


# ============== Dictionary Utilities ==============

def filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from dictionary."""
    return {k: v for k, v in data.items() if v is not None}


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_dict(data: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


# ============== List Utilities ==============

def paginate(items: List, page: int, page_size: int) -> Dict[str, Any]:
    """Paginate a list of items."""
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    
    return {
        'items': items[start:end],
        'page': page,
        'page_size': page_size,
        'total': total,
        'total_pages': (total + page_size - 1) // page_size,
        'has_next': end < total,
        'has_prev': page > 1
    }


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split list into chunks."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# ============== Response Utilities ==============

def success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """Create a standard success response."""
    response = {
        'success': True,
        'message': message
    }
    if data is not None:
        response['data'] = data
    return response


def error_response(message: str, errors: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a standard error response."""
    response = {
        'success': False,
        'message': message
    }
    if errors:
        response['errors'] = errors
    return response


# ============== Logging Utilities ==============

def log_request(request_id: str, method: str, path: str, user_id: Optional[str] = None):
    """Log API request."""
    logger.info(f"[{request_id}] {method} {path} - User: {user_id or 'anonymous'}")


def log_error(request_id: str, error: Exception, context: Optional[Dict] = None):
    """Log error with context."""
    logger.error(f"[{request_id}] Error: {str(error)}", exc_info=True, extra=context or {})
