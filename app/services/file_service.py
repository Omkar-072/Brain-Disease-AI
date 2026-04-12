"""
Brain Disease AI - File Upload Service
Handles secure file uploads for brain scan images
"""
import os
import hashlib
import aiofiles
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException, status
from datetime import datetime
import uuid

from app.config import get_settings

settings = get_settings()


class FileUploadService:
    """Service for handling file uploads"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.max_size = settings.MAX_UPLOAD_SIZE
        self.allowed_extensions = settings.allowed_extensions_list
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_extension(self, filename: str) -> bool:
        """Validate file extension"""
        ext = Path(filename).suffix.lower()
        return ext in self.allowed_extensions
    
    def _generate_unique_filename(self, original_filename: str, user_id: int) -> str:
        """Generate a unique filename to prevent collisions"""
        ext = Path(original_filename).suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"user_{user_id}_{timestamp}_{unique_id}{ext}"
    
    def _get_user_upload_dir(self, user_id: int) -> Path:
        """Get user-specific upload directory"""
        user_dir = self.upload_dir / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for integrity verification"""
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def save_upload(
        self,
        file: UploadFile,
        user_id: int
    ) -> Tuple[str, str, int, str]:
        """
        Save uploaded file securely.
        
        Args:
            file: The uploaded file
            user_id: ID of the user uploading
            
        Returns:
            Tuple of (file_path, original_filename, file_size, file_hash)
        """
        # Validate file extension
        if not self._validate_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(self.allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file size
        if file_size > self.max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size: {self.max_size / (1024*1024):.1f}MB"
            )
        
        # Validate file is not empty
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Generate unique filename and path
        unique_filename = self._generate_unique_filename(file.filename, user_id)
        user_dir = self._get_user_upload_dir(user_id)
        file_path = user_dir / unique_filename
        
        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        
        # Calculate file hash
        file_hash = await self._calculate_file_hash(file_path)
        
        return str(file_path), file.filename, file_size, file_hash
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file from storage"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception:
            return False
    
    def get_file_path(self, file_path: str) -> Optional[Path]:
        """Get absolute file path if it exists"""
        path = Path(file_path)
        if path.exists():
            return path
        return None
    
    async def get_file_info(self, file_path: str) -> Optional[dict]:
        """Get information about a stored file"""
        path = Path(file_path)
        if not path.exists():
            return None
        
        stat = path.stat()
        file_hash = await self._calculate_file_hash(path)
        
        return {
            "path": str(path),
            "filename": path.name,
            "size": stat.st_size,
            "hash": file_hash,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime)
        }


# Global instance
file_upload_service = FileUploadService()
