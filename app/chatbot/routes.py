"""
Brain Disease AI - Chatbot API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
import logging

logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session
from typing import Optional
import uuid
from datetime import datetime

from app.database import get_db, User, ChatSession, ChatMessage
from app.schemas import (
    ChatMessageCreate, ChatMessageResponse, ChatSessionResponse, ChatBotResponse
)
from app.auth.security import get_current_user_flexible, get_current_active_user  # Import both
from app.chatbot.engine import ChatbotEngine
from typing import Union

router = APIRouter(tags=["Chatbot"])

# Global chatbot instance
chatbot = ChatbotEngine()


@router.post("/message", response_model=ChatBotResponse)
async def send_message(
    message: ChatMessageCreate,
    session_id_query: Optional[str] = None,
    db: Session = Depends(get_db),
    request: Request = None  # to manually check for user if needed
):
    """
    Send a message to the chatbot and get a response.
    
    - **message**: User message text (can include session_id in body)
    - **session_id_query**: Optional session ID from query param
    """
    # 1. Resolve session_id (from body OR query)
    # The frontend usually sends it in the JSON body
    session_id_str = message.session_id or session_id_query
    
    # 2. Try to get user if they are logged in (optional)
    current_user = None
    try:
        from app.auth.security import get_current_user
        # Manually invoke dependency-like logic or use Request
        # For simplicity in this fix, we'll try to find the user if a token exists
        auth_header = request.headers.get("Authorization")
        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
            
        if token:
            from app.auth.security import verify_token
            token_data = verify_token(token, "access")
            if token_data:
                current_user = db.query(User).filter(User.id == token_data.user_id).first()
    except Exception as e:
        logger.debug(f"Guest user using chatbot: {e}")

    session = None
    
    # 3. Only save to DB if user is logged in
    if current_user and current_user.is_active:
        # Get or create chat session
        if session_id_str and session_id_str.isdigit():
             session_id_int = int(session_id_str)
             session = db.query(ChatSession).filter(
                ChatSession.id == session_id_int,
                ChatSession.user_id == current_user.id
            ).first()
        
        if not session:
            # Create new session if none found or if session_id was a random string
            session = ChatSession(
                user_id=current_user.id,
                session_token=session_id_str or str(uuid.uuid4()),
                is_active=True
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            
        # Save user message
        user_msg = ChatMessage(
            session_id=session.id,
            sender="user",
            content=message.message
        )
        db.add(user_msg)
    
    # 4. Get chatbot response
    response = chatbot.chat(message.message)
    
    # 5. Save bot response if session exists
    if session:
        bot_msg = ChatMessage(
            session_id=session.id,
            sender="bot",
            content=response["message"],
            detected_intent=response.get("intent"),
            confidence=response.get("confidence")
        )
        db.add(bot_msg)
        db.commit()
    
    return ChatBotResponse(
        response=response["message"],
        intent=response.get("intent"),
        confidence=response.get("confidence"),
        suggestions=response.get("suggestions", [])
    )


@router.post("/session/start", response_model=ChatSessionResponse)
async def start_chat_session(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start a new chat session"""
    session = ChatSession(
        user_id=current_user.id,
        session_token=str(uuid.uuid4()),
        is_active=True
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Add welcome message
    welcome_msg = ChatMessage(
        session_id=session.id,
        sender="bot",
        content="Hello! I'm your Brain Health Assistant. How can I help you today?",
        detected_intent="greeting",
        confidence=1.0
    )
    db.add(welcome_msg)
    db.commit()
    
    return ChatSessionResponse(
        session_id=session.id,
        session_token=session.session_token,
        started_at=session.started_at,
        messages=[ChatMessageResponse(
            id=welcome_msg.id,
            sender=welcome_msg.sender,
            content=welcome_msg.content,
            detected_intent=welcome_msg.detected_intent,
            sent_at=welcome_msg.sent_at
        )]
    )


@router.get("/session/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get chat session with message history"""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.sent_at.asc()).all()
    
    return ChatSessionResponse(
        session_id=session.id,
        session_token=session.session_token,
        started_at=session.started_at,
        messages=[ChatMessageResponse(
            id=msg.id,
            sender=msg.sender,
            content=msg.content,
            detected_intent=msg.detected_intent,
            sent_at=msg.sent_at
        ) for msg in messages]
    )


@router.post("/session/{session_id}/end")
async def end_chat_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """End a chat session"""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    session.is_active = False
    session.ended_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Chat session ended", "session_id": session_id}


@router.get("/sessions")
async def get_my_chat_sessions(
    skip: int = 0,
    limit: int = 10,
    active_only: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of user's chat sessions"""
    query = db.query(ChatSession).filter(ChatSession.user_id == current_user.id)
    
    if active_only:
        query = query.filter(ChatSession.is_active == True)
    
    sessions = query.order_by(
        ChatSession.started_at.desc()
    ).offset(skip).limit(limit).all()
    
    return {
        "sessions": [
            {
                "id": s.id,
                "session_token": s.session_token,
                "started_at": s.started_at,
                "ended_at": s.ended_at,
                "is_active": s.is_active,
                "message_count": len(s.messages)
            }
            for s in sessions
        ]
    }


@router.post("/quick-query")
async def quick_query(
    question: str,
    db: Session = Depends(get_db)
):
    """
    Quick query endpoint (no authentication required).
    For simple FAQ-style questions without saving to database.
    """
    response = chatbot.chat(question)
    
    return {
        "question": question,
        "response": response["message"],
        "intent": response.get("intent"),
        "suggestions": response.get("suggestions", []),
        "disclaimer": "This is for informational purposes only. Consult a healthcare professional for medical advice."
    }
