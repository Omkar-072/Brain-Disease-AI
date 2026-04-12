"""
Brain Disease AI - Email Service
Handles all email notifications using aiosmtplib
"""
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


async def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: Optional[str] = None
) -> bool:
    """
    Send an email using SMTP.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML body content
        text_content: Plain text body (optional)
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning("Email not configured. Skipping email send.")
        return False
    
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{settings.EMAILS_FROM_NAME} <{settings.EMAILS_FROM_EMAIL}>"
        message["To"] = to_email
        
        # Add text content if provided
        if text_content:
            message.attach(MIMEText(text_content, "plain"))
        
        # Add HTML content
        message.attach(MIMEText(html_content, "html"))
        
        # Send email
        await aiosmtplib.send(
            message,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            start_tls=True
        )
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        return False


async def send_login_notification(
    email: str,
    name: str,
    ip_address: str,
    login_time: str
) -> bool:
    """Send login notification email"""
    subject = "New Login to Your Brain Disease AI Account"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .info-box {{ background: white; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #667eea; }}
            .warning {{ color: #e74c3c; font-weight: bold; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Brain Disease AI</h1>
                <p>Login Notification</p>
            </div>
            <div class="content">
                <p>Hello <strong>{name}</strong>,</p>
                <p>We detected a new login to your account:</p>
                
                <div class="info-box">
                    <p><strong>📍 IP Address:</strong> {ip_address}</p>
                    <p><strong>🕐 Time:</strong> {login_time}</p>
                </div>
                
                <p class="warning">⚠️ If this wasn't you, please change your password immediately and contact support.</p>
                
                <p>Stay safe,<br>The Brain Disease AI Team</p>
            </div>
            <div class="footer">
                <p>This is an automated notification. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return await send_email(email, subject, html_content)


async def send_password_reset_email(
    email: str,
    name: str,
    otp: str,
    token: str
) -> bool:
    """Send password reset OTP email"""
    subject = "Password Reset Request - Brain Disease AI"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .otp-box {{ background: #667eea; color: white; font-size: 32px; letter-spacing: 8px; text-align: center; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .warning {{ color: #e74c3c; font-size: 14px; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Brain Disease AI</h1>
                <p>Password Reset</p>
            </div>
            <div class="content">
                <p>Hello <strong>{name}</strong>,</p>
                <p>We received a request to reset your password. Use the OTP below to complete the process:</p>
                
                <div class="otp-box">
                    <strong>{otp}</strong>
                </div>
                
                <p><strong>⏰ This OTP is valid for 15 minutes.</strong></p>
                
                <p class="warning">⚠️ If you didn't request this password reset, please ignore this email or contact support if you have concerns.</p>
                
                <p>Best regards,<br>The Brain Disease AI Team</p>
            </div>
            <div class="footer">
                <p>This is an automated message. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return await send_email(email, subject, html_content)


async def send_scan_result_notification(
    email: str,
    name: str,
    scan_id: int,
    disease: str,
    confidence: float
) -> bool:
    """Send scan result notification email"""
    subject = "Your Brain Scan Results are Ready - Brain Disease AI"
    
    confidence_percent = f"{confidence * 100:.1f}%"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .result-box {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; border: 2px solid #667eea; }}
            .result-disease {{ font-size: 24px; color: #667eea; font-weight: bold; }}
            .result-confidence {{ font-size: 18px; color: #666; margin-top: 10px; }}
            .disclaimer {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ffc107; font-size: 14px; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Brain Disease AI</h1>
                <p>Scan Results Available</p>
            </div>
            <div class="content">
                <p>Hello <strong>{name}</strong>,</p>
                <p>Your brain scan analysis (Scan #{scan_id}) is complete. Here's a summary:</p>
                
                <div class="result-box">
                    <div class="result-disease">{disease.replace('_', ' ').title()}</div>
                    <div class="result-confidence">Confidence: {confidence_percent}</div>
                </div>
                
                <div class="disclaimer">
                    <strong>⚠️ Medical Disclaimer:</strong> This AI-generated result is for informational purposes only and should not be considered as a medical diagnosis. Please consult a qualified healthcare professional for proper diagnosis and treatment.
                </div>
                
                <p>Log in to your account to view the detailed report and recommendations.</p>
                
                <p>Best regards,<br>The Brain Disease AI Team</p>
            </div>
            <div class="footer">
                <p>This is an automated notification. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return await send_email(email, subject, html_content)


async def send_welcome_email(email: str, name: str) -> bool:
    """Send welcome email to new users"""
    subject = "Welcome to Brain Disease AI! 🧠"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .feature {{ display: flex; margin: 15px 0; padding: 15px; background: white; border-radius: 5px; }}
            .feature-icon {{ font-size: 24px; margin-right: 15px; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Welcome to Brain Disease AI!</h1>
            </div>
            <div class="content">
                <p>Hello <strong>{name}</strong>,</p>
                <p>Thank you for joining Brain Disease AI! We're excited to have you on board.</p>
                
                <h3>What you can do:</h3>
                
                <div class="feature">
                    <span class="feature-icon">🔬</span>
                    <div>
                        <strong>Upload Brain Scans</strong>
                        <p>Upload MRI/CT scans for AI-powered analysis</p>
                    </div>
                </div>
                
                <div class="feature">
                    <span class="feature-icon">🤖</span>
                    <div>
                        <strong>AI Detection</strong>
                        <p>Get instant predictions for 5 brain diseases</p>
                    </div>
                </div>
                
                <div class="feature">
                    <span class="feature-icon">💬</span>
                    <div>
                        <strong>Chat Support</strong>
                        <p>Ask our AI chatbot about symptoms and precautions</p>
                    </div>
                </div>
                
                <p>Get started by logging into your account and uploading your first scan!</p>
                
                <p>Best regards,<br>The Brain Disease AI Team</p>
            </div>
            <div class="footer">
                <p>© 2024 Brain Disease AI. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return await send_email(email, subject, html_content)
