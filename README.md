# Brain Disease Detection AI

An AI-powered web application for early detection of brain diseases using medical imaging and deep learning.

## 🧠 Features

- **AI-Powered Analysis**: Deep learning models (CNN) for brain scan analysis
- **Multi-Disease Detection**: Support for 5 major brain conditions:
  - Stroke
  - Epilepsy
  - Alzheimer's Disease
  - Parkinson's Disease
  - Brain Tumor
- **Medical Chatbot**: AI assistant for health queries
- **Secure Authentication**: JWT-based auth with bcrypt password hashing
- **Modern Dashboard**: Track scans, view results, and manage health data
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   cd brain_disease_ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your configuration
   # Required: SECRET_KEY (generate a secure random string)
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Access the application**
   - Open browser: http://localhost:8000
   - API Docs: http://localhost:8000/api/docs (debug mode only)

## 📁 Project Structure

```
brain_disease_ai/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── app/
│   ├── config.py          # Application configuration
│   ├── utils.py           # Utility functions
│   ├── schemas.py         # Pydantic schemas
│   ├── database/
│   │   ├── connection.py  # Database connection
│   │   └── models.py      # SQLAlchemy models
│   ├── auth/
│   │   ├── security.py    # JWT & password utilities
│   │   └── routes.py      # Authentication endpoints
│   ├── routes/
│   │   ├── user_routes.py # User management
│   │   ├── scan_routes.py # Brain scan operations
│   │   └── info_routes.py # Disease information
│   ├── services/
│   │   ├── email_service.py   # Email notifications
│   │   └── file_service.py    # File handling
│   ├── ai_models/
│   │   ├── preprocessing.py   # Image preprocessing
│   │   ├── model.py          # CNN architecture
│   │   ├── predictor.py      # Prediction service
│   │   └── train.py          # Model training
│   └── chatbot/
│       ├── engine.py         # Chatbot logic
│       └── routes.py         # Chat endpoints
├── templates/              # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── chat.html
│   └── scan_detail.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following:

```env
# Application
APP_NAME="Brain Disease AI"
DEBUG=true
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=sqlite:///./brain_disease.db
# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/brain_disease_db

# Email (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# AI Model
MODEL_PATH=./models/brain_disease_model.h5
AI_FRAMEWORK=tensorflow
```

## 📡 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | User login |
| POST | `/api/v1/auth/refresh` | Refresh token |
| POST | `/api/v1/auth/password-reset/request` | Request password reset |
| POST | `/api/v1/auth/password-reset/verify` | Verify reset OTP |
| POST | `/api/v1/auth/password-reset/confirm` | Set new password |

### Brain Scans
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/scans/upload` | Upload scan for analysis |
| GET | `/api/v1/scans/` | Get user's scans |
| GET | `/api/v1/scans/{id}` | Get scan details |
| DELETE | `/api/v1/scans/{id}` | Delete scan |

### Chatbot
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat/message` | Send message to chatbot |
| GET | `/api/v1/chat/history` | Get chat history |

### Information
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/info/diseases` | List all diseases |
| GET | `/api/v1/info/diseases/{name}` | Get disease info |
| GET | `/api/v1/info/treatments` | List treatments |
| GET | `/api/v1/info/hospitals` | Get hospital recommendations |

## 🤖 AI Model

### Supported Frameworks
- TensorFlow 2.x (default)
- PyTorch

### Architecture
- Transfer learning with EfficientNetB0 (TensorFlow) or ResNet18 (PyTorch)
- Input: 224x224 RGB images
- Output: 6 classes (Normal + 5 diseases)

### Training Your Own Model

1. Prepare dataset in the following structure:
   ```
   data/
   ├── train/
   │   ├── normal/
   │   ├── stroke/
   │   ├── epilepsy/
   │   ├── alzheimer/
   │   ├── parkinson/
   │   └── brain_tumor/
   └── test/
       └── (same structure)
   ```

2. Run training:
   ```bash
   python -m app.ai_models.train --data-dir ./data --epochs 50
   ```

## 🔒 Security

- JWT-based authentication with token refresh
- Bcrypt password hashing with salt
- Input validation and sanitization
- CORS protection
- Rate limiting on sensitive endpoints
- Secure file upload handling

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v
```

## 📦 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Recommendations

1. Use PostgreSQL instead of SQLite
2. Set `DEBUG=false`
3. Use strong, unique `SECRET_KEY`
4. Enable HTTPS
5. Use proper CORS origins
6. Set up monitoring and logging
7. Use a production WSGI server (uvicorn with gunicorn)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚠️ Disclaimer

This application is for educational and informational purposes only. The AI analysis should NOT be considered as a medical diagnosis. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.

## 📄 License

MIT License - See LICENSE file for details.

## 👥 Authors

Brain Disease AI Team

## 📞 Support

For questions or support, please open an issue on GitHub.
