"""
Brain Disease AI - Chatbot Engine
Rule-based and NLP-powered chatbot for medical queries
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import random

from app.config import DISEASES, TREATMENTS

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Represents a detected user intent"""
    name: str
    confidence: float
    entities: Dict[str, Any]


# Intent patterns (regex-based)
INTENT_PATTERNS = {
    "greeting": [
        r"\b(hi|hello|hey|greetings|good\s*(morning|afternoon|evening))\b",
        r"^(hi|hello|hey)\s*$"
    ],
    "farewell": [
        r"\b(bye|goodbye|see\s*you|take\s*care|farewell)\b"
    ],
    "help": [
        r"\b(help|assist|support|how\s*(do|can)\s*i|what\s*can\s*you)\b",
        r"\b(how\s*to\s*use|guide|tutorial)\b"
    ],
    "symptoms": [
        r"\b(symptom|sign|indication|what\s*are\s*the\s*symptoms)\b",
        r"\b(feel|feeling|experiencing)\s*(headache|dizzy|numb|weak|confused)\b"
    ],
    "disease_info": [
        r"\b(what\s*is|tell\s*me\s*about|explain|information\s*about)\s*(stroke|epilepsy|alzheimer|parkinson|brain\s*tumor)\b",
        r"\b(stroke|epilepsy|alzheimer|parkinson|brain\s*tumor)\s*(information|details|facts)\b"
    ],
    "treatment": [
        r"\b(treatment|cure|medicine|medication|therapy|how\s*to\s*treat)\b",
        r"\b(can\s*it\s*be\s*(cured|treated)|is\s*there\s*a\s*cure)\b"
    ],
    "precaution": [
        r"\b(precaution|prevent|avoid|safety|risk\s*factor)\b",
        r"\b(how\s*to\s*(prevent|avoid)|what\s*to\s*avoid)\b"
    ],
    "scan_upload": [
        r"\b(upload|scan|mri|ct\s*scan|image|analyze)\b",
        r"\b(how\s*to\s*upload|submit\s*scan)\b"
    ],
    "results": [
        r"\b(result|prediction|diagnosis|analysis|report)\b",
        r"\b(my\s*results|scan\s*results|what\s*does.*mean)\b"
    ],
    "disclaimer": [
        r"\b(disclaimer|accuracy|reliable|trust|can\s*i\s*trust)\b",
        r"\b(is\s*this\s*accurate|how\s*accurate)\b"
    ],
    "emergency": [
        r"\b(emergency|urgent|immediate|serious|critical)\b",
        r"\b(should\s*i\s*(go\s*to|visit)\s*(hospital|doctor|er))\b"
    ],
    "hospital": [
        r"\b(hospital|doctor|specialist|clinic|where\s*to\s*go)\b",
        r"\b(recommend.*hospital|find.*doctor)\b"
    ],
    "thanks": [
        r"\b(thank|thanks|appreciate|grateful)\b"
    ]
}

# Disease keyword patterns
DISEASE_KEYWORDS = {
    "stroke": ["stroke", "paralysis", "brain attack", "cerebrovascular"],
    "epilepsy": ["epilepsy", "seizure", "convulsion", "fits"],
    "alzheimer": ["alzheimer", "dementia", "memory loss", "forgetfulness"],
    "parkinson": ["parkinson", "tremor", "shaking", "movement disorder"],
    "brain_tumor": ["brain tumor", "tumor", "cancer", "growth", "mass"]
}


class ChatbotEngine:
    """Rule-based chatbot engine with NLP capabilities"""
    
    def __init__(self):
        self.context: Dict[str, Any] = {}
        self.conversation_history: List[Dict] = []
        
    def detect_intent(self, message: str) -> Intent:
        """
        Detect user intent from message.
        
        Args:
            message: User message
            
        Returns:
            Detected Intent with confidence
        """
        message_lower = message.lower().strip()
        
        best_intent = "unknown"
        best_confidence = 0.0
        entities = {}
        
        # Check each intent pattern
        for intent_name, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower, re.IGNORECASE)
                if match:
                    # Calculate confidence based on match quality
                    match_ratio = len(match.group()) / len(message_lower)
                    confidence = min(0.5 + match_ratio * 0.5, 0.95)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent_name
        
        # Extract disease entities
        for disease, keywords in DISEASE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    entities["disease"] = disease
                    break
        
        # Default confidence for unknown intent
        if best_intent == "unknown":
            best_confidence = 0.3
        
        return Intent(name=best_intent, confidence=best_confidence, entities=entities)
    
    def generate_response(self, intent: Intent, message: str) -> Dict[str, Any]:
        """
        Generate response based on detected intent.
        
        Args:
            intent: Detected intent
            message: Original user message
            
        Returns:
            Response dictionary with message, suggestions, etc.
        """
        response = {
            "message": "",
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": []
        }
        
        # Get disease from entities if available
        disease = intent.entities.get("disease")
        
        # Route to appropriate handler
        handlers = {
            "greeting": self._handle_greeting,
            "farewell": self._handle_farewell,
            "help": self._handle_help,
            "symptoms": self._handle_symptoms,
            "disease_info": self._handle_disease_info,
            "treatment": self._handle_treatment,
            "precaution": self._handle_precaution,
            "scan_upload": self._handle_scan_upload,
            "results": self._handle_results,
            "disclaimer": self._handle_disclaimer,
            "emergency": self._handle_emergency,
            "hospital": self._handle_hospital,
            "thanks": self._handle_thanks,
            "unknown": self._handle_unknown
        }
        
        handler = handlers.get(intent.name, self._handle_unknown)
        return handler(intent, disease)
    
    def _handle_greeting(self, intent: Intent, disease: Optional[str]) -> Dict:
        greetings = [
            "Hello! I'm your Brain Health Assistant. How can I help you today?",
            "Hi there! I'm here to help you with brain health information. What would you like to know?",
            "Welcome! I can help you with brain disease information, symptoms, and more. What's on your mind?"
        ]
        return {
            "message": random.choice(greetings),
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Tell me about brain diseases",
                "What are stroke symptoms?",
                "How to upload a scan?"
            ]
        }
    
    def _handle_farewell(self, intent: Intent, disease: Optional[str]) -> Dict:
        farewells = [
            "Goodbye! Take care of your health. Feel free to return if you have more questions.",
            "See you! Remember, always consult a doctor for medical concerns. Stay healthy!",
            "Bye! Wishing you good health. Don't hesitate to come back if you need help."
        ]
        return {
            "message": random.choice(farewells),
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": []
        }
    
    def _handle_help(self, intent: Intent, disease: Optional[str]) -> Dict:
        help_message = """
I can help you with the following:

🧠 **Disease Information**: Learn about stroke, epilepsy, Alzheimer's, Parkinson's, and brain tumors.

🔬 **Symptom Checker**: Understand common symptoms of brain diseases.

💊 **Treatment Info**: Get information about treatments and medications.

🏥 **Hospital Finder**: Find recommended hospitals and specialists.

📤 **Scan Analysis**: Learn how to upload your brain scans for AI analysis.

⚠️ **Precautions**: Get prevention tips and lifestyle guidance.

Just ask me anything related to brain health!
        """
        return {
            "message": help_message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "What is Alzheimer's?",
                "Symptoms of stroke",
                "How to prevent brain diseases"
            ]
        }
    
    def _handle_symptoms(self, intent: Intent, disease: Optional[str]) -> Dict:
        if disease and disease in DISEASES:
            info = DISEASES[disease]
            symptoms_list = "\n".join([f"• {s}" for s in info["symptoms"]])
            message = f"""
**Symptoms of {info['name']}:**

{symptoms_list}

⚠️ **Important**: If you experience any of these symptoms, please consult a medical professional immediately.
            """
        else:
            message = """
Here are common symptoms of brain diseases:

**Stroke**: Sudden numbness, confusion, trouble speaking, vision problems, severe headache

**Epilepsy**: Seizures, temporary confusion, staring spells, uncontrollable jerking

**Alzheimer's**: Memory loss, confusion, difficulty thinking, personality changes

**Parkinson's**: Tremor, slowed movement, rigid muscles, impaired posture

**Brain Tumor**: Persistent headaches, seizures, vision problems, nausea

Which disease would you like to know more about?
            """
        
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Symptoms of stroke",
                "Symptoms of epilepsy",
                "When to see a doctor"
            ]
        }
    
    def _handle_disease_info(self, intent: Intent, disease: Optional[str]) -> Dict:
        if disease and disease in DISEASES:
            info = DISEASES[disease]
            symptoms = ", ".join(info["symptoms"][:3])
            risks = ", ".join(info["risk_factors"][:3])
            
            message = f"""
**{info['name']}**

📋 **Description**: {info['description']}

🔍 **Key Symptoms**: {symptoms}

⚠️ **Risk Factors**: {risks}

Would you like to know about treatments or precautions for {info['name']}?
            """
            suggestions = [
                f"Treatment for {info['name']}",
                f"Precautions for {info['name']}",
                "Upload a scan"
            ]
        else:
            message = """
I can provide information about these brain diseases:

1. **Stroke** - A brain attack caused by interrupted blood supply
2. **Epilepsy** - Neurological disorder with recurrent seizures
3. **Alzheimer's** - Progressive memory and cognitive decline
4. **Parkinson's** - Movement disorder causing tremors and stiffness
5. **Brain Tumor** - Abnormal cell growth in the brain

Which one would you like to learn about?
            """
            suggestions = [
                "Tell me about stroke",
                "What is Alzheimer's?",
                "Parkinson's disease info"
            ]
        
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": suggestions
        }
    
    def _handle_treatment(self, intent: Intent, disease: Optional[str]) -> Dict:
        if disease and disease in TREATMENTS:
            info = TREATMENTS[disease]
            disease_name = DISEASES[disease]["name"]
            
            meds = ", ".join(info["medications"][:3])
            procedures = ", ".join(info["procedures"][:2])
            lifestyle = ", ".join(info["lifestyle"][:3])
            specialists = ", ".join(info["specialists"][:2])
            
            message = f"""
**Treatment Options for {disease_name}**

💊 **Medications**: {meds}

🏥 **Procedures**: {procedures}

🌿 **Lifestyle Changes**: {lifestyle}

👨‍⚕️ **Specialists to Consult**: {specialists}

⚠️ **Disclaimer**: Treatment should be determined by qualified healthcare professionals based on individual assessment. This information is for educational purposes only.
            """
        else:
            message = """
Treatment varies by disease. Please specify which condition you'd like treatment information for:

• Stroke
• Epilepsy
• Alzheimer's Disease
• Parkinson's Disease
• Brain Tumor

Each requires different approaches and should be managed by medical professionals.
            """
        
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Treatment for stroke",
                "Alzheimer's medication",
                "Find a specialist"
            ]
        }
    
    def _handle_precaution(self, intent: Intent, disease: Optional[str]) -> Dict:
        general_precautions = """
**General Brain Health Precautions:**

🏃 **Exercise Regularly**: At least 30 minutes of moderate activity daily

🥗 **Healthy Diet**: Mediterranean diet, rich in fruits, vegetables, and omega-3

😴 **Adequate Sleep**: 7-9 hours of quality sleep each night

🧩 **Mental Stimulation**: Puzzles, reading, learning new skills

🚭 **Avoid Smoking**: Significantly increases stroke risk

🍷 **Limit Alcohol**: Excessive drinking damages brain cells

💊 **Manage Health Conditions**: Control blood pressure, diabetes, cholesterol

😌 **Stress Management**: Practice relaxation techniques

👥 **Social Connections**: Stay socially active and engaged
        """
        
        if disease and disease in DISEASES:
            risks = DISEASES[disease].get("risk_factors", [])
            risks_text = "\n".join([f"• Avoid/Manage: {r}" for r in risks[:4]])
            
            message = f"""
**Precautions for {DISEASES[disease]['name']}:**

{risks_text}

{general_precautions}
            """
        else:
            message = general_precautions
        
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Risk factors for stroke",
                "How to prevent Alzheimer's",
                "Brain-healthy diet"
            ]
        }
    
    def _handle_scan_upload(self, intent: Intent, disease: Optional[str]) -> Dict:
        message = """
**How to Upload Your Brain Scan:**

1️⃣ **Log in** to your account

2️⃣ Go to **Dashboard** → **Upload Scan**

3️⃣ **Select your scan file** (MRI, CT, or PET scan)
   - Supported formats: JPEG, PNG, DICOM, NIfTI
   - Maximum size: 10MB

4️⃣ Choose **scan type** (MRI/CT/PET)

5️⃣ Click **Upload & Analyze**

6️⃣ Wait 1-2 minutes for AI analysis

7️⃣ View your **results and report**

⚠️ **Important**: Our AI provides screening assistance only. Always consult a doctor for proper diagnosis.
        """
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "What file formats are supported?",
                "How accurate is the AI?",
                "View my results"
            ]
        }
    
    def _handle_results(self, intent: Intent, disease: Optional[str]) -> Dict:
        message = """
**Understanding Your Scan Results:**

📊 **Prediction**: The AI identifies the most likely condition based on scan patterns.

📈 **Confidence Score**: Shows how certain the AI is (higher = more confident).

📋 **Report**: Detailed findings with treatment suggestions.

**Confidence Levels:**
- 🟢 **Very High (90%+)**: Strong pattern match
- 🟡 **High (75-89%)**: Good confidence
- 🟠 **Moderate (50-74%)**: Consider additional tests
- 🔴 **Low (<50%)**: Inconclusive, consult doctor

⚠️ **Remember**: AI results are for informational purposes. A qualified healthcare professional should make the final diagnosis.

Would you like to upload a scan or view your history?
        """
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Upload a scan",
                "View my scan history",
                "What does my result mean?"
            ]
        }
    
    def _handle_disclaimer(self, intent: Intent, disease: Optional[str]) -> Dict:
        message = """
**⚠️ Medical Disclaimer**

This AI system is designed for **informational and educational purposes only**.

❌ It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.

✅ **What our AI does:**
- Analyzes brain scan patterns
- Provides preliminary screening
- Offers educational information

❌ **What our AI does NOT do:**
- Provide medical diagnosis
- Replace doctor consultations
- Guarantee 100% accuracy

**Our Accuracy**: While trained on medical data, AI can make errors. Always verify with healthcare professionals.

**Recommendation**: Use this as a screening tool, then consult a qualified neurologist or physician for confirmation.

🏥 **If you have symptoms, please see a doctor immediately.**
        """
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Find a hospital",
                "Upload a scan",
                "Learn about symptoms"
            ]
        }
    
    def _handle_emergency(self, intent: Intent, disease: Optional[str]) -> Dict:
        message = """
🚨 **EMERGENCY ALERT**

If you or someone else is experiencing:
- Sudden severe headache
- Sudden numbness or weakness
- Difficulty speaking or understanding
- Sudden vision problems
- Severe confusion
- Seizures
- Loss of consciousness

**CALL EMERGENCY SERVICES IMMEDIATELY!**

🇮🇳 India: **112** or **108** (Ambulance)
🇺🇸 USA: **911**
🇬🇧 UK: **999**
🌍 International: Contact your local emergency number

⏱️ **Time is critical for brain emergencies!**

DO NOT wait for AI analysis in emergencies. Seek immediate medical attention!
        """
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Find nearest hospital",
                "Stroke symptoms",
                "First aid tips"
            ]
        }
    
    def _handle_hospital(self, intent: Intent, disease: Optional[str]) -> Dict:
        message = """
**🏥 Recommended Hospitals & Specialists**

**Top Neurology Centers:**

1. **Apollo Hospitals**
   📍 Multiple locations across India
   📞 +91-1860-500-1066

2. **AIIMS Delhi**
   📍 New Delhi, India
   📞 +91-11-26588500

3. **Fortis Healthcare**
   📍 Multiple locations
   📞 +91-8010-994-994

4. **Max Healthcare**
   📍 Delhi NCR
   📞 +91-11-2651-5050

5. **Manipal Hospitals**
   📍 Bangalore & other cities
   📞 +91-80-2502-4444

**Specialists to Consult:**
- Neurologist
- Neurosurgeon
- Neuro-oncologist (for tumors)
- Movement disorder specialist (for Parkinson's)
- Epileptologist (for seizures)

💡 **Tip**: Bring your scan reports and history when visiting.
        """
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "What to bring to appointment",
                "Questions to ask doctor",
                "Get directions"
            ]
        }
    
    def _handle_thanks(self, intent: Intent, disease: Optional[str]) -> Dict:
        responses = [
            "You're welcome! Feel free to ask if you have more questions. Stay healthy! 😊",
            "Happy to help! Take care of your brain health. I'm here if you need anything else.",
            "My pleasure! Remember, early detection saves lives. Don't hesitate to reach out anytime."
        ]
        return {
            "message": random.choice(responses),
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Learn about brain health",
                "Upload a scan",
                "Goodbye"
            ]
        }
    
    def _handle_unknown(self, intent: Intent, disease: Optional[str]) -> Dict:
        message = """
I'm not sure I understood that. I'm a brain health assistant and can help with:

🧠 Brain disease information (Stroke, Epilepsy, Alzheimer's, Parkinson's, Brain Tumor)
🔬 Symptoms and early signs
💊 Treatment options
🏥 Hospital recommendations
📤 Scan upload guidance
⚠️ Prevention tips

Could you rephrase your question? Or choose from the suggestions below.
        """
        return {
            "message": message,
            "intent": intent.name,
            "confidence": intent.confidence,
            "suggestions": [
                "Help me understand",
                "What can you do?",
                "Brain disease info"
            ]
        }
    
    def chat(self, message: str) -> Dict[str, Any]:
        """
        Main chat interface.
        
        Args:
            message: User input message
            
        Returns:
            Chatbot response dictionary
        """
        # Detect intent
        intent = self.detect_intent(message)
        
        # Generate response
        response = self.generate_response(intent, message)
        
        # Store in history
        self.conversation_history.append({
            "user": message,
            "bot": response["message"],
            "intent": intent.name
        })
        
        return response
