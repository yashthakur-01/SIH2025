import google.generativeai as genai
from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import json
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns

# Import attractions data
from data.attractions_data import get_all_attractions, get_attractions_by_city, get_attractions_by_interest, get_attraction_by_id
from data.hotels_data import get_all_hotels, get_hotels_by_city, get_hotel_by_id

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection with fallback
try:
    mongo_url = os.environ.get('MONGO_URL')
    if mongo_url:
        print(f"[DEBUG] Attempting MongoDB connection to: {mongo_url[:30]}...")
        client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=10000)
        db = client[os.environ.get('DB_NAME', 'jharkhand_tourism')]
        print("[INFO] MongoDB client initialized successfully")
    else:
        raise Exception("MONGO_URL not found in environment variables")
except Exception as e:
    client = None
    db = None
    print(f"Warning: MongoDB connection failed: {str(e)}, using mock data")

# Create the main app without a prefix
app = FastAPI(title="Jharkhand Tourism Platform")

# Add root endpoint for health checks
@app.get("/")
async def health_check():
    return {"message": "Jharkhand Tourism Platform API is running", "status": "healthy"}

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize LLM Chat
# Gemini GenAI setup
def get_gemini_response(system_message: str, user_message: str):
    api_key = os.environ.get('GEMINI_API_KEY')
    print(f"[DEBUG] Using GEMINI_API_KEY: {api_key}")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"{system_message}\nUser: {user_message}"
    response = model.generate_content(prompt)
    print(f"[DEBUG] Gemini API response: {response}")
    return response.text if hasattr(response, 'text') else str(response)

# Pydantic Models
class VendorRegistration(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # "hotel", "artisan", "guide"
    location: str
    phone: str
    services: List[str]
    nearby_spots: List[str]
    pricing: Dict[str, Any] = Field(default_factory=dict)
    availability: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Booking(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tourist_name: str
    tourist_phone: str
    vendor_id: str
    vendor_type: str
    service_type: str
    booking_date: str
    message: str = ""
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Feedback(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_type: str  # "tourist", "vendor", "guide"
    vendor_id: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    comment: str
    location: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_message: str
    language: str = "english"  # "english" or "hindi"
    user_type: str = "tourist"  # "tourist", "vendor", "guide", "admin"

class TouristSpot(BaseModel):
    id: str
    name: str
    description: str
    coordinates: Dict[str, float]  # {"lat": float, "lng": float}
    type: str
    city: str
    interest_tags: List[str] = Field(default_factory=list)
    best_time: Optional[str] = None
    duration: Optional[str] = None
    image: Optional[str] = None

class Attraction(BaseModel):
    id: str
    name: str
    city: str
    coordinates: Dict[str, float]
    type: str
    interest_tags: List[str]
    description: str
    best_time: str
    duration: str
    image: str

class Hotel(BaseModel):
    id: str
    name: str
    city: str
    coordinates: Dict[str, float]
    rooms: int
    contact: str
    amenities: List[str]
    price_range: str

# Helper function for sentiment analysis
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Simple sentiment analysis using keywords"""
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "perfect", "beautiful", "nice"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disappointing", "poor", "pathetic"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "positive"
        score = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
    elif negative_count > positive_count:
        sentiment = "negative" 
        score = max(0.2, 0.5 - (negative_count - positive_count) * 0.1)
    else:
        sentiment = "neutral"
        score = 0.5
    
    return {"sentiment": sentiment, "score": score}

# Mock data for fallback
MOCK_VENDORS = [
    {
        "id": "vendor1",
        "name": "Ranchi Heritage Hotel",
        "type": "hotel",
        "location": "Ranchi",
        "phone": "+91-9876543210",
        "services": ["Accommodation", "Food", "Travel"],
        "nearby_spots": ["Rock Garden", "Tagore Hill"],
        "pricing": {"per_night": 2500},
        "availability": ["2025-09-22", "2025-09-23"]
    },
    {
        "id": "vendor2",
        "name": "Jharkhand Culture Guide",
        "type": "guide",
        "location": "Ranchi",
        "phone": "+91-8765432109",
        "services": ["Cultural Tours", "Local Sightseeing"],
        "nearby_spots": ["Hundru Falls", "Jonha Falls"],
        "pricing": {"per_day": 1500},
        "availability": ["2025-09-22"]
    }
]

# Routes
@api_router.get("/")
async def root():
    return {"message": "Jharkhand Tourism Platform API", "version": "1.0"}



# Attractions/Tourist Spots endpoints
@api_router.get("/attractions", response_model=List[Attraction])
async def get_attractions(city: Optional[str] = None, interest: Optional[str] = None):
    """Get all attractions with optional filtering"""
    if city:
        attractions = get_attractions_by_city(city)
    elif interest:
        attractions = get_attractions_by_interest(interest)
    else:
        attractions = get_all_attractions()
    
    return [Attraction(**attraction) for attraction in attractions]

@api_router.get("/attractions/{attraction_id}", response_model=Attraction)
async def get_attraction(attraction_id: str):
    """Get specific attraction by ID"""
    attraction = get_attraction_by_id(attraction_id)
    if not attraction:
        raise HTTPException(status_code=404, detail="Attraction not found")
    return Attraction(**attraction)

# Legacy tourist spots endpoint for backward compatibility
@api_router.get("/spots", response_model=List[TouristSpot])
async def get_tourist_spots():
    """Get tourist spots (legacy endpoint)"""
    attractions = get_all_attractions()
    spots = []
    for attraction in attractions:
        spot = TouristSpot(
            id=attraction["id"],
            name=attraction["name"],
            description=attraction["description"],
            coordinates=attraction["coordinates"],
            type=attraction["type"],
            city=attraction["city"],
            interest_tags=attraction.get("interest_tags", []),
            best_time=attraction.get("best_time"),
            duration=attraction.get("duration"),
            image=attraction.get("image")
        )
        spots.append(spot)
    return spots

@api_router.get("/hotels", response_model=List[Hotel])
async def get_hotels(city: Optional[str] = None):
    """Get all hotels or filter by city"""
    if city:
        hotels = get_hotels_by_city(city)
    else:
        hotels = get_all_hotels()
    return [Hotel(**hotel) for hotel in hotels]

@api_router.get("/hotels/{hotel_id}", response_model=Hotel)
async def get_hotel(hotel_id: str):
    """Get specific hotel by ID"""
    hotel = get_hotel_by_id(hotel_id)
    if not hotel:
        raise HTTPException(status_code=404, detail="Hotel not found")
    return Hotel(**hotel)

# Vendor Management
@api_router.post("/vendors", response_model=VendorRegistration)
async def register_vendor(vendor: VendorRegistration):
    vendor_dict = vendor.dict()
    vendor_dict['created_at'] = vendor_dict['created_at'].isoformat()
    if db is not None:
        try:
            await db.vendors.insert_one(vendor_dict)
        except:
            pass  # Continue even if database save fails
    return vendor

@api_router.get("/vendors", response_model=List[VendorRegistration])
async def get_vendors(vendor_type: Optional[str] = None, location: Optional[str] = None):
    if db is None:
        # Use mock data when MongoDB is unavailable
        vendors = MOCK_VENDORS.copy()
        if vendor_type:
            vendors = [v for v in vendors if v['type'] == vendor_type]
        if location:
            vendors = [v for v in vendors if location.lower() in v['location'].lower()]
        
        # Add missing fields for VendorRegistration
        for vendor in vendors:
            if 'created_at' not in vendor:
                vendor['created_at'] = datetime.now(timezone.utc)
        
        return [VendorRegistration(**vendor) for vendor in vendors]
    
    try:
        filter_query = {}
        if vendor_type:
            filter_query['type'] = vendor_type
        if location:
            filter_query['location'] = {"$regex": location, "$options": "i"}
        
        vendors = await db.vendors.find(filter_query).to_list(1000)
        for vendor in vendors:
            if isinstance(vendor.get('created_at'), str):
                vendor['created_at'] = datetime.fromisoformat(vendor['created_at'])
        return [VendorRegistration(**vendor) for vendor in vendors]
    except Exception as e:
        # Fallback to mock data on database error
        vendors = MOCK_VENDORS.copy()
        for vendor in vendors:
            if 'created_at' not in vendor:
                vendor['created_at'] = datetime.now(timezone.utc)
        return [VendorRegistration(**vendor) for vendor in vendors]

@api_router.get("/vendors/{vendor_id}")
async def get_vendor(vendor_id: str):
    if db is not None:
        try:
            vendor = await db.vendors.find_one({"id": vendor_id})
            if vendor:
                if isinstance(vendor.get('created_at'), str):
                    vendor['created_at'] = datetime.fromisoformat(vendor['created_at'])
                return VendorRegistration(**vendor)
        except:
            pass
    
    # Fallback to mock data
    for vendor in MOCK_VENDORS:
        if vendor['id'] == vendor_id:
            if 'created_at' not in vendor:
                vendor['created_at'] = datetime.now(timezone.utc)
            return VendorRegistration(**vendor)
    
    raise HTTPException(status_code=404, detail="Vendor not found")

# Booking Management
@api_router.post("/bookings", response_model=Booking)
async def create_booking(booking: Booking):
    booking_dict = booking.dict()
    booking_dict['created_at'] = booking_dict['created_at'].isoformat()
    if db is not None:
        try:
            await db.bookings.insert_one(booking_dict)
        except:
            pass  # Continue even if database save fails
    return booking

@api_router.get("/bookings", response_model=List[Booking])
async def get_bookings(vendor_id: Optional[str] = None, status: Optional[str] = None):
    if db is None:
        return []  # Return empty list when no database
    
    try:
        filter_query = {}
        if vendor_id:
            filter_query['vendor_id'] = vendor_id
        if status:
            filter_query['status'] = status
        
        bookings = await db.bookings.find(filter_query).to_list(1000)
        for booking in bookings:
            if isinstance(booking.get('created_at'), str):
                booking['created_at'] = datetime.fromisoformat(booking['created_at'])
        return [Booking(**booking) for booking in bookings]
    except:
        return []  # Return empty list on error

# Feedback Management
@api_router.post("/feedback", response_model=Feedback)
async def submit_feedback(feedback: Feedback):
    feedback_dict = feedback.dict()
    feedback_dict['created_at'] = feedback_dict['created_at'].isoformat()
    if db is not None:
        try:
            await db.feedback.insert_one(feedback_dict)
        except:
            pass  # Continue even if database save fails
    return feedback

@api_router.get("/feedback", response_model=List[Feedback])
async def get_feedback(vendor_id: Optional[str] = None):
    if db is None:
        return []  # Return empty list when no database
    
    try:
        filter_query = {}
        if vendor_id:
            filter_query['vendor_id'] = vendor_id
        
        feedback_list = await db.feedback.find(filter_query).to_list(1000)
        for feedback in feedback_list:
            if isinstance(feedback.get('created_at'), str):
                feedback['created_at'] = datetime.fromisoformat(feedback['created_at'])
        return [Feedback(**feedback) for feedback in feedback_list]
    except:
        return []  # Return empty list on error

# Contact Information
@api_router.get("/contact/{vendor_id}")
async def get_contact(vendor_id: str):
    if db is not None:
        try:
            vendor = await db.vendors.find_one({"id": vendor_id})
            if vendor:
                return {"phone": vendor.get('phone'), "name": vendor.get('name')}
        except:
            pass
    
    # Fallback to mock data
    for vendor in MOCK_VENDORS:
        if vendor['id'] == vendor_id:
            return {"phone": vendor.get('phone'), "name": vendor.get('name')}
    
    raise HTTPException(status_code=404, detail="Vendor not found")

# Multilingual Chatbot
@api_router.post("/chat")
async def chat_with_bot(message: ChatMessage):
    def get_contextual_fallback(user_message: str, language: str):
        """Generate contextual fallback responses based on user input"""
        user_lower = user_message.lower()
        
        # Tourism-related responses
        if any(word in user_lower for word in ['place', 'visit', 'tourist', 'attraction', 'spot', 'go', 'see']):
            if language == "hindi":
                return "झारखंड में कई खूबसूरत जगहें हैं! 🏔️\n\n🌊 हुंद्रू फॉल्स - राज्य का सबसे प्रसिद्ध झरना\n🦁 बेतला राष्ट्रीय उद्यान - वन्यजीव सफारी के लिए\n🏛️ रांची - राजधानी शहर\n⛰️ नेतरहाट - झारखंड का दिल\n\nआप किस जगह के बारे में और जानना चाहते हैं?"
            return "Jharkhand has many beautiful places to visit! 🏔️\n\n🌊 Hundru Falls - The state's most famous waterfall\n🦁 Betla National Park - For wildlife safari\n🏛️ Ranchi - The capital city\n⛰️ Netarhat - The heart of Jharkhand\n\nWhich place would you like to know more about?"
        
        elif any(word in user_lower for word in ['food', 'eat', 'cuisine', 'dish']):
            if language == "hindi":
                return "झारखंड का स्थानीय भोजन बहुत स्वादिष्ट है! 🍽️\n\n🥘 लिट्टी चोखा\n🍖 मटन करी\n🌾 चावल और दाल\n🥯 दूधपूरी\n🍜 रागी रोटी\n\nकौन सा व्यंजन आपको दिलचस्प लगता है?"
            return "Jharkhand has delicious local cuisine! 🍽️\n\n🥘 Litti Chokha\n🍖 Mutton Curry\n🌾 Rice and Dal\n🥯 Dhudhpuri\n🍜 Ragi Roti\n\nWhich dish interests you?"
        
        elif any(word in user_lower for word in ['culture', 'festival', 'tradition', 'dance']):
            if language == "hindi":
                return "झारखंड की संस्कृति बहुत समृद्ध है! 🎭\n\n🎪 सरहुल - मुख्य त्योहार\n💃 छऊ नृत्य\n🎨 आदिवासी कला\n🏛️ पारंपरिक शिल्प\n🎵 लोक संगीत\n\nआप किस सांस्कृतिक पहलू के बारे में जानना चाहते हैं?"
            return "Jharkhand has a rich cultural heritage! 🎭\n\n🎪 Sarhul - Main festival\n💃 Chhau dance\n🎨 Tribal art\n🏛️ Traditional crafts\n🎵 Folk music\n\nWhich cultural aspect would you like to explore?"
        
        elif any(word in user_lower for word in ['hotel', 'stay', 'accommodation', 'book']):
            if language == "hindi":
                return "झारखंड में ठहरने के लिए कई विकल्प हैं! 🏨\n\n🏨 रांची हेरिटेज होटल - ₹2500/रात\n🏡 गेस्ट हाउस\n⛺ इको रिसॉर्ट्स\n🏕️ कैंपिंग साइट्स\n\nआप किस प्रकार का आवास पसंद करेंगे?"
            return "There are many accommodation options in Jharkhand! 🏨\n\n🏨 Ranchi Heritage Hotel - ₹2500/night\n🏡 Guest Houses\n⛺ Eco Resorts\n🏕️ Camping Sites\n\nWhat type of accommodation would you prefer?"
        
        # Default fallback
        if language == "hindi":
            return "नमस्ते! मैं आपका झारखंड पर्यटन सहायक हूँ। 🙏\n\nमैं आपकी मदद कर सकता हूँ:\n• पर्यटन स्थलों की जानकारी\n• स्थानीय भोजन\n• संस्कृति और त्योहार\n• आवास विकल्प\n\nआप क्या जानना चाहते हैं?"
        return "Hello! I'm your Jharkhand tourism assistant! 🙏\n\nI can help you with:\n• Tourist attractions\n• Local food\n• Culture & festivals\n• Accommodation options\n\nWhat would you like to know?"
    
    # Use contextual fallback response
    fallback_response = get_contextual_fallback(message.user_message, message.language)
    
    try:
        language_instruction = ""
        if message.language == "hindi":
            language_instruction = "Please respond in Hindi (Devanagari script). "
        system_message = f"{language_instruction}You are a helpful tourism assistant for Jharkhand state in India. Provide information about tourist spots, local culture, food, festivals, and travel tips. Be friendly and informative."
        response = get_gemini_response(system_message, message.user_message)
        # Try to save to database if available
        if db is not None:
            try:
                chat_record = {
                    "id": message.id,
                    "user_message": message.user_message,
                    "bot_response": response,
                    "language": message.language,
                    "user_type": message.user_type,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                await db.chat_history.insert_one(chat_record)
            except Exception as db_exc:
                print(f"[DB ERROR] {db_exc}")
        return {"response": response, "language": message.language}
    except Exception as e:
        print(f"[GEMINI ERROR] {e}")
        # Try to save to database with fallback response
        if db is not None:
            try:
                chat_record = {
                    "id": message.id,
                    "user_message": message.user_message,
                    "bot_response": fallback_response,
                    "language": message.language,
                    "user_type": message.user_type,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                await db.chat_history.insert_one(chat_record)
            except Exception as db_exc:
                print(f"[DB ERROR] {db_exc}")
        return {"response": fallback_response, "language": message.language}

# Emergency endpoints
class EmergencySOSRequest(BaseModel):
    location: Dict[str, float]
    message: str = "Emergency SOS"

@api_router.post("/emergency/sos")
async def emergency_sos(request: EmergencySOSRequest):
    emergency_record = {
        "id": str(uuid.uuid4()),
        "location": request.location,
        "message": request.message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active"
    }
    
    if db is not None:
        try:
            await db.emergency_alerts.insert_one(emergency_record)
        except:
            pass  # Continue even if database save fails
    
    # In a real implementation, this would trigger notifications to authorities
    return {"status": "SOS sent", "emergency_id": emergency_record["id"]}

# Include the router in the main app
app.include_router(api_router)

# Get CORS origins from environment variable
cors_origins = os.environ.get('CORS_ORIGINS', "*")
if cors_origins == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = cors_origins.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
