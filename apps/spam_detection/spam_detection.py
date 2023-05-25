from fastapi import APIRouter
from Models.spam.detection import SpamDetector
spam_detection_router = APIRouter(prefix="/spam", tags=["Spam Detection"])



@spam_detection_router.post("/detect")
def detect_spam(text: str, lang: str = "en") -> dict:
    spam_detector = SpamDetector("CONTENT", "TYPE", lang=lang)
    spam_detector.fit_model()
    return {"spam": spam_detector.is_spam(text)}


