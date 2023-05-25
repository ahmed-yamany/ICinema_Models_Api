from fastapi import FastAPI
from apps.spam_detection.spam_detection import spam_detection_router
from apps.recommendation_system.recommendation_system import recommendation_system_router
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.include_router(spam_detection_router)
app.include_router(recommendation_system_router)

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=80, reload=True)
