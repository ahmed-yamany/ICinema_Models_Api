from fastapi import APIRouter
from Models.Recommender.Cosine import MoviesRecommendationSystem
recommendation_system_router = APIRouter(prefix="/recommendation_system", tags=["Recommendation system"])



@recommendation_system_router.post("/recommend")
def recommendation_system(number: int, lang: str, user: dict, movies: list) -> list:

    return MoviesRecommendationSystem(number, user, movies, lang).get_recommended_movies()


