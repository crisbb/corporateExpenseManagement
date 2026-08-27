from fastapi import FastAPI

from app.routers.todos import router as todos_router
from app.routers.users import router as users_router


app = FastAPI(title="My TODO API")

app.include_router(todos_router)

app.include_router(users_router)

@app.get("/")
def hello():
    return {"message": "Hello from Python!"}
