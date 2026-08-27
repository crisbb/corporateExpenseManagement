from fastapi import APIRouter, HTTPException

from app.schemas import UserCreat
router = APIRouter(prefix="/users", tags=["users"])
users_db:list[dict] = []
_next_id = 1

@router.post('/users', status_code=201)
def create_user(user: UserCreat):
    global _next_id
    record = {'id': _next_id, **user.model_dump()}
    users_db.append(record)
    _next_id += 1
    return record

@router.get('/users')
def list_users():
    return users_db


@router.get("/users/{user_id}")
def get_user(user_id: int):
    for u in users_db:
        if u['id'] == user_id:
            return u
    raise HTTPException(status_code=404, detail='user not Fount')
