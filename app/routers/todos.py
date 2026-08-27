from fastapi import APIRouter, HTTPException

from app.schemas import TodoCreate, TodoUpdate

router = APIRouter(prefix="/todos", tags=["todos"])

_todos_db: list[dict] = []
_next_id = 1


def find_todo(todo_id: int) -> dict:
    for t in _todos_db:
        if t["id"] == todo_id:
            return t
    raise HTTPException(status_code=404, detail="todo not found")


@router.post("", status_code=201)
def create_todo(todo: TodoCreate):
    global _next_id
    record = {"id": _next_id, **todo.model_dump()}
    _todos_db.append(record)
    _next_id += 1
    return record


@router.get("")
def list_todos():
    return _todos_db


@router.get("/{todo_id}")
def get_todo(todo_id: int):
    return find_todo(todo_id)


@router.patch("/{todo_id}")
def update_todo(todo_id: int, updates: TodoUpdate):
    todo = find_todo(todo_id)
    todo.update(updates.model_dump(exclude_unset=True))
    return todo


@router.delete("/{todo_id}", status_code=204)
def delete_todo(todo_id: int):
    todo = find_todo(todo_id)
    _todos_db.remove(todo)
