from pydantic import BaseModel, Field


class TodoCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    done: bool = False


class TodoUpdate(BaseModel):
    title: str | None = None
    done: bool | None = None

class UserCreat(BaseModel):
    name: str = Field(min_length=2, max_length=20)
    age: int = Field(ge=0, le=150)
    email: str| None = None