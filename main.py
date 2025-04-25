from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str = None
    is_done: bool = False

items = []

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/items")
def create_item(item: Item):
    items.append(item)
    return items



@app.get("/items/{item_id}")
def get_item(item_id: int) -> Item:
    if(item_id >= len(items)):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        return items[item_id]
