from fastapi import FastAPI

app = FastAPI()


@app.get("/items/")
async def read_items(response_model: str = "default"):
    return {"ok": True}
