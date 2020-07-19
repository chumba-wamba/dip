from fastapi import FastAPI, Path, Query, Body
from typing import Optional

app = FastAPI()


@app.get("/root")
@app.get("/index")
@app.get("/")
def root():
    return {"message": "welcome to the DIP API"}
