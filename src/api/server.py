"""Optional local FastAPI server."""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Fatigue Monitor Local")
templates = Jinja2Templates(directory="dashboard")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)