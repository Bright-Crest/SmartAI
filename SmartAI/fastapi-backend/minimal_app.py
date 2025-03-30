import sys
print(f"Python version: {sys.version}")

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="FastAPI Minimal App", docs_url="/docs")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(
        "minimal_app:app",
        host="127.0.0.1",
        port=8003,
        log_level="info"
    ) 