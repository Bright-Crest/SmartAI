from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return {"message": "Hello, World from Flask!"}

@app.route('/health')
def health():
    return {"status": "healthy"}

if __name__ == '__main__':
    print("Starting Flask server on port 8005...")
    app.run(debug=True, port=8005) 