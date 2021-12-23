from flask import Flask
from flask_cors import CORS
from utils import GetResponse
from flask import request

app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": ["http://localhost:3051/","http://18.118.116.85:3051/"]}})


@app.route('/', methods = ['POST'])
def get_response():
    try:
        obj = GetResponse(request.form["message"])
        res = obj.chatbot_response()
        return {"message": res, "error": False}
    except Exception as e:
        return {"message": "Something went wrong", "error": str(e)}


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)