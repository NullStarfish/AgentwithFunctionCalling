from flask import Flask, request, redirect, url_for, send_from_directory
import os
from api import APIClient
from agent import ImageAnalysisAgent

app = Flask(__name__)
api_client = APIClient()
image_agent = ImageAnalysisAgent(api_client)

baseurl = "http://47.97.8.27"
imageurl = baseurl + "/image/"

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file:
        filename = file.filename
        file.save(os.path.join(os.getcwd(), "pictures", filename))
        result = image_agent.process(imageurl + filename)
        return result
    return {'status': 'error', 'message': 'No file uploaded'}

@app.route('/getmessage')
def getmessage():
    message = api.sendPicture(imageurl + "111.png")
    return message

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory('pictures', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

