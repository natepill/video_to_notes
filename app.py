# Server imports
from flask import Flask, render_template, request, redirect, jsonify
import requests
import json
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['GET','POST'])
def get_data_from_image():

    print(request.json)

    url = request.json.get("videoID") #"http://video.google.com/timedtext?lang=en&v=gHkELWFqGKQ"

    page =  requests.get(url)

    #print(page.text)
    soup = BeautifulSoup(page.text, 'html.parser')
    print("TEST_____________")
    sentences = []
    for item in soup.find_all('text'):
        sentences.append(item.get_text())

    print(json.dumps(str(sentences)))
    return str(json.dumps(sentences))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
