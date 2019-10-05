# Server imports
from flask import Flask, render_template, request, redirect, jsonify
import requests
import json
from bs4 import BeautifulSoup

from topic_modeling import main_topics_small_corpus, main_topics_large_corpus
from main_idea_extraction import main_ideas

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
    print(main_topics_small_corpus(sentences,2,5))
    print("TEST*******************")
    print(main_topics_large_corpus(sentences,2,5))
    print("TEST2*******************")
    print(main_ideas(sentences))
    return str(json.dumps(sentences))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
