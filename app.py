# Server imports
from flask import Flask, render_template, request, redirect, jsonify
import requests
import json
from bs4 import BeautifulSoup
import pprint

from topic_modeling import main_topics_small_corpus, main_topics_large_corpus
from main_idea_extraction import main_ideas
from tf_idf_built_in import tf_idf

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

    # print(sentences)
    # print(main_ideas(sentences))

    # One string instead of array of sentences
    # sentences_to_string = ' '.join(sentences)

    ranked_main_ideas = main_ideas(sentences)
    extracted_main_ideas = []

    # Print Top 10 extracted_main_ideas
    for i in range(10):
        extracted_main_ideas.append(ranked_main_ideas[i][1])

    to_return = {}

    to_return["main_topics"] = main_topics_large_corpus(sentences,2,5)
    to_return["main_ideas"] = extracted_main_ideas
    to_return["key_words"] = tf_idf(extracted_main_ideas, to_return["main_topics"])
    to_return["summary"] = ["Lorem ipsum other random latin words we like to use"]
    print(to_return)
    # print(json.dumps(to_return))
    return str(json.dumps(to_return))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
