# Server imports
from flask import Flask, render_template, request, redirect, jsonify
import requests
import json
from bs4 import BeautifulSoup
import re

# internal moduless
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
    # http://video.google.com/timedtext?lang=en&v=gHkELWFqGKQ

    youtube_url = request.json.get("videoID")

    vid_id_pattern = re.compile("v=.*")

    video_id = vid_id_pattern.search(youtube_url).group(0)

    print("youtube_url:", youtube_url)
    print("video_id:", video_id)



    video_text_url = f'http://video.google.com/timedtext?lang=en&{video_id}'

    print("video_text_url:", video_text_url)

    page = requests.get(video_text_url)

    #print(page.text)
    soup = BeautifulSoup(page.text, 'html.parser')
    # print("TEST_____________")
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
    to_return["main_ideas"] = extracted_main_ideas[2:]
    to_return["key_words"] = tf_idf(extracted_main_ideas, to_return["main_topics"])
    to_return["summary"] = extracted_main_ideas[0:2]
    print(to_return)
    # print(json.dumps(to_return))
    return str(json.dumps(to_return))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
