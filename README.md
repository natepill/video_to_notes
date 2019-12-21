# Video2Notes

Visit {INSERT LIVE LINK} and enter the YouTube url for the video that you would like to generate notes for. The given YouTube video __must have English captions__ in order for our application to obtain the video's text. The generated notes are formatted in the style of the Cornell Note taking system {INSERT REFERENCE TO A SOURCE}. Once the text from the given YouTube video is obtained, we generate the notes into 4 sections per the Cornell Notes styling guide:
* Key Vocabulary
* Main Topics
* Main Ideas
* Text Summarization

This application is meant a educational tool to supplement student studies or event reviews, but should not be used in place of writing your own notes or other studying methods.

## Methodology:

Obtain text from videos: Web scrape: http://video.google.com/timedtext?lang=en&{VIDEO_ID}
Main Ideas: Graph based TextRank algorithm on all sentences from the corpus
Key Words: TF-IDF scoring on extracted Main Ideas
Main Topics: Non-negative Matrix Factorization (NMF) and LatentDirichletAllocation (LDA)
Text Summarization: Encoder/Decoder w/ LSTM and AttentionLayer (Work in progress). Currently is top 2 ranked sentences from generated Main Ideas.


## Issues and Blockers:
* Text Summarization: Having issues finding appropriately sized outputs in Text Summarization datasets. The current dataset that our model is trained on is only generating sentences of max length 8 words since the model is trained on summarized reviews from the Amazon Reviews dataset. Need to find better dataset to train model on or different summarization approach.
* Video Transcripts: We only have access to the transcripts of videos that already have generated English captions so an issue is acquiring transcripts for any YouTube video. The YouTube API only allows you to extract transcripts from videos that you're the owner of. We tried scraping the auto-generated captions that YouTube creates for each YouTube video, but it seems as though that content is protected.



## Next Steps

* Video Transcripts: In an effort to make gain accessibility to the transcripts of more videos, we plan on making requests to the Google Cloud API for audio transcription. The user will continue to provide just the link of the YouTube video they would like notes for, but instead of web scraping for videos that already have existing captions, we would download the YouTube video from the given url to a .WAV or .FLAK audio format and send that file along a request to the Google Cloud API for transcription. Considerations may include:
- Most videos that will be requested to be transcribed will require a long form request that opens up a data stream to the API https://cloud.google.com/speech-to-text/docs/async-recognize
- Wait time for transcription will be dependent on the length of the video given, while web scraping for transcripts is nearly instant. It would be beneficial to first check if captions exist for a given YouTube video, if not, then try to transcribe it.
* Train text summarization model on better dataset with longer summaries for supervised learning, or find a better unsupervised approached for text summarization

* Add user accounts with the ability to save, organize, and share generated notes


## Frameworks used and external resources:
* Flask
* JS
* Keras
* sklearn
* networkx
* BeautifulSoup
