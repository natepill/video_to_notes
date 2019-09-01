import speech_recognition as sr

# r = sr.Recognizer()
#
# PATH = 'example_audio.wav'
#
# with sr.AudioFile(PATH) as source:
#     try:
#         audio = r.listen(source)
#         with open('example.txt', 'w') as file:
#             file.write(r.recognize_google(audio))
#
#     except Exception as e:
#         print(e)



r = sr.Recognizer()

PATH = 'example_audio.wav'

with sr.AudioFile(PATH) as source:
    try:
        audio = r.record(source)
        with open('example.txt', 'w') as file:
            file.write(r.recognize_google(audio))

    except Exception as e:
        print(e)
