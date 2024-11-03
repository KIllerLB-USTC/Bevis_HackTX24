import requests
from openai import OpenAI
from playsound import playsound

def text_tospeech(text):
    Apikey="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"
    client = OpenAI(api_key= Apikey)
    speech_file_path = "./speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text)
    response.stream_to_file(speech_file_path)

# URL of your FastAPI endpoint
url = "http://localhost:8000/api/analyze"

# Open the image file in binary mode using a context manager
with open("test_data/sample_chart.jpg", "rb") as image_file:
    files = {
        "file": ("image.jpg", image_file, "image/jpeg")
    }
    # Make the POST request
    response = requests.post(url, files=files)

# Print the response
print(response.json())

text = response.json()['text']
text_tospeech(text)
playsound('./speech.mp3')


