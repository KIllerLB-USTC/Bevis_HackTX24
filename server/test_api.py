import requests

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