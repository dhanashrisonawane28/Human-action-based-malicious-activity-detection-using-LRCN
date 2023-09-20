import requests
import time
import  os

def telegram_send(vartopass1,vartoPass2):
    base_url = "https://api.telegram.org/bot6416572645:AAFQQhRiAosOOHZDFgH3H7hoWVyl5J7aE1Y/sendPhoto"

    image_path = "C:/Users/Vishal/PycharmProjects/AnamolyDetection/Src/selected_frame.jpg"
    while not os.path.exists(image_path):
        pass  # Wait until the file exists

    my_file = open(image_path, "rb")

    parameters = {
        "chat_id" : "998635769",
        "caption" : f"{vartopass1}_______{vartoPass2}"
    }


    files = {
        "photo" : my_file
    }

    resp = requests.get(base_url, data = parameters, files=files)
    print(resp.text)