# Find the Sinch phone number assigned to your app
# and your application key and secret
# at dashboard.sinch.com/voice/apps
import requests

key = ""
secret = ""
fromNumber = ""
to = ""
locale = ""
url = "https://calling.api.sinch.com/calling/v1/callouts"

payload = {
  "method": "ttsCallout",
  "ttsCallout": {
    "cli": fromNumber,
    "destination": {
      "type": "number",
      "endpoint": to
    },
    "locale": locale,
    "text": "Hello, this is a call from Sinch. Congratulations! You made your first call."
  }
}

headers = { "Content-Type": "application/json" }

response = requests.post(url, json=payload, headers=headers, auth=(key, secret))

data = response.json()
print(data)