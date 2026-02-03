import requests
from .config_manager import load_config


class Slack :
    def __init__(self) :
        config = load_config()
        dictLogic = config['slack']
        
        self.token = dictLogic['token']
        self.channel = dictLogic['channel']

    def message(self, message):
        response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer " + self.token},
        data={"channel": self.channel,"text": message}
        )
        print(message)