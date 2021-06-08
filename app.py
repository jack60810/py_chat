import os
import random
import json
import torch
from flask import Flask, abort, request
# https://github.com/line/line-bot-sdk-python
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import torch.nn as nn
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('/home/ubuntu/flask_app/intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "/home/ubuntu/flask_app/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def create_app():
    app = Flask(__name__)

    line_bot_api = LineBotApi(
        'pp/ehS+JQg2L1cG1NwOf/Z4e7SE4FT3jAUqiBjzTctD6ublLP7/hS2OP+oLTl+ITPU1KCGHY7X9fqH/IGbWPmbDYIT9c29xtYFOa6yMEv2zpaNgi28fM2Fujw6TvljQo0/L7Iiu9jKH8MOMqXljJqAdB04t89/1O/w1cDnyilFU=')
    handler = WebhookHandler("fa9519103155e682fe14ffcb46b9f28f")

    @app.route("/", methods=["GET", "POST"])
    def callback():

        if request.method == "GET":
            return "Hello GCP"
        if request.method == "POST":
            signature = request.headers["X-Line-Signature"]
            body = request.get_data(as_text=True)

            try:
                handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400)

            return "OK"

    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        get_message = event.message.text
        sentence = tokenize(get_message)
        mytext = ''
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    mytext = (f"{random.choice(intent['responses'])}")
            else:
                mytext = ("e04，我不懂")
    # Send To Line
        reply = TextSendMessage(text=f"{mytext}")
        line_bot_api.reply_message(event.reply_token, reply)
    return app

app = create_app()



