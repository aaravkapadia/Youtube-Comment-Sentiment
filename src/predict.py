from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch
from googleapiclient.discovery import build
from collections import Counter
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')
model = RobertaForSequenceClassification.from_pretrained('/Users/aaravkapadia/finetune/model')
tokenizer = AutoTokenizer.from_pretrained('/Users/aaravkapadia/finetune/model')
model.eval()
id_to_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def predict(comment):
    tokens = tokenizer(
        comment, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    with torch.no_grad():
        logits = model(**tokens).logits
    pred = torch.argmax(logits, dim=-1).item()
    return id_to_label[pred]

def get_comments(video, api_key, max_comments):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video,
        maxResults=max_comments,
        order='relevance'
    )
    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response['items']:
            comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
        request = youtube.commentThreads().list_next(request, response)
    return comments

def analyze(video, api_key, max_comments):
    comments = get_comments(video, api_key, max_comments)
    sentiment = [predict(comment) for comment in comments]
    sentimentCounter = Counter(sentiment)
    total = len(sentiment)
    print(f"Sample Size = {total}")
    print(f"This video had {(sentimentCounter['Positive']/total)*100:.1f}% positive commments")
    print(f"This video had {(sentimentCounter['Negative']/total)*100:.1f}% negative commments")
    print(f"This video had {(sentimentCounter['Neutral']/total)*100:.1f}% neutral commments")
    

analyze('0e3GPea1Tyg', api_key, 100)

