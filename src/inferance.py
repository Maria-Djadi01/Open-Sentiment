from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("alger-ia/dziribert_sentiment")
model = BertForSequenceClassification.from_pretrained("alger-ia/dziribert_sentiment")


def text_split(text):
    return text.split('\n')



def sent_analy_daridja(text):
    dict = {
        'Sentiment' : ['positive', 'neutral', 'negative'],
        'Number of feedbacks' : [0, 0, 0]
    }
    
    sentiment_model = pipeline("text-classification", model="alger-ia/dziribert_sentiment")
    result = sentiment_model(text_split(text))
    
    for i in result:
        if i['label'] == 'positive':
            dict['Number of feedbacks'][0] += 1
        elif i['label'] == 'neutral':
            dict['Number of feedbacks'][1] += 1
        else:
            dict['Number of feedbacks'][2] += 1
    return dict

def sent_analy_not_daridja(text):
    dict = {
        'Sentiment' : ['positive', 'neutral', 'negative'],
        'Number of feedbacks' : [0, 0, 0]
    }
    
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ['positive', 'neutral', 'negative']
    result = classifier(text, candidate_labels)

    for i in result:
        if i['label'][0] == 'positive':
            dict['Number of feedbacks'][0] += 1
        elif i['label'][0] == 'neutral':
            dict['Number of feedbacks'][1] += 1
        else:
            dict['Number of feedbacks'][2] += 1
    return dict

