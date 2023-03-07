import pandas as pd
import altair as alt
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("alger-ia/dziribert_sentiment")
model = BertForSequenceClassification.from_pretrained("alger-ia/dziribert_sentiment")


def plot(nb_feedbacks):
    dict = {
        'Sentiment': ['positive', 'negative', 'neutral'],
        'Number of feedbacks': nb_feedbacks
        
    }
    source = pd.DataFrame(dict)
    plt = alt.Chart(source).mark_bar().encode(
        x='Sentiment',
        y='Number of feedbacks',
    ).properties(
            width=550,
            height=200
    )

    return plt

def sent_analy_daridja_text(text):
    sentiment_model = pipeline("text-classification", model="alger-ia/dziribert_sentiment")
    result = sentiment_model(text)
    return result[0]['label']


def sent_analy_not_daridja_text(text):

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ['positive', 'neutral', 'negative']
    result = classifier(text, candidate_labels)
    return result;


def sent_analy_daridja_file(df):
    sentiment_model = pipeline("text-classification", model="alger-ia/dziribert_sentiment")
    
    df['Sentiment'] = df['Text'].apply(lambda text: sentiment_model(text)[0]['label'])
    print(df)
    
    count = df['Sentiment'].value_counts().to_dict()
    print(count)
    return plot([count['positive'], count['neutral'], count['negative']])

def sent_analy_not_daridja_file(df):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ['positive', 'neutral', 'negative']

    df['Sentiment'] = df['Text'].apply(lambda text: classifier(text, candidate_labels)['labels'][0])
    
    count = df['Sentiment'].value_counts().to_dict()

    return (df.to_csv('out.csv') , plot([count['positive'], count['neutral'], count['negative']]))
