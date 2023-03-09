import os

import pandas as pd
import altair as alt
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("alger-ia/dziribert_sentiment")
model = BertForSequenceClassification.from_pretrained("alger-ia/dziribert_sentiment")

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root_path, "models", "classification_model")

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_path)

def plot(nb_feedbacks):
    dict = {
        'Sentiment': ['positive', 'neutral', 'negative'],
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
    
    count = df['Sentiment'].value_counts().to_dict()
    out_file_name = 'open_sentiment.csv'
    df.to_csv(out_file_name, index=False)
    return out_file_name, plot([count['positive'], count['neutral'], count['negative']])

def sent_analy_not_daridja_file(df):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ['positive', 'neutral', 'negative']

    df['Sentiment'] = df['Text'].apply(lambda text: classifier(text, candidate_labels)['labels'][0])

    count = df['Sentiment'].value_counts().to_dict()

    out_file_name = 'open_sentiment.csv'
    df.to_csv(out_file_name)
    return out_file_name, plot([count['positive'], count['neutral'], count['negative']])

def sent_analy_mixed_language_text(text):
    return sent_analy_daridja_text(text) if classify_text(text) == 1 else sent_analy_not_daridja_text(text)

def sent_analy_mixed_language_file(df):
    # Daridja classifier
    sentiment_model = pipeline("text-classification", model="alger-ia/dziribert_sentiment")
    
    # not daridja classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ['positive', 'neutral', 'negative']

    df["classification"] = df["Text"].apply(classify_text)
    daridja_df = df[df['classification'] == 1]
    not_daridja_df = df[df['classification'] == 0]

    daridja_df['Sentiment'] = daridja_df['Text'].apply(lambda text: sentiment_model(text)[0]['label'])

    not_daridja_df['Sentiment'] = not_daridja_df['Text'].apply(lambda text: classifier(text, candidate_labels)['labels'][0])
    
    daridja_df.drop(columns=['classification'], inplace=True)
    not_daridja_df.drop(columns=['classification'], inplace=True)
    
    df_final = pd.concat([not_daridja_df, not_daridja_df])

    count = df['Sentiment'].value_counts().to_dict()

    out_file_name = 'open_sentiment.csv'
    df_final.to_csv(out_file_name)
    return out_file_name, plot([count['positive'], count['neutral'], count['negative']])




def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    pred = probs.argmax().item()
    return pred
