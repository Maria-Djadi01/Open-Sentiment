import gradio as gr
import altair as alt
import pandas as pd
from inferance import sent_analy_daridja, sent_analy_not_daridja


def plot_dict(dict):
    source = pd.DataFrame(dict)

    plt = alt.Chart(source).mark_bar().encode(
        x='Sentiment',
        y='Number of feedbacks',
    ).properties(
            width=550,
            height=200
    )

    return plt

def sentiment(text, daridja):
    result = sent_analy_daridja(text) if daridja else sent_analy_not_daridja(text)
    return(plot_dict(result))

demo = gr.Interface(fn=sentiment, inputs=[gr.Textbox(lines=6, placeholder='Enter text here'), 'checkbox'], outputs=["plot"])
demo.launch()
