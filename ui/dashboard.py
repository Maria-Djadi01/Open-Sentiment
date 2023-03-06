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

with gr.Blocks() as demo:
    gr.Markdown("## Sentiment Analysis")
    
    with gr.Tab("Mixed language sentiment classifier"):
        with gr.Column():
            gr.Markdown("### Select the languages you wanna analyze with")
            dar = gr.Checkbox(label="Daridja")
            fr = gr.Checkbox(label="French")
            en = gr.Checkbox(label="English")
        with gr.Row():
            file_input = gr.inputs.File(label="Upload a file")
            file_output = gr.outputs.File(label="Download the file")  
        analyze_button1 = gr.Button(label="Analyze")
        analyze_button1.click(fn=sentiment, inputs=file_input, outputs=file_output)
        result_ml = gr.Plot(label="Sentiment analysis")
    
    with gr.Tab("Specific input classifier"):
        with gr.Column():
            gr.Markdown("### Select the languages you wanna analyze with")
            dar = gr.Checkbox(label="Daridja")
            fr = gr.Checkbox(label="French")
            en = gr.Checkbox(label="English")
        with gr.Row():
            text_input = gr.Textbox(label="Enter a text")
            output = gr.Textbox(label="Result")
        analyze_button2 = gr.Button(label="Analyze")
        analyze_button2.click(fn = sentiment, inputs=text_input, outputs=output)
        
demo.launch()
