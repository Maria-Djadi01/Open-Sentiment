import gradio as gr
import altair as alt
import pandas as pd
from inferance import sent_analy_daridja_text, sent_analy_not_daridja_text, sent_analy_daridja_file, sent_analy_not_daridja_file, sent_analy_mixed_language_file, sent_analy_mixed_language_text

# from ..src.inferance import sent_analy_daridja_text, sent_analy_not_daridja_text, sent_analy_daridja_file, sent_analy_not_daridja_file, sent_analy_mixed_language_file, sent_analy_mixed_language_text

def plot_dict(dict):
    source = pd.DataFrame(dict)

    chart = alt.Chart(source).mark_bar(color="darkblue").encode(
        x='Sentiment',
        y='Number of feedbacks',
    ).properties(
        title='Open Sentiment analysis',
    )

    return chart

def sentiment(text_input, dar, ar, fr, en):
    if(not dar and (ar or fr or en)):
        return sent_analy_not_daridja_text(text_input)
    elif(dar and not(ar or fr or en)):
        return sent_analy_daridja_text(text_input)
    else: return sent_analy_mixed_language_text(text_input)

def sentiment_file(file_input, dar, ar, fr, en):

    df = pd.read_csv(file_input.name, names=['Text'])
    if(not dar and (ar or fr or en)):
        return sent_analy_not_daridja_file(df)
    elif(dar and not(ar or fr or en)):
        return sent_analy_daridja_file(file_input)
    else: sent_analy_mixed_language_file(df)




def launch_demo(share=True):

    with gr.Blocks() as demo:
        gr.Markdown("# Sentiment Analysis")
        
        with gr.Tab("Mixed language sentiment classifier"):
            with gr.Column():
                gr.Markdown("## Select the languages")
                dar = gr.Checkbox(label="Daridja")
                ar = gr.Checkbox(label="Arabic")
                fr = gr.Checkbox(label="French")
                en = gr.Checkbox(label="English")
            with gr.Row():
                file_input = gr.File(label="Upload a file")
                file_output = gr.File(label="Download CSV", interactive=False)  
            analyze_button1 = gr.Button(label="Analyze")
            result_ml = gr.Plot(label="Sentiment analysis")
            analyze_button1.click(fn=sentiment_file, inputs=[file_input, dar, ar, fr, en], outputs=[file_output, result_ml])
        
        with gr.Tab("Specific input classifier"):
            with gr.Column():
                gr.Markdown("## Select the languages")
                dar = gr.Checkbox(label="Daridja")
                ar = gr.Checkbox(label="Arabic")
                fr = gr.Checkbox(label="French")
                en = gr.Checkbox(label="English")
            with gr.Row():
                text_input = gr.Textbox(label="Enter a text")
                output = gr.Textbox(label="Result")
            analyze_button2 = gr.Button(label="Analyze")
            analyze_button2.click(fn = sentiment, inputs=[text_input, dar, ar, fr, en], outputs=output)

    demo.launch(share=share)
