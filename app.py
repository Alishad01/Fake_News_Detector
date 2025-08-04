"""
Fake News Detection and Headline Generation Gradio App

- Headline generation using GPT-2
- Fake news detection using BERT

To run locally:
    pip install -r requirements.txt
    python app.py

To deploy:
    Upload this file and requirements.txt to GitHub or Gist.
    Deploy on Hugging Face Spaces, Streamlit Cloud, or similar.

Author: Your Name
"""

import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizerFast, BertForSequenceClassification
import torch

# Load GPT-2 for headline generation
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_model.eval()

def generate_headline(prompt_text="Breaking: ", max_length=30):
    inputs = gpt2_tokenizer.encode(prompt_text, return_tensors='pt')
    outputs = gpt2_model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Load BERT for fake news classification
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
bert_model.eval()

def predict_fake_news(text):
    inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return {"Real": float(probs[0]), "Fake": float(probs[1])}

headline_interface = gr.Interface(
    fn=generate_headline,
    inputs=[gr.Textbox(label="Prompt", value="Breaking: "), gr.Slider(10, 60, value=30, label="Max Length")],
    outputs="text",
    title="Headline Generator (GPT-2)"
)

fake_news_interface = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(label="News Headline or Text"),
    outputs=gr.Label(num_top_classes=2),
    title="Fake News Detector (BERT)"
)

app = gr.TabbedInterface([headline_interface, fake_news_interface], ["Generate Headline", "Detect Fake News"])

if __name__ == "__main__":
    # Main entry point for local or cloud deployment
    app.launch(share=True, server_port=7860)
    # If port 7860 is busy, try:
    # app.launch(share=True, server_port=7861)