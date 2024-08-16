from flask import Flask
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer, PegasusForConditionalGeneration, PegasusTokenizer
import evaluate
import nltk

# Ensure that the NLTK sentence tokenizer is available
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Load the T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the PEGASUS model and tokenizer
pegasus_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

# Load the ROUGE metric
rouge = evaluate.load("rouge")

# Function to generate a summary using T5
def generate_t5_summary(text):
    num_beams = 25  # Further increase beams for more diverse summaries
    length_penalty = 1.0  # Neutral to balance summary length
    no_repeat_ngram_size = 2  # Allow for more bigram coverage
    max_length = 150  # Focus on concise yet informative summaries
    min_length = 80  # Ensure summary includes core content
    do_sample = False

    t5_inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    t5_summary_ids = t5_model.generate(t5_inputs, max_length=max_length, min_length=min_length, 
                                       num_beams=num_beams, length_penalty=length_penalty, 
                                       no_repeat_ngram_size=no_repeat_ngram_size, 
                                       do_sample=do_sample, early_stopping=True)
    t5_summary = t5_tokenizer.decode(t5_summary_ids[0], skip_special_tokens=True)
    
    return t5_summary

# Function to generate a summary using PEGASUS
def generate_pegasus_summary(text):
    num_beams = 25
    length_penalty = 1.2
    no_repeat_ngram_size = 2
    max_length = 150
    min_length = 80
    do_sample = False

    pegasus_inputs = pegasus_tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=512)
    pegasus_summary_ids = pegasus_model.generate(pegasus_inputs['input_ids'], max_length=max_length, min_length=min_length, 
                                                 num_beams=num_beams, length_penalty=length_penalty, 
                                                 no_repeat_ngram_size=no_repeat_ngram_size, 
                                                 do_sample=do_sample, early_stopping=True)
    pegasus_summary = pegasus_tokenizer.decode(pegasus_summary_ids[0], skip_special_tokens=True)
    
    return pegasus_summary

# Function to generate a combined summary with an emphasis on bigrams
def generate_weighted_combined_summary(text, weight_t5=0.4, weight_pegasus=0.6):
    t5_summary = generate_t5_summary(text)
    pegasus_summary = generate_pegasus_summary(text)

    # Tokenize summaries into sentences
    t5_sentences = nltk.sent_tokenize(t5_summary)
    pegasus_sentences = nltk.sent_tokenize(pegasus_summary)

    # Combine sentences with a focus on maximizing bigram overlap
    combined_sentences = []
    combined_sentences.extend(t5_sentences[:int(len(t5_sentences) * weight_t5)])
    combined_sentences.extend(pegasus_sentences[:int(len(pegasus_sentences) * weight_pegasus)])

    # Combine the sentences into the final summary
    combined_summary = " ".join(combined_sentences)
    
    return combined_summary

# Function to calculate ROUGE scores
def calculate_rouge_scores(generated_summary, reference_summary):
    scores = rouge.compute(predictions=[generated_summary], references=[reference_summary])
    return {
        "ROUGE-1": scores['rouge1'],
        "ROUGE-2": scores['rouge2'],
        "ROUGE-L": scores['rougeL']
    }

# Define the Gradio interface
def gradio_interface():
    iface = gr.Interface(fn=generate_weighted_combined_summary, 
                         inputs="textbox", 
                         outputs="textbox", 
                         title="Text Summarizer",
                         description="Enter a text to generate a summary using combined T5 and PEGASUS models.")
    return iface.launch(prevent_thread_lock=True)

# Define the home route for Flask
@app.route("/")
def home():
    return gradio_interface()

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
