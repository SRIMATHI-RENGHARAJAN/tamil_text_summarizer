import nltk
import numpy as np
import re
import networkx as nx
import string
from collections import Counter
from flask import Flask, render_template, request

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------
# NLTK Setup
# -----------------------
def ensure_nltk_resources():
    for res in ["punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
        except LookupError:
            nltk.download(res)

ensure_nltk_resources()

# -----------------------
# Tamil Stopwords
# -----------------------
TAMIL_STOPWORDS = set([
    'அது', 'இது', 'என்று', 'என்ற', 'என', 'இந்த', 'அந்த', 'ஒரு', 'நான்',
    'நீ', 'அவன்', 'அவள்', 'அவர்', 'நாம்', 'நாங்கள்', 'நீங்கள்', 'அவர்கள்',
    'அவை', 'இவை', 'பல', 'என்', 'தன்', 'தன்னுடைய', 'எனது', 'உனது', 'அவனது',
    'அவளது', 'அவரது', 'அவர்களது', 'அவைகளது', 'இவைகளது', 'மேலும்', 'மற்றும்',
    'ஆகிய', 'உள்ள', 'இல்லை', 'போல', 'கொண்டு', 'ஆக', 'வேண்டும்', 'முடியும்',
    'பின்', 'முன்', 'போது', 'கொண்ட', 'வரை', 'போன்ற', 'ஒன்று', 'இரண்டு', 'மூன்று',
    'சில', 'பல', 'என்பது', 'அல்லது', 'ஆனால்', 'எனவே', 'ஆகவே', 'அதனால்', 'இதனால்'
])

# -----------------------
# Tokenizers
# -----------------------
def tamil_sentence_tokenize(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r'([.!?|।॥])\s+', text)

    sentences = []
    for i in range(0, len(parts), 2):
        segment = parts[i]
        punctuation = parts[i + 1] if i + 1 < len(parts) else ""
        sentence = (segment + punctuation).strip()
        if sentence:
            sentences.append(sentence)

    return sentences

def tamil_word_tokenize(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).split()

# -----------------------
# Frequency Calculations
# -----------------------
def build_word_frequency(sentences):
    words = []
    for s in sentences:
        tokens = tamil_word_tokenize(s)
        words.extend([w.lower() for w in tokens if w.lower() not in TAMIL_STOPWORDS])
    return Counter(words)

def calculate_sentence_scores(sentences, word_freq):
    scores = []
    for s in sentences:
        words = [w.lower() for w in tamil_word_tokenize(s) if w.lower() not in TAMIL_STOPWORDS]
        score = sum(word_freq[w] for w in words) / max(1, len(words))
        scores.append(score)
    return scores

# -----------------------
# TextRank Summarizer
# -----------------------
def textrank_summary(text, num_sentences=3):
    sentences = tamil_sentence_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    word_freq = build_word_frequency(sentences)
    sent_scores = calculate_sentence_scores(sentences, word_freq)

    sim_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                s1 = set(tamil_word_tokenize(sentences[i].lower()))
                s2 = set(tamil_word_tokenize(sentences[j].lower()))
                if s1 and s2:
                    sim_matrix[i][j] = len(s1 & s2) / len(s1 | s2)

    graph = nx.from_numpy_array(sim_matrix)

    try:
        pagerank_scores = nx.pagerank(graph)
    except:
        pagerank_scores = {i: 0 for i in range(len(sentences))}

    combined = [
        (0.7 * pagerank_scores[i] + 0.3 * sent_scores[i], i, s)
        for i, s in enumerate(sentences)
    ]

    top_indices = sorted(combined, reverse=True)[:num_sentences]
    top_indices = sorted([idx for _, idx, _ in top_indices])

    summary = ". ".join([sentences[i] for i in top_indices])
    return summary if summary.endswith(".") else summary + "."

# -----------------------
# Frequency Summarizer
# -----------------------
def frequency_summary(text, num_sentences=3):
    sentences = tamil_sentence_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    word_freq = build_word_frequency(sentences)
    max_freq = max(word_freq.values(), default=1)

    for w in word_freq:
        word_freq[w] /= max_freq

    scores = {}
    for i, s in enumerate(sentences):
        words = [w.lower() for w in tamil_word_tokenize(s) if w.lower() not in TAMIL_STOPWORDS]
        if words:
            score = sum(word_freq[w] for w in words) / len(words)
            scores[i] = score

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    top = sorted([i for i, _ in top])

    summary = ". ".join([sentences[i] for i in top])
    return summary if summary.endswith(".") else summary + "."

# -----------------------
# Abstractive Summarizer (mT5 XLSum)
# -----------------------
abs_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
abs_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

def abstractive_summary(text, max_length=100):
    input_text = "summarize: " + text
    inputs = abs_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

    ids = abs_model.generate(
        inputs,
        max_length=max_length,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return abs_tokenizer.decode(ids[0], skip_special_tokens=True)

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    input_text = ""
    method = "textrank"
    lines = 3

    if request.method == "POST":
        input_text = request.form["input_text"]
        method = request.form["method"]
        lines = int(request.form["num_lines"])

        if method == "textrank":
            summary = textrank_summary(input_text, lines)

        elif method == "frequency":
            summary = frequency_summary(input_text, lines)

        elif method == "abstractive":
            summary = abstractive_summary(input_text)

    return render_template("index.html", summary=summary, input_text=input_text, method=method, lines=lines)

if __name__ == "__main__":
    app.run(debug=True)
