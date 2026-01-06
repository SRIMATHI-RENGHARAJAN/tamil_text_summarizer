# Tamil Text Summarizer (தமிழ் உரை சுருக்கி)

A web-based text summarization application specifically designed for **Tamil language** text. This application provides multiple summarization techniques including extractive and abstractive methods.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Multiple Summarization Methods:**
  - **TextRank** - Graph-based extractive summarization using PageRank algorithm
  - **Frequency-Based** - Extractive summarization based on word frequency analysis
  - **Abstractive (mT5)** - Neural network-based abstractive summarization using multilingual T5 model

- **Tamil Language Support:**
  - Custom Tamil sentence tokenizer
  - Tamil stopwords filtering
  - Support for Tamil punctuation marks

- **User-Friendly Interface:**
  - Clean, modern glassmorphism UI design
  - Responsive design for mobile and desktop
  - Customizable number of output sentences

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Flask | Web framework |
| NLTK | Natural Language Processing |
| NetworkX | Graph-based algorithms (TextRank) |
| NumPy | Numerical computations |
| Transformers | Hugging Face library for mT5 model |
| PyTorch | Deep learning backend |

## 📁 Project Structure

```
text_summarizer/
├── app.py              # Main Flask application
├── summarizer.py       # Summarization algorithms
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend template
├── static/
│   └── images/         # Static assets
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd text_summarizer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources (automatic on first run):**
   The application automatically downloads required NLTK resources (`punkt`, `stopwords`) on first run.

## ▶️ Usage

1. **Start the application:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Using the summarizer:**
   - Paste your Tamil text in the input box
   - Select the number of sentences for the summary
   - Choose your preferred summarization method
   - Click "சுருக்கவும்" (Summarize) to generate the summary

## 📖 Summarization Methods Explained

### 1. TextRank
Uses a graph-based ranking algorithm similar to Google's PageRank. Sentences are ranked based on their similarity to other sentences in the text. The algorithm combines:
- PageRank scores (70% weight)
- Sentence frequency scores (30% weight)

### 2. Frequency-Based
Calculates word frequencies across the document and scores sentences based on the cumulative frequency of their words. Sentences with higher scores are selected for the summary.

### 3. Abstractive (mT5)
Uses the `csebuetnlp/mT5_multilingual_XLSum` pre-trained model from Hugging Face. This method generates new sentences that capture the meaning of the original text, rather than extracting existing sentences.

## ⚙️ Configuration

You can modify the following parameters in `summarizer.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_sentences` | 3 | Number of sentences in extractive summary |
| `max_length` | 100 | Maximum length for abstractive summary |
| `min_length` | 30 | Minimum length for abstractive summary |
| `num_beams` | 4 | Beam search width for mT5 |

## 📝 API Reference

### Summarization Functions

```python
# TextRank summarization
textrank_summary(text, num_sentences=3)

# Frequency-based summarization
frequency_summary(text, num_sentences=3)

# Abstractive summarization
abstractive_summary(text, max_length=100)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the mT5 multilingual model
- [CSEBUETNLP](https://github.com/csebuetnlp) for the XLSum model
- Tamil NLP community for linguistic resources

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

<p align="center">
  Made with ❤️ for Tamil Language Processing
</p>
