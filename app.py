from flask import Flask, render_template, request
from summarizer import textrank_summary, frequency_summary, abstractive_summary

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    input_text = ""
    lines = ""
    method = "textrank"

    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        try:
            lines = int(request.form.get("num_lines", 3))
        except ValueError:
            lines = 3
        method = request.form.get("method", "textrank")

        if input_text.strip():
            if method == "frequency":
                summary = frequency_summary(input_text, num_sentences=lines)
            elif method == "abstractive":
                summary = abstractive_summary(input_text)
            else:
                summary = textrank_summary(input_text, num_sentences=lines)

    return render_template("index.html", summary=summary, input_text=input_text, lines=lines, method=method)

if __name__ == "__main__":
    app.run(debug=True)
