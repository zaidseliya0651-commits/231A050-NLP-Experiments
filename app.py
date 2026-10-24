import os
import re
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, redirect, url_for, send_file

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to store trained components
global_model = None
global_vectorizer = None
global_df = None
model_accuracy = 0.0

# Text cleaning function from the project notebook[span_1](start_span)[span_1](end_span)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()       # remove extra spaces
    return text

# Topic identification logic[span_2](start_span)[span_2](end_span)
topic_keywords = {
    "Teacher": ["teacher", "faculty", "professor", "teaching", "explain", "explanation", "lecture", "lectures"],
    "Course": ["course", "subject", "syllabus", "unit", "topic", "topics", "semester"],
    "Practical": ["practical", "lab", "laboratory", "experiment", "programming", "coding"],
    "Infrastructure": ["classroom", "infrastructure", "computer", "computers", "equipment", "facility", "facilities"]
}

def detect_topic(text):
    text = clean_text(text)
    scores = {topic: sum(1 for word in words if word in text) for topic, words in topic_keywords.items()}
    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else "General"

# Priority detection logic[span_3](start_span)[span_3](end_span)
high_priority_words = ["poor", "bad", "worst", "confusing", "unclear", "problem", "issue", "difficult", "not clear", "very slow", "lack"]
medium_priority_words = ["average", "okay", "improve", "improvement", "moderate"]

def detect_priority(text, sentiment):
    t = clean_text(text)
    if sentiment == "Negative" or any(word in t for word in high_priority_words):
        return "High"
    if sentiment == "Neutral" or any(word in t for word in medium_priority_words):
        return "Medium"
    return "Low"

# Recommendation mapping[span_4](start_span)[span_4](end_span)
recommendations = {
    ("Teacher", "Negative"): "Review teaching clarity and provide additional explanations.",
    ("Teacher", "Neutral"): "Collect specific suggestions and improve explanation where needed.",
    ("Teacher", "Positive"): "Maintain the effective teaching practices.",
    ("Course", "Negative"): "Review course content, difficulty, and topic coverage.",
    ("Course", "Neutral"): "Collect suggestions to improve course structure and content.",
    ("Course", "Positive"): "Continue the current course structure and content.",
    ("Practical", "Negative"): "Review practical instructions, experiments, and lab support.",
    ("Practical", "Neutral"): "Improve practical guidance and collect detailed lab feedback.",
    ("Practical", "Positive"): "Continue the effective practical activities.",
    ("Infrastructure", "Negative"): "Inspect facilities and address the reported infrastructure issue.",
    ("Infrastructure", "Neutral"): "Review facilities and identify possible improvements.",
    ("Infrastructure", "Positive"): "Maintain the current facilities and resources.",
    ("General", "Negative"): "Review the feedback and investigate the underlying concern.",
    ("General", "Neutral"): "Collect more specific feedback for improvement.",
    ("General", "Positive"): "Maintain the positive aspects mentioned by students."
}

def make_recommendation(topic, sentiment):
    return recommendations.get((topic, sentiment), "Review the feedback and take an appropriate action.")

# HTML Template with clean white design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Feedback Insight & Action System</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 0; }
        header { background: #ffffff; border-bottom: 1px solid #e2e8f0; padding: 20px 40px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        h1 { margin: 0; font-size: 24px; color: #0f172a; }
        .subtitle { font-size: 14px; color: #64748b; margin-top: 5px; }
        .container { max-width: 1000px; margin: 30px auto; background: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #e2e8f0; }
        .section { margin-bottom: 30px; }
        h2 { font-size: 18px; color: #334155; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px; }
        input[type="file"], textarea { width: 100%; padding: 12px; margin-top: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; box-sizing: border-box; }
        textarea { height: 100px; resize: vertical; }
        button { background: #2563eb; color: white; border: none; padding: 12px 24px; font-size: 14px; font-weight: 600; border-radius: 6px; cursor: pointer; margin-top: 10px; }
        button:hover { background: #1d4ed8; }
        .result-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-top: 20px; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        .badge-pos { background: #dcfce7; color: #166534; }
        .badge-neu { background: #fef9c3; color: #854d0e; }
        .badge-neg { background: #fee2e2; color: #991b1b; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }
        th, td { border: 1px solid #e2e8f0; padding: 10px; text-align: left; }
        th { background: #f1f5f9; color: #475569; }
        .download-btn { background: #10b981; text-decoration: none; display: inline-block; padding: 10px 20px; color: white; border-radius: 6px; font-weight: 600; margin-top: 10px; }
        .download-btn:hover { background: #059669; }
    </style>
</head>
<body>
    <header>
        <h1>🎓 Student Feedback Insight & Action Recommendation System</h1>
        <div class="subtitle">NLP Mini Project | Zaid Seliya (UIN: 231A050)</div>
    </header>
    <div class="container">
        {% if not trained %}
        <div class="section">
            <h2>Step 1: Upload Dataset (CSV)</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv" required>
                <button type="submit">Upload & Train Model</button>
            </form>
        </div>
        {% else %}
        <div class="section" style="display: flex; justify-content: space-between; align-items: center; background: #f0fdf4; padding: 15px 20px; border-radius: 8px; border: 1px solid #bbf7d0;">
            <div>
                <strong>Model Status:</strong> Trained Successfully!<br>
                <span style="font-size: 13px; color: #166534;">Test Set Accuracy: {{ "%.2f"|format(accuracy * 100) }}%</span>
            </div>
            <a href="/reset" style="color: #dc2626; text-decoration: none; font-weight: 600; font-size: 14px;">Upload New Dataset</a>
        </div>

        <div class="section">
            <h2>Test New Student Feedback</h2>
            <form action="/analyze" method="post">
                <textarea name="feedback_text" placeholder="Type or paste student feedback here..." required>{{ input_text }}</textarea>
                <button type="submit">Analyze Feedback</button>
            </form>

            {% if result %}
            <div class="result-box">
                <h3>Analysis Results:</h3>
                <p><strong>Feedback:</strong> "{{ result.Feedback }}"</p>
                <p><strong>Sentiment:</strong> 
                    <span class="badge {% if result.Sentiment == 'Positive' %}badge-pos{% elif result.Sentiment == 'Neutral' %}badge-neu{% else %}badge-neg{% endif %}">
                        {{ result.Sentiment }}
                    </span>
                </p>
                <p><strong>Topic:</strong> {{ result.Topic }}</p>
                <p><strong>Priority:</strong> {{ result.Priority }}</p>
                <p><strong>Suggested Recommendation:</strong> {{ result.Recommendation }}</p>
            </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>Generated Insight Report Preview</h2>
            <div style="max-height: 300px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Feedback</th>
                            <th>Sentiment</th>
                            <th>Topic</th>
                            <th>Priority</th>
                            <th>Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in report_preview %}
                        <tr>
                            <td>{{ row.feedback }}</td>
                            <td>{{ row.sentiment }}</td>
                            <td>{{ row.topic }}</td>
                            <td>{{ row.priority }}</td>
                            <td>{{ row.recommendation }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            <a class="download-btn" href="/download">Download Full Report (CSV)</a>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    global global_model, model_accuracy
    trained = global_model is not None
    return render_template_string(HTML_TEMPLATE, trained=trained, accuracy=model_accuracy)

@app.route('/upload', methods=['POST'])
def upload():
    global global_model, global_vectorizer, global_df, model_accuracy
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Load dataset & train model as specified in notebook[span_5](start_span)[span_5](end_span)
    df = pd.read_csv(filepath)
    if not {"feedback", "sentiment"}.issubset(df.columns):
        return "Error: CSV must contain 'feedback' and 'sentiment' columns."
    
    df = df.dropna(subset=["feedback", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()
    df["clean_feedback"] = df["feedback"].apply(clean_text)
    
    X = df["clean_feedback"]
    y = df["sentiment"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    model_accuracy = accuracy_score(y_test, y_pred)
    
    # Enrich dataframe with topic, priority, recommendation
    df["topic"] = df["feedback"].apply(detect_topic)
    df["priority"] = [detect_priority(t, s) for t, s in zip(df["feedback"], df["sentiment"])]
    df["recommendation"] = [make_recommendation(t, s) for t, s in zip(df["topic"], df["sentiment"])]
    
    global_model = model
    global_vectorizer = vectorizer
    global_df = df
    
    df.to_csv("student_feedback_insight_report.csv", index=False)
    return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    global global_model, global_vectorizer, global_df, model_accuracy
    text = request.form.get('feedback_text', '')
    
    cleaned = clean_text(text)
    features = global_vectorizer.transform([cleaned])
    sentiment = global_model.predict(features)[0]
    topic = detect_topic(text)
    priority = detect_priority(text, sentiment)
    recommendation = make_recommendation(topic, sentiment)
    
    result = {
        "Feedback": text,
        "Sentiment": sentiment,
        "Topic": topic,
        "Priority": priority,
        "Recommendation": recommendation
    }
    
    report_preview = global_df[['feedback', 'sentiment', 'topic', 'priority', 'recommendation']].head(10).to_dict('records')
    return render_template_string(HTML_TEMPLATE, trained=True, accuracy=model_accuracy, result=result, input_text=text, report_preview=report_preview)

@app.route('/download')
def download():
    return send_file("student_feedback_insight_report.csv", as_attachment=True)

@app.route('/reset')
def reset():
    global global_model, global_vectorizer, global_df, model_accuracy
    global_model = None
    global_vectorizer = None
    global_df = None
    model_accuracy = 0.0
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
