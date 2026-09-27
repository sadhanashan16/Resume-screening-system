"""
app.py
-------
A small Flask server that makes the ACTUAL Python/Scikit-learn pipeline
(text_extraction.py, nlp_extractor.py, matcher_classifier.py) available to
a real website. This is what genuinely links the frontend and backend:

    Browser (templates/index.html)
        --uploads resumes + job description-->
    Flask route /api/screen
        --calls--> your real Python/Scikit-learn code
        --returns JSON results-->
    Browser renders them

Run with:  python app.py
Then open: http://127.0.0.1:5000 in your browser.
"""

import os
import tempfile

from flask import Flask, request, jsonify, render_template

from text_extraction import extract_text
from nlp_extractor import parse_resume
from matcher_classifier import (
    compute_tfidf_similarity,
    generate_pseudo_labels,
    train_and_classify,
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/screen", methods=["POST"])
def screen():
    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    uploaded_files = request.files.getlist("resumes")
    if not uploaded_files:
        return jsonify({"error": "Please upload at least one resume."}), 400

    texts, filenames, skipped = [], [], []

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded_files:
            if not f.filename:
                continue
            saved_path = os.path.join(tmpdir, f.filename)
            f.save(saved_path)
            try:
                text = extract_text(saved_path)
            except ValueError as e:
                skipped.append(f"{f.filename} ({e})")
                continue
            if text.strip():
                texts.append(text)
                filenames.append(f.filename)
            else:
                skipped.append(f"{f.filename} (no extractable text)")

    if not texts:
        return jsonify({"error": "No readable resumes found.", "skipped": skipped}), 400

    # --- This is the real ML pipeline, same code main.py uses ---
    parsed = [parse_resume(t) for t in texts]
    similarity_scores, resume_vectors, _ = compute_tfidf_similarity(texts, job_description)
    labels = generate_pseudo_labels(similarity_scores)
    predictions = train_and_classify(resume_vectors, labels)
    # ---------------------------------------------------------------

    def label(preds, i):
        if preds[i] is None:
            return "N/A"
        return "Shortlist" if preds[i] == 1 else "Reject"

    results = []
    for i, filename in enumerate(filenames):
        results.append({
            "filename": filename,
            "score": round(similarity_scores[i], 4),
            "skills": parsed[i]["skills"],
            "education": parsed[i]["education"],
            "experience_years": parsed[i]["experience_years"],
            "naive_bayes": label(predictions["naive_bayes"], i),
            "svm": label(predictions["svm"], i),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return jsonify({"results": results, "skipped": skipped})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
