# AI-Based Resume Screening and Job Matching System

An AI system that screens resumes and matches them against a job description
automatically — built to solve the problem of manually reviewing hundreds of
resumes per role, which is slow, inconsistent, and prone to human bias.

## What it does

1. **Reads resumes** in PDF, DOCX, or TXT format from a folder.
2. **Extracts key details** — skills, education, years of experience — using
   NLP (tokenization + keyword/pattern matching).
3. **Matches each resume to a job description** using **TF-IDF** vectorization
   and cosine similarity.
4. **Classifies and ranks candidates** using **Naive Bayes** and **SVM**
   classifiers trained on the resume features.
5. **Outputs a ranked shortlist** of the top candidates as CSV files.

## Tech stack

Python, Scikit-learn, Pandas, NLTK, PyPDF2, python-docx

## Project structure

```
resume_screening_system/
├── app.py                   # Flask server — REAL website, calls the actual Python/Scikit-learn pipeline
├── templates/
│   └── index.html           # Frontend for app.py (talks to it over the network)
├── app.html                 # Standalone single-file demo (JS reimplementation, no backend needed)
├── main.py                  # Python/Scikit-learn pipeline as a terminal script
├── text_extraction.py       # Reads text out of PDF / DOCX / TXT resumes
├── nlp_extractor.py         # Extracts skills, education, experience
├── matcher_classifier.py    # TF-IDF matching + Naive Bayes / SVM classification
├── requirements.txt
├── sample_data/
│   ├── job_description.txt
│   └── resumes/             # 5 sample resumes (.txt) to demo with
└── README.md
```

## Option A — Real website with a live Python backend (recommended for submission)

This is the one where the frontend genuinely calls your Python/Scikit-learn code.

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Upload resumes, paste a job
description, click **Screen candidates** — the browser sends everything to your
Flask server, which runs `text_extraction.py` → `nlp_extractor.py` →
`matcher_classifier.py` (the exact same TF-IDF / Naive Bayes / SVM code `main.py`
uses) and sends real results back as JSON.

Leave the terminal window open while you use the site — that's your server running.
Press `Ctrl+C` in the terminal to stop it.

## Option B — Standalone single-file demo (app.html)

Double-click `app.html` — no install, no terminal, works offline. It reimplements
the same TF-IDF / Naive Bayes / SVM logic in JavaScript so it runs entirely in the
browser with zero setup. Good as a backup demo or a quick preview, but it is **not**
connected to the Python code — see Option A for the version that actually is.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --resumes sample_data/resumes --job sample_data/job_description.txt --top 5
```

This prints a ranked table to the terminal and writes two files:
- `all_candidates_ranked.csv` — every resume, ranked by similarity score
- `shortlist.csv` — just the top N candidates

To use it with your own data: put PDF/DOCX resumes in a folder and point
`--resumes` at it, and put your job description text in a `.txt` file and
point `--job` at it.

## How the ML actually works (for your viva / demo)

- **TF-IDF (Term Frequency–Inverse Document Frequency)** turns each resume
  and the job description into a vector of word importance scores, so
  common words (like "the", "and") matter less and distinctive, relevant
  words (like "TensorFlow", "SQL") matter more.
- **Cosine similarity** measures the angle between a resume's vector and
  the job description's vector — a score close to 1 means a strong match,
  close to 0 means a weak match. This produces the ranking.
- **Naive Bayes and SVM** are supervised classifiers that learn to label a
  resume "Shortlist" or "Reject" from its TF-IDF features. Since there's no
  real historical hiring dataset here, this project generates training
  labels from the similarity score itself (above a threshold = positive
  example) so the full pipeline runs end-to-end and both algorithms from
  the spec are genuinely used. **For a real deployment, you'd replace this
  with actual past hiring decisions** (resume → was this person hired or
  shortlisted?) as the training labels — that's the natural "next step" to
  mention if asked.

## Limitations / possible improvements (good talking points)

- Skill/education extraction is keyword-based, not a trained NER model —
  simple and explainable, but can be expanded with a spaCy NER model.
- `.doc` (legacy Word format) isn't supported directly — convert to
  `.docx` or PDF first, or add a library like `textract`.
- Classifier labels are currently derived from similarity score, not real
  hiring outcomes — swap in real labeled data for production use.
