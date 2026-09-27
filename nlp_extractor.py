"""
nlp_extractor.py
-----------------
Pulls structured info (skills, education, years of experience) out of
raw resume text using simple NLP techniques: tokenization + keyword/pattern
matching. This is a lightweight, dependency-friendly approach (no heavy
spaCy model download needed) that's easy to explain in a viva.
"""

import re
import nltk
from nltk.tokenize import word_tokenize

# Download the tokenizer data once (safe to call every run; it skips
# the download if already present).
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# A skill vocabulary you can expand freely. This is the "domain knowledge"
# that makes the extraction useful without needing a trained NER model.
SKILL_KEYWORDS = [
    "python", "java", "c++", "c", "sql", "r", "javascript", "html", "css",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "data analysis", "data science", "pandas", "numpy", "scikit-learn",
    "sklearn", "tensorflow", "pytorch", "keras", "excel", "power bi",
    "tableau", "aws", "azure", "gcp", "docker", "kubernetes", "git",
    "django", "flask", "react", "node.js", "communication", "leadership",
    "project management", "agile", "scrum",
]

EDUCATION_KEYWORDS = [
    "b.tech", "btech", "bachelor", "b.sc", "bsc", "m.tech", "mtech",
    "master", "m.sc", "msc", "mba", "phd", "diploma", "b.e", "be",
    "computer science", "information technology", "engineering",
]


def extract_skills(text: str) -> list:
    """Case-insensitive keyword search for known skills in the resume text."""
    text_lower = text.lower()
    found = [skill for skill in SKILL_KEYWORDS if skill in text_lower]
    return sorted(set(found))


def extract_education(text: str) -> list:
    """Case-insensitive keyword search for education-related terms."""
    text_lower = text.lower()
    found = [kw for kw in EDUCATION_KEYWORDS if kw in text_lower]
    return sorted(set(found))


def extract_experience_years(text: str) -> float:
    """
    Looks for patterns like '3 years of experience', '5+ years', etc.
    Returns the largest number found (a reasonable proxy for total experience).
    If nothing is found, returns 0.
    """
    pattern = r"(\d+(?:\.\d+)?)\s*\+?\s*year"
    matches = re.findall(pattern, text.lower())
    years = [float(m) for m in matches]
    return max(years) if years else 0.0


def tokenize(text: str) -> list:
    """Basic word tokenization, used as a preprocessing step for TF-IDF."""
    return word_tokenize(text)


def parse_resume(text: str) -> dict:
    """
    Runs all extractors on a single resume's text and returns a
    structured summary.
    """
    return {
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience_years": extract_experience_years(text),
        "raw_text": text,
    }
