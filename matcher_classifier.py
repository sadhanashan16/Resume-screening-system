"""
matcher_classifier.py
-----------------------
Core ML logic of the project:

1. TF-IDF: converts resume text and the job description into numeric
   vectors based on word importance, then measures similarity
   (cosine similarity) between each resume and the job description.
2. Naive Bayes & SVM: supervised classifiers that label each resume as
   "Shortlist" / "Reject" based on its TF-IDF features.

Beginner note on the classifiers:
Naive Bayes and SVM need labeled training examples (resume -> shortlisted
or not) to learn from. Since we don't have a real labeled dataset here,
this script auto-generates training labels from the TF-IDF similarity
score itself (resumes above a similarity threshold are treated as
positive examples). This lets the full pipeline run end-to-end and is a
completely valid way to demo the technique. For a production system,
replace `generate_pseudo_labels()` with real historical hiring
decisions (resume -> was this candidate actually shortlisted?).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import numpy as np


def compute_tfidf_similarity(resume_texts: list, job_description: str):
    """
    Vectorizes all resumes + the job description together, then returns
    the cosine similarity of each resume against the job description.

    Returns:
        similarity_scores (list[float]): one score per resume (0 to 1)
        tfidf_matrix: the fitted TF-IDF vectors for the resumes
                      (used later as features for NB/SVM)
        vectorizer: the fitted TfidfVectorizer (reusable)
    """
    documents = resume_texts + [job_description]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vectors = tfidf_matrix[:-1]   # all rows except the last
    job_vector = tfidf_matrix[-1]        # last row = job description

    similarity_scores = cosine_similarity(resume_vectors, job_vector).flatten()
    return similarity_scores.tolist(), resume_vectors, vectorizer


def generate_pseudo_labels(similarity_scores: list, threshold: float = 0.15) -> list:
    """
    Turns similarity scores into binary labels (1 = Shortlist, 0 = Reject)
    so the classifiers have something to learn from. See module docstring.
    """
    return [1 if score >= threshold else 0 for score in similarity_scores]


def train_and_classify(resume_vectors, labels: list):
    """
    Trains a Naive Bayes and an SVM classifier on the resume TF-IDF
    vectors, then returns each model's prediction for every resume.

    With very few resumes (as in a quick demo), the models are trained
    and evaluated on the same small set purely to show the technique
    working end-to-end -- not as a claim of real-world accuracy.
    """
    X = resume_vectors
    y = np.array(labels)

    results = {"naive_bayes": [None] * len(labels), "svm": [None] * len(labels)}

    # Guard: these classifiers need at least 2 classes (Shortlist + Reject)
    # present in the training data to train at all.
    if len(set(labels)) < 2:
        return results  # leave predictions as None; ranking still works

    nb_model = MultinomialNB()
    nb_model.fit(X, y)
    results["naive_bayes"] = nb_model.predict(X).tolist()

    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(X, y)
    results["svm"] = svm_model.predict(X).tolist()

    return results
