"""
main.py
--------
Run the whole resume screening pipeline:

    1. Load resumes from a folder (PDF/DOCX/TXT).
    2. Extract skills / education / experience with NLP.
    3. Compute TF-IDF similarity of each resume vs. the job description.
    4. Classify each resume with Naive Bayes and SVM.
    5. Rank everything and save a shortlist to a CSV file.

Usage (from inside this folder):
    python main.py --resumes sample_data/resumes --job sample_data/job_description.txt --top 5
"""

import argparse
import os
import pandas as pd

from text_extraction import load_resumes_from_folder
from nlp_extractor import parse_resume
from matcher_classifier import (
    compute_tfidf_similarity,
    generate_pseudo_labels,
    train_and_classify,
)


def run_pipeline(resumes_folder: str, job_description_path: str, top_n: int = 5):
    # 1. Load resumes
    print(f"Loading resumes from '{resumes_folder}' ...")
    resumes_raw = load_resumes_from_folder(resumes_folder)
    if not resumes_raw:
        raise SystemExit("No readable resumes found. Check the folder path.")

    filenames = list(resumes_raw.keys())
    texts = list(resumes_raw.values())

    # 2. Load job description
    with open(job_description_path, "r", encoding="utf-8") as f:
        job_description = f.read()

    # 3. NLP extraction (skills / education / experience) per resume
    print("Extracting skills, education, and experience ...")
    parsed = [parse_resume(t) for t in texts]

    # 4. TF-IDF similarity between each resume and the job description
    print("Computing TF-IDF similarity ...")
    similarity_scores, resume_vectors, _ = compute_tfidf_similarity(texts, job_description)

    # 5. Naive Bayes + SVM classification (Shortlist / Reject)
    print("Running Naive Bayes and SVM classifiers ...")
    labels = generate_pseudo_labels(similarity_scores)
    predictions = train_and_classify(resume_vectors, labels)

    # 6. Assemble results into a table
    rows = []
    for i, filename in enumerate(filenames):
        rows.append({
            "filename": filename,
            "similarity_score": round(similarity_scores[i], 4),
            "skills_found": ", ".join(parsed[i]["skills"]),
            "education_found": ", ".join(parsed[i]["education"]),
            "experience_years": parsed[i]["experience_years"],
            "naive_bayes_prediction": (
                "Shortlist" if predictions["naive_bayes"][i] == 1 else
                "Reject" if predictions["naive_bayes"][i] == 0 else "N/A"
            ),
            "svm_prediction": (
                "Shortlist" if predictions["svm"][i] == 1 else
                "Reject" if predictions["svm"][i] == 0 else "N/A"
            ),
        })

    results_df = pd.DataFrame(rows).sort_values("similarity_score", ascending=False)
    results_df.reset_index(drop=True, inplace=True)

    # 7. Save full results + top-N shortlist
    results_df.to_csv("all_candidates_ranked.csv", index=False)
    shortlist_df = results_df.head(top_n)
    shortlist_df.to_csv("shortlist.csv", index=False)

    print("\n=== Top candidates ===")
    print(shortlist_df.to_string(index=False))
    print(f"\nFull ranking saved to: all_candidates_ranked.csv")
    print(f"Shortlist saved to:    shortlist.csv")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Resume Screening & Job Matching System")
    parser.add_argument("--resumes", required=True, help="Folder containing resume files")
    parser.add_argument("--job", required=True, help="Path to job description .txt file")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to shortlist")
    args = parser.parse_args()

    run_pipeline(args.resumes, args.job, args.top)
