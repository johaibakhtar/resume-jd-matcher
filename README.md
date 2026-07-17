# Resume ↔ Job Description Matcher

An NLP-based tool that matches resumes to job descriptions using sentence embeddings and highlights skill gaps.

## Features
- Semantic similarity matching using Sentence-BERT (`all-MiniLM-L6-v2`)
- Skill gap analysis — shows matched, missing, and extra skills
- Interactive Streamlit web app

## Tech Stack
Python, Streamlit, Sentence-Transformers, scikit-learn, BeautifulSoup, Pandas

## How to Run
1. Clone this repo
2. Create a virtual environment and activate it
3. Install dependencies
4. Run the app
## Datasets Used
- [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- Job postings sample dataset (Trulia/Indeed scrape)

*Note: Dataset files are not included in this repo due to size limits. Download them separately from the links above.*

## How It Works
1. Resume and job description text are cleaned and preprocessed
2. Both are converted into semantic embeddings using Sentence-BERT
3. Cosine similarity is computed to generate a match score
4. A keyword-based skill gap analysis highlights matched, missing, and extra skills