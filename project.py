import streamlit as st
import pandas as pd
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Setup ----------
st.set_page_config(page_title="Resume-JD Matcher", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

SKILLS_LIST = [
    "python", "java", "javascript", "sql", "excel", "aws", "azure", "docker",
    "kubernetes", "machine learning", "data analysis", "power bi", "tableau",
    "html", "css", "react", "node.js", "git", "linux", "agile", "scrum",
    "accounting", "budgeting", "forecasting", "financial analysis", "quickbooks",
    "sap", "erp", "audit", "taxation", "bookkeeping",
    "recruiting", "onboarding", "payroll", "employee relations", "talent acquisition",
    "performance management", "hris", "labor relations",
    "crm", "salesforce", "negotiation", "lead generation", "cold calling",
    "digital marketing", "seo", "social media", "customer service",
    "curriculum development", "lesson planning", "classroom management",
    "special education", "student assessment",
    "leadership", "communication", "project management", "team management",
    "problem solving", "time management", "training", "presentation"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()

def extract_skills(text, skills_list=SKILLS_LIST):
    text_lower = text.lower()
    return set(skill for skill in skills_list if skill in text_lower)

def skill_gap_analysis(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    return {
        "matched": sorted(resume_skills & jd_skills),
        "missing": sorted(jd_skills - resume_skills),
        "extra": sorted(resume_skills - jd_skills)
    }

# ---------- UI ----------
st.title("📄 Resume ↔ Job Description Matcher")
st.write("Paste a resume and a job description below to see the match score and skill gaps.")

col1, col2 = st.columns(2)

with col1:
    resume_input = st.text_area("Resume Text", height=300, placeholder="Paste resume text here...")

with col2:
    jd_input = st.text_area("Job Description Text", height=300, placeholder="Paste job description here...")

if st.button("🔍 Match"):
    if not resume_input.strip() or not jd_input.strip():
        st.warning("Please paste both resume and job description text.")
    else:
        with st.spinner("Analyzing..."):
            resume_clean = clean_text(resume_input)
            jd_clean = clean_text(jd_input)

            resume_emb = model.encode([resume_clean])
            jd_emb = model.encode([jd_clean])
            score = cosine_similarity(resume_emb, jd_emb).flatten()[0]

            gaps = skill_gap_analysis(resume_input, jd_input)

        st.subheader(f"Match Score: {score:.2%}")
        st.progress(float(score))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### ✅ Matched Skills")
            st.write(", ".join(gaps["matched"]) if gaps["matched"] else "None found")
        with c2:
            st.markdown("### ❌ Missing Skills")
            st.write(", ".join(gaps["missing"]) if gaps["missing"] else "None missing!")
        with c3:
            st.markdown("### ➕ Extra Skills")
            st.write(", ".join(gaps["extra"]) if gaps["extra"] else "None")


