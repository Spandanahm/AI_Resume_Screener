# ---------------------------------------------------------
# AI Resume Screener — Skills + JD-Keyword Explainable AI + Bias Reduction + ATS Breakdown
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4c3cff; font-size:48px; font-weight:800;'>AI Resume Screener</h1>", unsafe_allow_html=True)


# ---------------------------------------------------------
# UI CSS
# ---------------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #ece6ff, #fdeaff);
}
section[data-testid="stSidebar"] {
    background: white;
    padding: 20px;
    border-right: 1px solid #eee;
}
.sidebar-title { font-size:22px; font-weight:700; color:#5b4dff; }
.analysis-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------
# SIDEBAR — Instructions (Always Visible)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📘 Instructions</div>", unsafe_allow_html=True)
    st.markdown("""
    1. Upload Job Description  
    2. Upload Multiple Resumes  
    3. Click **Analyze**  
    4. View Ranking + ATS Breakdown + AI Suggestions  
    """)
    st.markdown("---")
    st.write("✔ Bias Reduction Enabled")
    st.write("✔ Explainable Skills + Keywords")
    


# ---------------------------------------------------------
# SKILLS LIBRARY
# ---------------------------------------------------------
SKILLS_LIB = [
    "python","java","c++","sql","excel","machine learning","nlp","data science",
    "react","django","flask","git","docker","aws","pandas","numpy","javascript",
    "mongodb","html","css","azure","gcp","tensorflow","keras","pytorch","spark",
    "hadoop","tableau","powerbi","linux","bash","rest api","graphql","nodejs","microservices","ci/cd"
]

STOPWORDS = set([
    "and","or","the","a","an","in","on","with","for","of","to","by","is","are","be","as","that","this",
    "will","should","from","at","it","you","your","we","our","role","responsibilities","responsibility",
    "years","year","experience","candidate","candidates","skills","skill","required","preferred"
])

def norm(text):
    return re.sub(r"\s+", " ", text.lower().strip())

# ---------------------------------------------------------
# BIAS REDUCTION: remove sensitive identifiers
# ---------------------------------------------------------
def bias_reduce(text: str) -> str:
    t = norm(text)

    # remove emails
    t = re.sub(r"\S+@\S+\.\S+", " ", t)

    # remove phone numbers
    t = re.sub(r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b", " ", t)

    # remove DOB / dates
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)

    # gender terms
    for g in [" he ", " she ", " him ", " her ", " his ", " hers ", " mr ", " mrs ", " miss ", " ms ", " dr "]:
        t = t.replace(g, " ")

    # nationality / sensitive attributes
    sensitive = ["indian","american","male","female","married","single",
                 "nationality","citizen","religion","hindu","christian","muslim"]
    for s in sensitive:
        t = re.sub(rf"\b{s}\b", " ", t)

    # remove addresses
    t = re.sub(r"\b\d{1,4}\s+\w+\s+(street|st|road|rd|lane|ln|avenue|ave|boulevard|blvd)\b", " ", t)

    # remove "personal information" sections
    t = re.sub(r"personal information.*?(skills|experience|education)", " ", t, flags=re.DOTALL)

    # remove "declaration"
    t = re.sub(r"declaration.*", " ", t, flags=re.DOTALL)

    # clean spaces
    t = re.sub(r"\s+", " ", t).strip()

    return t

# ---------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------
def extract_text(file):
    try:
        name = file.name.lower()
        if name.endswith(".pdf"):
            from PyPDF2 import PdfReader
            return " ".join((p.extract_text() or "") for p in PdfReader(file).pages)
        elif name.endswith(".docx"):
            from docx import Document
            return " ".join(p.text for p in Document(file).paragraphs)
        return file.read().decode("utf-8", errors="ignore")
    except:
        return ""

# ---------------------------------------------------------
# PARSER
# ---------------------------------------------------------
def basic_resume_parser(text):
    safe = bias_reduce(text)
    years = 0
    m = re.search(r"(\d{1,2})\s*(?:\+)?\s*(years|yrs)", safe)
    if m:
        try:
            years = int(m.group(1))
        except:
            years = 0

    found = set()
    for s in SKILLS_LIB:
        if " " in s and s in safe:
            found.add(s)
        elif re.search(rf"\b{re.escape(s)}\b", safe):
            found.add(s)

    return {"text": safe, "years_experience": years, "skills": sorted(found)}

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model_match.pkl"):
        d = joblib.load("model_match.pkl")
        return d["model"], d["vec"]

    df = pd.DataFrame({
        "text": ["python machine learning engineer", "java backend developer"],
        "years_experience": [5, 2],
        "label": [1, 0],
    })

    vec = TfidfVectorizer(stop_words="english")
    Xtxt = vec.fit_transform(df["text"])
    X = hstack([Xtxt, df[["years_experience"]].values])

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit(X, df["label"])
    joblib.dump({"model": model, "vec": vec}, "model_match.pkl")
    return model, vec

model, vec = load_model()

# ---------------------------------------------------------
# JD Keyword Extraction
# ---------------------------------------------------------
def extract_jd_keywords(jd_text):
    jd_norm = norm(jd_text)
    jd_skills = [s for s in SKILLS_LIB if ((" " in s and s in jd_norm) or re.search(rf"\b{s}\b", jd_norm))]
    jd_skills = sorted(set(jd_skills))

    words = re.findall(r"\b[a-zA-Z0-9\-/]+\b", jd_norm)
    other = []
    for w in words:
        if w not in STOPWORDS and len(w) >= 3 and not re.fullmatch(r"\d+", w) and w not in jd_skills:
            other.append(w)

    other = list(dict.fromkeys(other))[:40]
    return {"jd_skills": jd_skills, "jd_keywords": jd_skills + other}

# ---------------------------------------------------------
# REMOVE TABS → ONLY ONE SCREEN
# ---------------------------------------------------------
tab1 = st.container()

# ---------------------------------------------------------
# TAB 1 — ANALYZE (UNCHANGED)
# ---------------------------------------------------------
with tab1:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("## 📄 Upload Files")

    jd_file = st.file_uploader("Upload Job Description", type=["pdf","docx","txt"])
    resume_files = st.file_uploader("Upload Multiple Resumes", type=["pdf","docx","txt"], accept_multiple_files=True)

    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

    if analyze_btn:
        if not jd_file:
            st.error("Please upload JD")
            st.stop()
        if not resume_files:
            st.error("Upload resumes")
            st.stop()

        # ---- JD processing ----
        raw_jd = extract_text(jd_file)
        jd_sanitized = bias_reduce(raw_jd)
        jd_info = extract_jd_keywords(jd_sanitized)
        jd_skills = jd_info["jd_skills"]
        jd_keywords = jd_info["jd_keywords"]

        results = []
        progress = st.progress(0)

        for i, rf in enumerate(resume_files):
            progress.progress(int((i+1)/len(resume_files)*100))

            raw_text = extract_text(rf)
            parsed = basic_resume_parser(raw_text)

            # Similarity
            try:
                v = TfidfVectorizer(stop_words="english")
                sims = v.fit_transform([jd_sanitized, parsed["text"]])
                sim_score = float(cosine_similarity(sims[0:1], sims[1:2])[0][0])
            except:
                sim_score = 0

            # Suitability model
            try:
                Xtxt = vec.transform([parsed["text"]])
                Xnum = np.array([[parsed["years_experience"]]])
                suit = float(model.predict_proba(hstack([Xtxt, Xnum]))[:,1][0])
            except:
                suit = 0

            # JD Skill matching
            matched_skills = [s for s in parsed["skills"] if s in jd_skills]
            missing_skills = [s for s in jd_skills if s not in parsed["skills"]]

            # Keyword match
            matched_keywords = [k for k in jd_keywords if k in parsed["text"]]
            missing_keywords = [k for k in jd_keywords if k not in matched_keywords]

            # ATS scoring
            keyword_pct = len(matched_keywords) / len(jd_keywords) * 100 if jd_keywords else 0
            skill_pct = len(matched_skills) / len(jd_skills) * 100 if jd_skills else 0
            experience_pct = 100 if parsed["years_experience"] >= 3 else parsed["years_experience"] * 25
            readability_pct = min(len(parsed["text"].split()) / 200, 1) * 100

            ats = round(0.35*keyword_pct + 0.35*skill_pct + 0.2*experience_pct + 0.1*readability_pct, 2)

            # Final score
            final_score = round((0.6*suit + 0.4*sim_score)*100, 2)

            # Feedback
            explanation = (
                "Excellent match" if final_score >= 80 else
                "Good match" if final_score >= 60 else
                "Average match" if final_score >= 40 else
                "Weak match"
            )

            suggestions = []
            if missing_skills:
                suggestions.append("Add missing JD skills: " + ", ".join(missing_skills))
            if missing_keywords:
                suggestions.append("Consider adding missing JD keywords: " + ", ".join(missing_keywords[:10]))
            if suit < 0.4:
                suggestions.append("Add measurable achievements for stronger suitability.")
            if sim_score < 0.4:
                suggestions.append("Rewrite summary using more JD keywords.")

            results.append({
                "Resume": rf.name,
                "Final Score": final_score,
                "ATS Score": ats,
                "Matched Skills": "; ".join(matched_skills),
                "Missing Skills": "; ".join(missing_skills),
                "Matched Keywords": "; ".join(matched_keywords),
                "Missing Keywords": "; ".join(missing_keywords),
                "All Detected Skills": "; ".join(parsed["skills"]),
                "Explanation": explanation,
                "Suggestions": " || ".join(suggestions)
            })

        # ---- Ranking ----
        df = pd.DataFrame(results)
        df = df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df["Rank"] = df.index

        st.success("Analysis complete!")
        st.dataframe(df, use_container_width=True)

        fig = px.bar(df, x="Resume", y="Final Score", text="Final Score")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 🧠 Detailed Explainability")
        for _, row in df.iterrows():
            st.markdown(f"### 🏅 Rank {row['Rank']} — {row['Resume']}")
            st.info(row["Explanation"])

            with st.expander("Matched Skills"):
                st.write(row["Matched Skills"] or "None")

            with st.expander("Missing Skills"):
                st.write(row["Missing Skills"] or "None")

            with st.expander("Matched Keywords"):
                st.write(row["Matched Keywords"] or "None")

            with st.expander("Missing Keywords"):
                st.write(row["Missing Keywords"] or "None")

            with st.expander("All Skills Found"):
                st.write(row["All Detected Skills"] or "None")

            with st.expander("AI Suggestions"):
                for s in row["Suggestions"].split(" || "):
                    if s.strip():
                        st.write("- " + s)

            st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)
