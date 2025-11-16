# app.py
# AI Resume Screener — Skills + JD-Keyword Explainable AI + Bias Reduction + ATS Breakdown
# Updated: removed History tab, added top headline, bias reduction remains in parser

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

# ---------------------------------------------------------
# UI CSS (unchanged visual style)
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
.headline {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #2b2b7a;
    margin-top: 18px;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}
.small-muted { color:#666; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADLINE (big centered title)
# ---------------------------------------------------------
st.markdown('<div class="headline">AI Resume Screener</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#555; margin-bottom:18px;'>Bias-safe, explainable resume matching & ranking</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Skills library (core technical tokens)
# keep/edit this list to tune detection
# ---------------------------------------------------------
SKILLS_LIB = [
    "python","java","c++","sql","excel","machine learning","nlp","data science",
    "react","django","flask","git","docker","aws","pandas","numpy","javascript",
    "mongodb","html","css","azure","gcp","tensorflow","keras","pytorch","spark",
    "hadoop","tableau","powerbi","linux","bash","rest api","graphql","nodejs","microservices","ci/cd"
]

# Stopwords basic list for extracting JD keywords (so we don't include 'and','to',...)
STOPWORDS = set([
    "and","or","the","a","an","in","on","with","for","of","to","by","is","are","be","as","that","this",
    "will","should","from","at","it","you","your","we","our","role","responsibilities","responsibility",
    "years","year","experience","candidate","candidates","skills","skill","required","preferred"
])

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())

# ---------------------------------------------------------
# BIAS REDUCTION: sanitize text to remove sensitive/identifying tokens
# ---------------------------------------------------------
def bias_reduce(text: str) -> str:
    t = norm(text)

    # remove emails
    t = re.sub(r"\S+@\S+\.\S+", " ", t)

    # remove phone numbers (common patterns)
    t = re.sub(r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b", " ", t)

    # remove dates and year numbers (DOB, graduation years)
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)

    # remove gendered pronouns and honorifics
    for token in [" he ", " she ", " him ", " her ", " his ", " hers ", " mr ", " mrs ", " miss ", " ms ", " dr "]:
        t = t.replace(token, " ")

    # remove nationality / religion / marital status tokens
    sensitive = ["indian","american","male","female","married","single","nationality","citizen","religion","hindu","christian","muslim"]
    for s in sensitive:
        t = re.sub(rf"\b{s}\b", " ", t)

    # remove address-like fragments (simple heuristic)
    t = re.sub(r"\b\d{1,4}\s+\w+\s+(street|st|road|rd|lane|ln|avenue|ave|boulevard|blvd)\b", " ", t)

    # remove "personal information" or "declaration" blocks
    t = re.sub(r"personal information.*?(skills|experience|education)", " ", t, flags=re.DOTALL)
    t = re.sub(r"declaration.*", " ", t, flags=re.DOTALL)

    # normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------------------------------------------------------
# Text extraction: same as before (pdf/docx/txt)
# ---------------------------------------------------------
def extract_text(file) -> str:
    try:
        name = file.name.lower()
        if name.endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(file)
            return " ".join((p.extract_text() or "") for p in reader.pages)
        elif name.endswith(".docx"):
            from docx import Document
            doc = Document(file)
            return " ".join(p.text for p in doc.paragraphs)
        else:
            # txt / fallback
            return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------------------------------------------------------
# Basic resume parser: applies bias reduction BEFORE skill extraction
# ---------------------------------------------------------
def basic_resume_parser(text: str) -> dict:
    safe = bias_reduce(text)   # <-- bias reduction is applied here
    years = 0
    m = re.search(r"(\d{1,2})\s*(?:\+)?\s*(years|yrs)", safe)
    if m:
        try:
            years = int(m.group(1))
        except:
            years = 0

    # detect skills from SKILLS_LIB (support multi-word tokens)
    found = set()
    for s in SKILLS_LIB:
        if " " in s:
            if s in safe:
                found.add(s)
        else:
            if re.search(rf"\b{re.escape(s)}\b", safe):
                found.add(s)
    return {"text": safe, "years_experience": years, "skills": sorted(found)}

# ---------------------------------------------------------
# Model loading: unchanged
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model_match.pkl"):
        d = joblib.load("model_match.pkl")
        return d["model"], d["vec"]
    # tiny synthetic fallback model (same as before)
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
# ATS scoring helpers (same as before)
# ---------------------------------------------------------
def extract_jd_keywords(jd_text: str):
    jd_norm = norm(jd_text)
    jd_skills = [s for s in SKILLS_LIB if (" " in s and s in jd_norm) or re.search(rf"\b{re.escape(s)}\b", jd_norm)]
    jd_skills = sorted(set(jd_skills))

    words = re.findall(r"\b[a-zA-Z0-9\-/]+\b", jd_norm)
    other_keys = []
    for w in words:
        if w in STOPWORDS:
            continue
        if re.fullmatch(r"\d+", w):
            continue
        if len(w) < 3:
            continue
        if w in jd_skills:
            continue
        other_keys.append(w)
    other_keys = list(dict.fromkeys(other_keys))
    other_keys = other_keys[:40]
    jd_keywords = jd_skills + [k for k in other_keys if k not in jd_skills]
    return {"jd_skills": jd_skills, "jd_keywords": jd_keywords}

def compute_experience_score(parsed_years: int, required_years: int = None):
    if required_years and required_years > 0:
        score = min(parsed_years / required_years, 1.0) * 100.0
        return round(score, 1)
    else:
        if parsed_years >= 5:
            return 100.0
        elif parsed_years >= 3:
            return 80.0
        elif parsed_years == 2:
            return 50.0
        elif parsed_years == 1:
            return 20.0
        else:
            return 0.0

def compute_readability_score(text: str):
    words = re.findall(r"\w+", text)
    wc = len(words)
    score = min(wc / 200.0, 1.0) * 100.0
    return round(score, 1)

def compute_ats_score(keyword_match_pct, skill_match_pct, experience_pct, readability_pct,
                      weights=(0.35, 0.35, 0.2, 0.1)):
    w_keyword, w_skill, w_exp, w_read = weights
    total = (w_keyword * keyword_match_pct +
             w_skill * skill_match_pct +
             w_exp * experience_pct +
             w_read * readability_pct)
    return round(total, 2)

# ---------------------------------------------------------
# Sidebar (instructions)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📘 Instructions</div>", unsafe_allow_html=True)
    st.markdown("""
    1. Upload Job Description  
    2. Upload Resumes  
    3. Click **Analyze**  
    4. Results include: Ranking, Bias-Reduced Score, ATS Breakdown, AI Suggestions  
    """)
    st.markdown("---")
    st.write("✔ Bias Reduction Enabled")
    st.write("✔ Skills + JD keywords explainability")

# ---------------------------------------------------------
# SINGLE TAB: ANALYZE (History tab removed)
# ---------------------------------------------------------
st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
st.markdown("## 📄 Upload Files")

jd_file = st.file_uploader("Upload Job Description", type=["pdf","docx","txt"])
resume_files = st.file_uploader("Upload Multiple Resumes", type=["pdf","docx","txt"], accept_multiple_files=True)

analyze_btn = st.button("🔍 Analyze", use_container_width=True)

if analyze_btn:
    if not jd_file:
        st.error("Please upload a Job Description.")
        st.stop()
    if not resume_files:
        st.error("Please upload at least one resume.")
        st.stop()

    # Extract and sanitize JD
    raw_jd = extract_text(jd_file)
    jd_sanitized = bias_reduce(raw_jd)
    jd_info = extract_jd_keywords(jd_sanitized)
    jd_skills = jd_info["jd_skills"]
    jd_keywords = jd_info["jd_keywords"]

    # detect required years in JD (e.g., "3+ years", "5 years")
    req_years = None
    mreq = re.search(r"(\d{1,2})\s*\+\s*years|\b(\d{1,2})\s+years\b", raw_jd, flags=re.IGNORECASE)
    if mreq:
        for g in mreq.groups():
            if g and g.isdigit():
                req_years = int(g); break

    results = []
    progress = st.progress(0)

    for i, rf in enumerate(resume_files):
        progress.progress(int((i+1)/len(resume_files)*100))

        raw_text = extract_text(rf)
        parsed = basic_resume_parser(raw_text)   # bias-reduced parsed text, skills, years

        # textual similarity (use sanitized JD and parsed text)
        try:
            v = TfidfVectorizer(stop_words="english")
            sim_matrix = v.fit_transform([jd_sanitized, parsed["text"]])
            sim_score = float(cosine_similarity(sim_matrix[0:1], sim_matrix[1:2])[0][0])
        except Exception:
            sim_score = 0.0

        # model suitability (uses logistic regression trained earlier)
        try:
            Xtxt = vec.transform([parsed["text"]])
            Xnum = np.array([[parsed["years_experience"]]])
            X = hstack([Xtxt, Xnum])
            suit = float(model.predict_proba(X)[:,1][0])
        except Exception:
            suit = 0.0

        # Skill-level matching (SKILLS_LIB controlled)
        matched_skills = sorted([s for s in parsed["skills"] if s in jd_skills])
        missing_skills = sorted([s for s in jd_skills if s not in parsed["skills"]])

        # JD keyword matching (broader than SKILLS_LIB)
        matched_keywords = []
        for kw in jd_keywords:
            if " " in kw:
                if kw in parsed["text"]:
                    matched_keywords.append(kw)
            else:
                if re.search(rf"\b{re.escape(kw)}\b", parsed["text"]):
                    matched_keywords.append(kw)
        matched_keywords = sorted(set(matched_keywords))
        missing_keywords = [k for k in jd_keywords if k not in matched_keywords]

        # compute ATS sub-scores
        keyword_match_pct = (len(matched_keywords) / len(jd_keywords) * 100.0) if jd_keywords else 0.0
        skill_match_pct = (len(matched_skills) / len(jd_skills) * 100.0) if jd_skills else 0.0
        experience_pct = compute_experience_score(parsed["years_experience"], req_years)
        readability_pct = compute_readability_score(parsed["text"])

        ats_score = compute_ats_score(keyword_match_pct, skill_match_pct, experience_pct, readability_pct)

        # final combined score
        final_score = round((0.6 * suit + 0.4 * sim_score) * 100, 2)

        # human-friendly explanation
        if final_score >= 80:
            hl = "Excellent match — strong alignment with JD."
        elif final_score >= 60:
            hl = "Good match — relevant skills present; could improve examples."
        elif final_score >= 40:
            hl = "Average match — some required skills missing."
        else:
            hl = "Weak match — lacks several key JD skills."

        # suggestions
        suggestions = []
        if missing_skills:
            suggestions.append("Add missing technical skills from JD: " + ", ".join(missing_skills[:8]))
        if missing_keywords:
            suggestions.append("Include these JD keywords in your summary/experience: " + ", ".join(missing_keywords[:8]))
        if parsed["years_experience"] < (req_years or 3):
            suggestions.append("If possible, highlight more relevant projects or internships to demonstrate experience.")
        if suit < 0.4:
            suggestions.append("Add measurable outcomes (metrics, percentages) for projects to improve model suitability.")
        if sim_score < 0.4:
            suggestions.append("Rewrite the top summary to include role-specific keywords from the JD.")
        if not suggestions:
            suggestions.append("Resume already aligns well; consider small wording tweaks and quantifiable achievements.")

        # Bias explanation message (concise)
        bias_msg = ("Bias reduction applied: personal identifiers (name, email, phone), gender clues and "
                    "sensitive attributes were removed before scoring to reduce bias.")

        results.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Resume": rf.name,
            "Final Score": final_score,
            "ATS Score": ats_score,
            "Keyword Match %": round(keyword_match_pct, 1),
            "Skill Match %": round(skill_match_pct, 1),
            "Experience %": round(experience_pct, 1),
            "Readability %": round(readability_pct, 1),
            "Similarity %": round(sim_score * 100, 1),
            "Suitability %": round(suit * 100, 1),
            "JD Skills": "; ".join(jd_skills),
            "JD Keywords": "; ".join(jd_keywords[:60]),
            "Matched Skills": "; ".join(matched_skills),
            "Missing Skills": "; ".join(missing_skills),
            "Matched Keywords": "; ".join(matched_keywords[:60]),
            "Missing Keywords": "; ".join(missing_keywords[:60]),
            "All Detected Skills": "; ".join(parsed["skills"]),
            "Bias Explanation": bias_msg,
            "Suggestions": " || ".join(suggestions)
        })

    # Build DataFrame and rank by Final Score (highest -> rank 1)
    df = pd.DataFrame(results)
    if df.empty:
        st.warning("No results produced.")
    else:
        df = df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df["Rank"] = df.index

        st.success("Analysis complete — Bias-Safe Explainable Results ready.")
        display_cols = ["Rank","Resume","Final Score","ATS Score","Keyword Match %","Skill Match %","Experience %","Readability %","Similarity %","Suitability %"]
        st.dataframe(df[display_cols], use_container_width=True)

        # Ranking chart (by Final Score)
        fig = px.bar(df, x="Resume", y="Final Score", text="Final Score", title="Resume Ranking (Bias-Safe)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # Detailed explainable output per resume
        st.markdown("## 🧠 Detailed Explainability & Suggestions")
        for _, row in df.iterrows():
            st.markdown(f"### 🏅 Rank {int(row['Rank'])} — {row['Resume']}")
            st.write(f"**Final Score:** {row['Final Score']}%  •  **ATS Score:** {row['ATS Score']}%")
            st.write(f"**Similarity:** {row['Similarity %']}%  •  **Suitability:** {row['Suitability %']}%")
            st.write(f"**Keyword Match:** {row['Keyword Match %']}%  •  **Skill Match:** {row['Skill Match %']}%  •  **Experience:** {row['Experience %']}%  •  **Readability:** {row['Readability %']}%")
            st.info(row["Bias Explanation"])

            with st.expander("📘 JD Skills (from skills library)"):
                st.write(row["JD Skills"] or "None")

            with st.expander("🔎 JD Keywords (important terms extracted from JD)"):
                st.write(row["JD Keywords"] or "None")

            with st.expander("✔ Matched Skills (in both JD and resume)"):
                st.write(row["Matched Skills"] or "None")

            with st.expander("❌ Missing Skills (required by JD but not in resume)"):
                st.write(row["Missing Skills"] or "None")

            with st.expander("🔑 Matched Keywords (JD keywords found in resume)"):
                st.write(row["Matched Keywords"] or "None")

            with st.expander("❗ Missing Keywords (important JD keywords not found)"):
                st.write(row["Missing Keywords"] or "None")

            with st.expander("🧾 All Detected Skills (from resume parsing)"):
                st.write(row["All Detected Skills"] or "None")

            with st.expander("🤖 AI Suggestions to Improve This Resume"):
                for s in (row["Suggestions"] or "").split(" || "):
                    if s.strip():
                        st.markdown(f"- {s}")

            st.markdown("---")

st.markdown("</div>", unsafe_allow_html=True)
