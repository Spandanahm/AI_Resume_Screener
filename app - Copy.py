# app.py
# AI Resume Screener — Skills-only matching + Ranking + Explainable Feedback
# Paste this file into your project folder (replace old app.py)

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

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
SAVE_PATH = "saved_results.csv"

# -----------------------------
# Styling
# -----------------------------
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
.small-muted { color:#666; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Skills library (used for detection)
# Keep this updated with keywords you care about
# -----------------------------
SKILLS_LIB = [
    "python","java","c++","sql","excel","machine learning","nlp","data science",
    "react","django","flask","git","docker","aws","pandas","numpy","javascript",
    "mongodb","html","css","azure","gcp","tensorflow","keras","pytorch","spark",
    "hadoop","tableau","powerbi","linux","bash","rest","graphql","nodejs"
]

# Helper: normalize text
def norm(text):
    return re.sub(r"\s+", " ", text.lower().strip())

# -----------------------------
# Text extraction
# -----------------------------
def extract_text(file):
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(file)
            return " ".join((p.extract_text() or "") for p in reader.pages)
        elif name.endswith(".docx"):
            from docx import Document
            doc = Document(file)
            return " ".join(p.text for p in doc.paragraphs)
        else:
            return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

# -----------------------------
# Basic parser: years + skill detection
# -----------------------------
def basic_resume_parser(text):
    t = norm(text)
    # years extraction (simple)
    years = 0
    m = re.search(r"(\d{1,2})\s*(?:years|yrs)", t)
    if m:
        try:
            years = int(m.group(1))
        except:
            years = 0

    # find skills (only exact token matches or multi-word matches)
    found = set()
    for s in SKILLS_LIB:
        # check whole word or phrase present
        if " " in s:
            if s in t:
                found.add(s)
        else:
            # match as separate word (avoid partials)
            if re.search(rf"\b{re.escape(s)}\b", t):
                found.add(s)
    return {"text": t, "years_experience": years, "skills": sorted(found)}

# -----------------------------
# Demo model loader (keeps previous behavior)
# -----------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model_match.pkl"):
        d = joblib.load("model_match.pkl")
        return d["model"], d["vec"]
    # tiny synthetic dataset for demo
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

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📘 Instructions</div>", unsafe_allow_html=True)
    st.markdown("""
    1. Upload **Job Description** (PDF/DOCX/TXT)  
    2. Upload **multiple resumes** (PDF/DOCX/TXT)  
    3. Click **Analyze** — results will rank resumes and show details  
    4. Check **History** for previously-saved runs
    """)
    st.markdown("---")
    st.write("✔ No Login Required")
    st.write("✔ Matches **only technical skills** (no stop-words)")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🧠 Analyze Resumes", "📜 History"])

# -----------------------------
# Tab 1 - Analyze
# -----------------------------
with tab1:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("## 📄 Upload Files")
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    resume_files = st.file_uploader("Upload Resumes (multiple)", type=["pdf","docx","txt"], accept_multiple_files=True)

    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

    if analyze_btn:
        if not jd_file:
            st.error("Please upload a Job Description.")
            st.stop()
        if not resume_files or len(resume_files) == 0:
            st.error("Please upload at least one resume.")
            st.stop()

        jd_text = extract_text(jd_file)
        jd_norm = norm(jd_text)

        # detect required skills present in JD (only from SKILLS_LIB)
        jd_skills = []
        for s in SKILLS_LIB:
            if " " in s:
                if s in jd_norm:
                    jd_skills.append(s)
            else:
                if re.search(rf"\b{re.escape(s)}\b", jd_norm):
                    jd_skills.append(s)
        jd_skills = sorted(set(jd_skills))

        results = []
        progress = st.progress(0)

        for i, rf in enumerate(resume_files):
            progress.progress(int((i+1)/len(resume_files) * 100))
            text = extract_text(rf)
            parsed = basic_resume_parser(text)

            # Similarity (textual)
            try:
                v = TfidfVectorizer(stop_words="english")
                sim_matrix = v.fit_transform([jd_text, text])
                sim_score = float(cosine_similarity(sim_matrix[0:1], sim_matrix[1:2])[0][0])
            except Exception:
                sim_score = 0.0

            # Model suitability
            try:
                Xtxt = vec.transform([parsed["text"]])
                Xnum = np.array([[parsed["years_experience"]]])
                X = hstack([Xtxt, Xnum])
                suit = float(model.predict_proba(X)[:,1][0])
            except Exception:
                suit = 0.0

            # Matched & missing skills (ONLY from SKILLS_LIB / JD)
            matched_skills = sorted([s for s in parsed["skills"] if s in jd_skills])
            missing_skills = sorted([s for s in jd_skills if s not in parsed["skills"]])

            # Final score (weighted)
            final_score = round((0.6 * suit + 0.4 * sim_score) * 100, 2)

            # Explanation
            if final_score >= 80:
                explanation = "Excellent match — strong technical alignment with the JD."
            elif final_score >= 60:
                explanation = "Good match — relevant skills present but room for stronger examples."
            elif final_score >= 40:
                explanation = "Average match — some skills present, but missing key technical items."
            else:
                explanation = "Weak match — resume lacks core skills requested by the JD."

            # AI suggestions (simple rule-based)
            suggestions = []
            if missing_skills:
                suggestions.append("Add these required skills from the JD: " + ", ".join(missing_skills[:8]))
            # Missing critical skill suggestions
            critical = {"python","machine learning","sql","aws","docker"}
            missing_critical = sorted(list(critical.intersection(set(jd_skills)) - set(parsed["skills"])))
            if missing_critical:
                suggestions.append("Consider adding/expanding: " + ", ".join(missing_critical))
            if suit < 0.4:
                suggestions.append("Add measurable achievements (project results, numbers) to improve suitability.")
            if sim_score < 0.4:
                suggestions.append("Rewrite summary and include more JD keywords in the top of the resume.")
            if not suggestions:
                suggestions.append("Good job — resume already includes the main JD skills.")

            results.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Resume": rf.name,
                "Score": final_score,
                "Similarity %": round(sim_score*100, 1),
                "Suitability %": round(suit*100, 1),
                "Matched Skills": "; ".join(matched_skills),
                "Missing Skills": "; ".join(missing_skills),
                "All Detected Skills": "; ".join(parsed["skills"]),
                "Explanation": explanation,
                "Suggestions": " || ".join(suggestions)
            })

        # Create DataFrame and rank
        df = pd.DataFrame(results)
        if df.empty:
            st.warning("No results produced.")
        else:
            df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            df["Rank"] = df.index

            st.success("Analysis complete — ranked by Score (higher is better).")
            st.dataframe(df[["Rank","Resume","Score","Similarity %","Suitability %","Matched Skills","Missing Skills"]], use_container_width=True)

            # Save results safely (consistent columns)
            cols = ["Timestamp","Resume","Score","Similarity %","Suitability %","Matched Skills","Missing Skills","All Detected Skills","Explanation","Suggestions","Rank"]
            df_to_save = df[cols]
            header = not os.path.exists(SAVE_PATH)
            df_to_save.to_csv(SAVE_PATH, mode="a", header=header, index=False)

            # Chart
            fig = px.bar(df, x="Resume", y="Score", text="Score", title="Resume Ranking (by Score)")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            # Detailed explainable feedback per ranked resume
            st.markdown("## 🧠 Detailed Feedback (by Rank)")
            for _, row in df.iterrows():
                st.markdown(f"### 🏅 Rank {int(row['Rank'])} — {row['Resume']}")
                st.write(f"**Final Score:** {row['Score']}%  •  **Similarity:** {row['Similarity %']}%  •  **Suitability:** {row['Suitability %']}%")
                st.info(row["Explanation"])
                with st.expander("✔ Matched Skills (skills both in JD and resume)"):
                    ms = row["Matched Skills"] if row["Matched Skills"] else "None"
                    st.write(ms)
                with st.expander("❌ Missing Skills (skills required by JD but absent in resume)"):
                    missing = row["Missing Skills"] if row["Missing Skills"] else "None"
                    st.write(missing)
                with st.expander("🧾 All Detected Skills in Resume (for debugging)"):
                    st.write(row["All Detected Skills"] if row["All Detected Skills"] else "None")
                with st.expander("🤖 AI Suggestions to Improve Resume"):
                    for s in (row["Suggestions"] or "").split(" || "):
                        if s.strip():
                            st.markdown(f"- {s}")
                st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Tab 2 - History
# -----------------------------
with tab2:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("## 📜 Analysis History")
    if not os.path.exists(SAVE_PATH):
        st.info("No saved history yet.")
    else:
        # Read safely and guard parse errors
        try:
            df_all = pd.read_csv(SAVE_PATH)
            st.dataframe(df_all, use_container_width=True)
        except Exception as e:
            st.error("Could not read saved results: " + str(e))
            st.write("You can delete or fix the CSV file and re-run.")

    st.markdown("</div>", unsafe_allow_html=True)
