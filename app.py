from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
CORS(app)

# ── Load & train models ──────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "career_dataset.csv"))

FEATURE_COLS = [
    "programming", "mathematics", "communication", "problemSolving",
    "creativity", "leadership", "analyticalThinking", "teamwork",
    "dataAnalysis", "networking",
    "logical_aptitude", "verbal_aptitude", "quantitative_aptitude"
]

X = df[FEATURE_COLS].values
y = df["career"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42
)

# Train models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train, y_train)

# Accuracy
rf_acc  = round(accuracy_score(y_test, rf.predict(X_test))  * 100, 2)
knn_acc = round(accuracy_score(y_test, knn.predict(X_test)) * 100, 2)
svm_acc = round(accuracy_score(y_test, svm.predict(X_test)) * 100, 2)

print(f"✅ Models trained  |  RF: {rf_acc}%  KNN: {knn_acc}%  SVM: {svm_acc}%")

# ── Career metadata ──────────────────────────────────────────────────────────

CAREER_META = {
    "Software Developer": {
        "icon": "💻", "color": "#06b6d4",
        "description": "Design, build and maintain scalable software systems and applications.",
        "avgSalary": "₹6–18 LPA", "growthRate": "+25% by 2030",
        "requiredSkills": ["Programming", "Problem Solving", "Teamwork", "Analytical Thinking"],
        "requiredLevels": {"programming": 7, "mathematics": 6, "problemSolving": 7, "analyticalThinking": 6},
        "learning": [
            {"title": "The Odin Project – Full Stack Path", "platform": "The Odin Project", "duration": "6–9 months", "type": "Course"},
            {"title": "CS50x – Intro to Computer Science", "platform": "Harvard / edX", "duration": "12 weeks", "type": "Course"},
            {"title": "AWS Certified Developer Associate", "platform": "AWS", "duration": "3 months", "type": "Certification"},
            {"title": "Build 5 Portfolio Projects", "platform": "GitHub", "duration": "Ongoing", "type": "Project"},
        ],
    },
    "Data Scientist": {
        "icon": "🧠", "color": "#f59e0b",
        "description": "Extract insights from large datasets using statistical modeling and ML.",
        "avgSalary": "₹8–22 LPA", "growthRate": "+36% by 2030",
        "requiredSkills": ["Mathematics", "Data Analysis", "Programming", "Analytical Thinking"],
        "requiredLevels": {"mathematics": 8, "dataAnalysis": 7, "programming": 6, "analyticalThinking": 8},
        "learning": [
            {"title": "Applied Data Science with Python", "platform": "Coursera / Michigan", "duration": "5 months", "type": "Course"},
            {"title": "Google Data Analytics Certificate", "platform": "Coursera", "duration": "6 months", "type": "Certification"},
            {"title": "Kaggle – 5 Competition Projects", "platform": "Kaggle", "duration": "3 months", "type": "Project"},
            {"title": "Hands-On ML with Scikit-Learn & TF", "platform": "O'Reilly", "duration": "Self-paced", "type": "Book"},
        ],
    },
    "ML Engineer": {
        "icon": "🤖", "color": "#a78bfa",
        "description": "Build, train, and deploy production ML models and intelligent pipelines at scale.",
        "avgSalary": "₹10–28 LPA", "growthRate": "+40% by 2030",
        "requiredSkills": ["Programming", "Mathematics", "Data Analysis", "Problem Solving"],
        "requiredLevels": {"programming": 8, "mathematics": 8, "dataAnalysis": 7, "analyticalThinking": 7},
        "learning": [
            {"title": "Deep Learning Specialization", "platform": "Coursera / DeepLearning.AI", "duration": "4 months", "type": "Course"},
            {"title": "MLOps Specialization", "platform": "Coursera", "duration": "3 months", "type": "Certification"},
            {"title": "TensorFlow Developer Certificate", "platform": "Google", "duration": "2 months", "type": "Certification"},
            {"title": "End-to-End ML Project on GitHub", "platform": "GitHub / HuggingFace", "duration": "2 months", "type": "Project"},
        ],
    },
    "Cybersecurity Analyst": {
        "icon": "🛡️", "color": "#f43f5e",
        "description": "Protect organizations from cyber threats through monitoring and incident response.",
        "avgSalary": "₹6–20 LPA", "growthRate": "+33% by 2030",
        "requiredSkills": ["Networking", "Analytical Thinking", "Problem Solving", "Programming"],
        "requiredLevels": {"networking": 7, "analyticalThinking": 7, "problemSolving": 7, "programming": 5},
        "learning": [
            {"title": "CompTIA Security+ Certification", "platform": "CompTIA", "duration": "2–3 months", "type": "Certification"},
            {"title": "Google Cybersecurity Certificate", "platform": "Coursera", "duration": "6 months", "type": "Course"},
            {"title": "TryHackMe – SOC Level 1 Path", "platform": "TryHackMe", "duration": "3 months", "type": "Project"},
            {"title": "CEH – Certified Ethical Hacker", "platform": "EC-Council", "duration": "2 months", "type": "Certification"},
        ],
    },
    "Cloud Engineer": {
        "icon": "☁️", "color": "#38bdf8",
        "description": "Design and manage scalable cloud infrastructure on AWS, Azure, or GCP.",
        "avgSalary": "₹7–22 LPA", "growthRate": "+28% by 2030",
        "requiredSkills": ["Networking", "Programming", "Problem Solving", "Teamwork"],
        "requiredLevels": {"networking": 7, "programming": 6, "problemSolving": 6, "analyticalThinking": 6},
        "learning": [
            {"title": "AWS Solutions Architect Associate", "platform": "AWS / A Cloud Guru", "duration": "2–3 months", "type": "Certification"},
            {"title": "Google Cloud Professional Data Engineer", "platform": "Google Cloud", "duration": "3 months", "type": "Certification"},
            {"title": "Kubernetes & Docker Fundamentals", "platform": "Udemy", "duration": "6 weeks", "type": "Course"},
            {"title": "Deploy Full-Stack App to AWS", "platform": "GitHub", "duration": "1 month", "type": "Project"},
        ],
    },
    "UX/UI Designer": {
        "icon": "🎨", "color": "#fb923c",
        "description": "Craft intuitive, beautiful user experiences that bridge human needs and digital products.",
        "avgSalary": "₹5–16 LPA", "growthRate": "+22% by 2030",
        "requiredSkills": ["Creativity", "Communication", "Problem Solving", "Analytical Thinking"],
        "requiredLevels": {"creativity": 8, "communication": 7, "problemSolving": 6},
        "learning": [
            {"title": "Google UX Design Certificate", "platform": "Coursera", "duration": "6 months", "type": "Certification"},
            {"title": "Figma Mastery Course", "platform": "Udemy", "duration": "4 weeks", "type": "Course"},
            {"title": "Design 3 Case Study Projects", "platform": "Behance / Dribbble", "duration": "2 months", "type": "Project"},
            {"title": "The Design of Everyday Things", "platform": "Book", "duration": "Self-paced", "type": "Book"},
        ],
    },
    "Product Manager": {
        "icon": "📊", "color": "#34d399",
        "description": "Lead cross-functional teams to define, build and launch products that users love.",
        "avgSalary": "₹9–25 LPA", "growthRate": "+19% by 2030",
        "requiredSkills": ["Communication", "Leadership", "Analytical Thinking", "Problem Solving"],
        "requiredLevels": {"communication": 8, "leadership": 7, "analyticalThinking": 7, "problemSolving": 7},
        "learning": [
            {"title": "Product Management Fundamentals", "platform": "Coursera / PM School", "duration": "3 months", "type": "Course"},
            {"title": "PSPO I – Scrum Product Owner", "platform": "Scrum.org", "duration": "1 month", "type": "Certification"},
            {"title": "Conduct User Research & Write PRD", "platform": "Self-directed", "duration": "1 month", "type": "Project"},
            {"title": "Inspired: How to Create Products", "platform": "Marty Cagan", "duration": "Self-paced", "type": "Book"},
        ],
    },
    "DevOps Engineer": {
        "icon": "⚙️", "color": "#818cf8",
        "description": "Bridge development and operations by automating pipelines and managing infrastructure.",
        "avgSalary": "₹7–20 LPA", "growthRate": "+24% by 2030",
        "requiredSkills": ["Programming", "Networking", "Problem Solving", "Analytical Thinking"],
        "requiredLevels": {"programming": 7, "networking": 7, "problemSolving": 7},
        "learning": [
            {"title": "DevOps Bootcamp – Nana Janashia", "platform": "TechWorld with Nana", "duration": "3 months", "type": "Course"},
            {"title": "Docker & Kubernetes Complete Guide", "platform": "Udemy", "duration": "6 weeks", "type": "Course"},
            {"title": "GitHub Actions CI/CD Pipeline", "platform": "GitHub", "duration": "2 weeks", "type": "Project"},
            {"title": "CKA – Certified Kubernetes Admin", "platform": "Linux Foundation", "duration": "2 months", "type": "Certification"},
        ],
    },
    "Business Analyst": {
        "icon": "📈", "color": "#fbbf24",
        "description": "Analyze business processes and translate complex requirements into actionable solutions.",
        "avgSalary": "₹5–15 LPA", "growthRate": "+14% by 2030",
        "requiredSkills": ["Communication", "Analytical Thinking", "Data Analysis", "Problem Solving"],
        "requiredLevels": {"communication": 7, "analyticalThinking": 7, "dataAnalysis": 6},
        "learning": [
            {"title": "Business Analysis Fundamentals", "platform": "Udemy", "duration": "6 weeks", "type": "Course"},
            {"title": "CBAP – BA Certification", "platform": "IIBA", "duration": "3 months", "type": "Certification"},
            {"title": "Excel & Power BI for Business", "platform": "Microsoft Learn", "duration": "1 month", "type": "Course"},
            {"title": "Requirements Analysis Project", "platform": "Self-directed", "duration": "1 month", "type": "Project"},
        ],
    },
    "Network Engineer": {
        "icon": "🌐", "color": "#2dd4bf",
        "description": "Design, implement and maintain robust network infrastructure for organizations.",
        "avgSalary": "₹5–16 LPA", "growthRate": "+15% by 2030",
        "requiredSkills": ["Networking", "Problem Solving", "Analytical Thinking", "Communication"],
        "requiredLevels": {"networking": 8, "problemSolving": 6, "analyticalThinking": 6},
        "learning": [
            {"title": "CCNA – Cisco Certified Network Associate", "platform": "Cisco / Udemy", "duration": "3 months", "type": "Certification"},
            {"title": "Network+ Certification", "platform": "CompTIA", "duration": "2 months", "type": "Certification"},
            {"title": "Packet Tracer Lab Projects", "platform": "Cisco NetAcad", "duration": "1 month", "type": "Project"},
            {"title": "CompTIA A+ Foundation", "platform": "CompTIA", "duration": "1.5 months", "type": "Course"},
        ],
    },
}

SKILL_KEY_MAP = {
    "programming": "Programming", "mathematics": "Mathematics",
    "communication": "Communication", "problemSolving": "Problem Solving",
    "creativity": "Creativity", "leadership": "Leadership",
    "analyticalThinking": "Analytical Thinking", "teamwork": "Teamwork",
    "dataAnalysis": "Data Analysis", "networking": "Networking",
}

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Smart Career Advisor API",
        "models": {"Random Forest": rf_acc, "KNN": knn_acc, "SVM": svm_acc}
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        skills = data.get("skills", {})
        aptitude = data.get("aptitude", {})

        features = np.array([[
            skills.get("programming", 5),
            skills.get("mathematics", 5),
            skills.get("communication", 5),
            skills.get("problemSolving", 5),
            skills.get("creativity", 5),
            skills.get("leadership", 5),
            skills.get("analyticalThinking", 5),
            skills.get("teamwork", 5),
            skills.get("dataAnalysis", 5),
            skills.get("networking", 5),
            aptitude.get("logical", 60),
            aptitude.get("verbal", 60),
            aptitude.get("quantitative", 60),
        ]])

        features_scaled = scaler.transform(features)

        # Get probabilities from all 3 models
        rf_proba  = rf.predict_proba(features_scaled)[0]
        knn_proba = knn.predict_proba(features_scaled)[0]
        svm_proba = svm.predict_proba(features_scaled)[0]

        # Ensemble: RF 45%, KNN 30%, SVM 25%
        ensemble_proba = 0.45 * rf_proba + 0.30 * knn_proba + 0.25 * svm_proba

        # Get top 5 careers
        top5_idx = np.argsort(ensemble_proba)[::-1][:5]
        careers = le.classes_

        results = []
        for idx in top5_idx:
            career_name = careers[idx]
            match_pct = round(float(ensemble_proba[idx]) * 100, 1)
            match_pct = max(match_pct, 35)  # floor at 35%

            meta = CAREER_META.get(career_name, {})

            # Skill gap analysis
            skill_gaps = []
            for sk_key, req_level in meta.get("requiredLevels", {}).items():
                current = skills.get(sk_key, 5)
                if current < req_level:
                    gap = req_level - current
                    skill_gaps.append({
                        "skill": SKILL_KEY_MAP.get(sk_key, sk_key),
                        "current": current,
                        "required": req_level,
                        "priority": "High" if gap >= 4 else "Medium" if gap >= 2 else "Low"
                    })

            # Dominant model
            rf_contrib  = 0.45 * rf_proba[idx]
            knn_contrib = 0.30 * knn_proba[idx]
            svm_contrib = 0.25 * svm_proba[idx]
            if knn_contrib >= rf_contrib and knn_contrib >= svm_contrib:
                model = "KNN"
            elif svm_contrib >= rf_contrib:
                model = "SVM"
            else:
                model = "Random Forest"

            results.append({
                "title": career_name,
                "match": match_pct,
                "description": meta.get("description", ""),
                "icon": meta.get("icon", "💼"),
                "color": meta.get("color", "#888"),
                "avgSalary": meta.get("avgSalary", "N/A"),
                "growthRate": meta.get("growthRate", "N/A"),
                "requiredSkills": meta.get("requiredSkills", []),
                "skillGaps": skill_gaps,
                "learningPath": meta.get("learning", []),
                "model": model,
                "modelScores": {
                    "Random Forest": round(float(rf_proba[idx]) * 100, 1),
                    "KNN": round(float(knn_proba[idx]) * 100, 1),
                    "SVM": round(float(svm_proba[idx]) * 100, 1),
                }
            })

        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on http://localhost:{port}")
    serve(app, host="0.0.0.0", port=port)
