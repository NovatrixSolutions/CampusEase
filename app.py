# =============================================================================
# Campus Ease V2 - AI Placement Prediction with Skills & Role Recommendation
# =============================================================================

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from groq import Groq

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

# Groq API setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# =============================================================================
# LOAD MODEL & SCALER
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "placement_model.pkl")
KERAS_PATH = os.path.join(BASE_DIR, "placement_model.keras")

with open(PKL_PATH, "rb") as f:
    meta = pickle.load(f)

scaler = meta["scaler"]
FEATURE_NAMES = meta["features"]
model = load_model(KERAS_PATH)

print("Model loaded successfully")
print("Features:", FEATURE_NAMES)

# =============================================================================
# ROLE DATABASE & SKILL MATCHING
# =============================================================================

ROLE_DATABASE = {
    "Backend Developer": {
        "required_skills": ["python", "java", "node", "nodejs", "api", "rest", "sql", "database", "mongodb", "postgresql"],
        "preferred_skills": ["docker", "kubernetes", "microservices", "redis", "kafka"],
        "min_technical_score": 60,
        "min_cgpa": 6.5
    },
    "Full Stack Developer": {
        "required_skills": ["javascript", "react", "node", "nodejs", "html", "css", "api", "sql"],
        "preferred_skills": ["typescript", "mongodb", "express", "angular", "vue"],
        "min_technical_score": 65,
        "min_cgpa": 7.0
    },
    "Data Scientist": {
        "required_skills": ["python", "machine learning", "ml", "statistics", "pandas", "numpy", "sql"],
        "preferred_skills": ["tensorflow", "pytorch", "deep learning", "nlp", "computer vision"],
        "min_technical_score": 70,
        "min_cgpa": 7.5
    },
    "DevOps Engineer": {
        "required_skills": ["linux", "docker", "kubernetes", "ci/cd", "jenkins", "git"],
        "preferred_skills": ["aws", "azure", "terraform", "ansible", "prometheus"],
        "min_technical_score": 65,
        "min_cgpa": 7.0
    },
    "Frontend Developer": {
        "required_skills": ["javascript", "react", "html", "css", "responsive"],
        "preferred_skills": ["typescript", "redux", "webpack", "sass", "tailwind"],
        "min_technical_score": 60,
        "min_cgpa": 6.5
    },
    "Mobile Developer": {
        "required_skills": ["android", "ios", "react native", "flutter", "kotlin", "swift"],
        "preferred_skills": ["firebase", "api", "ui/ux"],
        "min_technical_score": 65,
        "min_cgpa": 7.0
    },
    "QA Engineer": {
        "required_skills": ["testing", "selenium", "automation", "api testing"],
        "preferred_skills": ["jmeter", "postman", "cypress", "junit"],
        "min_technical_score": 55,
        "min_cgpa": 6.0
    },
    "Cloud Engineer": {
        "required_skills": ["aws", "azure", "gcp", "cloud", "networking"],
        "preferred_skills": ["terraform", "kubernetes", "serverless", "lambda"],
        "min_technical_score": 70,
        "min_cgpa": 7.5
    }
}

def extract_skill_features(skills_text):
    """Convert free-text skills to binary features"""
    skills_lower = skills_text.lower()
    
    # Core skills
    python_skill = 1 if any(k in skills_lower for k in ["python", "py"]) else 0
    dsa_skill = 1 if any(k in skills_lower for k in ["dsa", "data structure", "algorithm", "algorithms"]) else 0
    sql_skill = 1 if any(k in skills_lower for k in ["sql", "database", "mysql", "postgresql", "mongodb"]) else 0
    
    # Additional skill count (for scoring)
    skill_keywords = ["java", "javascript", "react", "node", "angular", "vue", "docker", 
                     "kubernetes", "aws", "azure", "gcp", "machine learning", "ml", "ai",
                     "tensorflow", "pytorch", "git", "linux", "api", "rest", "c++", "c",
                     "devops", "blockchain", "cybersecurity", "frontend", "backend"]
    
    additional_skills_count = sum(1 for kw in skill_keywords if kw in skills_lower)
    
    print(f"\nSkill Detection Debug:")
    print(f"  Python: {python_skill}")
    print(f"  DSA: {dsa_skill}")
    print(f"  SQL: {sql_skill}")
    print(f"  Additional Skills Count: {additional_skills_count}")
    
    return {
        "Python_Skill": python_skill,
        "DSA_Skill": dsa_skill,
        "SQL_Skill": sql_skill,
        "Additional_Skills_Count": min(additional_skills_count, 10)
    }

def recommend_roles(skills_text, cgpa, technical_score, top_n=3):
    """Recommend suitable roles based on skills and profile"""
    skills_lower = skills_text.lower()
    recommendations = []
    
    for role, criteria in ROLE_DATABASE.items():
        score = 0
        
        # Check required skills
        required_match = sum(1 for skill in criteria["required_skills"] if skill in skills_lower)
        required_total = len(criteria["required_skills"])
        required_percentage = (required_match / required_total) * 100
        
        # Check preferred skills
        preferred_match = sum(1 for skill in criteria["preferred_skills"] if skill in skills_lower)
        preferred_total = len(criteria["preferred_skills"])
        preferred_percentage = (preferred_match / preferred_total) * 100 if preferred_total > 0 else 0
        
        # Calculate match score
        score = (required_percentage * 0.7) + (preferred_percentage * 0.3)
        
        # Check minimum criteria
        meets_criteria = (
            cgpa >= criteria["min_cgpa"] and 
            technical_score >= criteria["min_technical_score"]
        )
        
        if required_match > 0:  # At least one required skill matches
            recommendations.append({
                "role": role,
                "match_score": round(score, 1),
                "required_skills_match": f"{required_match}/{required_total}",
                "preferred_skills_match": f"{preferred_match}/{preferred_total}",
                "meets_criteria": meets_criteria,
                "reason": f"You have {required_match} out of {required_total} required skills"
            })
    
    # Sort by match score
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    
    return recommendations[:top_n]

def get_ai_recommendations(data, skills_text, target_role, prediction, probability):
    """Get AI-powered recommendations using Groq"""
    try:
        prompt = f"""You are a career counselor for college students. Analyze this student profile and provide 3-5 specific, actionable recommendations to improve their placement chances.

Student Profile:
- CGPA: {data.get('CGPA', 0)}
- Technical Test Score: {data.get('Technical_Test_Score', 0)}/100
- Skills: {skills_text}
- Target Role: {target_role if target_role else 'Not specified'}
- Projects Completed: {data.get('Projects_Completed', 0)}
- Internship Experience: {'Yes' if data.get('Internship_Experience', 0) == 1 else 'No'}
- Communication Skills: {data.get('Communication_Skills', 0)}/10
- Certifications: {data.get('Certifications_Count', 0)}
- Hackathons: {data.get('Hackathons_Participated', 0)}

Placement Prediction: {prediction} ({probability}%)

Provide recommendations as a numbered list. Focus on:
1. Skill gaps for their target role
2. Practical steps to improve weak areas
3. Resources or platforms to use
4. Timeline suggestions

Keep each recommendation under 25 words. Be specific and actionable."""

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Parse numbered list
        recommendations = []
        for line in ai_response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering
                clean_line = line.lstrip('0123456789.-•) ').strip()
                if clean_line:
                    recommendations.append(clean_line)
        
        return recommendations if recommendations else [ai_response]
    
    except Exception as e:
        print(f"Groq API Error: {e}")
        return build_recommendations_fallback(data, skills_text, target_role)

def build_recommendations_fallback(data, skills_text, target_role):
    """Fallback recommendations if Groq API fails"""
    recommendations = []
    
    if float(data.get("Technical_Test_Score", 0)) < 60:
        recommendations.append("Practice coding on LeetCode, HackerRank daily for 1-2 hours")
    
    if int(data.get("Communication_Skills", 0)) < 6:
        recommendations.append("Join Toastmasters or practice mock interviews weekly")
    
    if int(data.get("Internship_Experience", 0)) == 0:
        recommendations.append("Apply for internships on LinkedIn, Internshala immediately")
    
    if int(data.get("Projects_Completed", 0)) < 2:
        recommendations.append("Build 2-3 projects and host on GitHub with documentation")
    
    if float(data.get("CGPA", 0)) < 6.5:
        recommendations.append("Focus on improving grades in remaining semesters")
    
    if target_role and target_role in ROLE_DATABASE:
        role_criteria = ROLE_DATABASE[target_role]
        skills_lower = skills_text.lower()
        missing_required = [s for s in role_criteria["required_skills"] if s not in skills_lower]
        if missing_required:
            recommendations.append(f"Learn {', '.join(missing_required[:3])} for {target_role}")
    
    if not recommendations:
        recommendations.append("Excellent profile! Stay updated with latest tech trends")
    
    return recommendations

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "service": "Campus Ease V2 - Skills & Role Based Prediction",
        "version": "2.0.0"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    
    Request Body:
    {
        "IQ": 110,
        "Prev_Sem_Result": 8.5,
        "CGPA": 8.2,
        "Academic_Performance": 8,
        "Internship_Experience": 1,
        "Extra_Curricular_Score": 7,
        "Communication_Skills": 8,
        "Projects_Completed": 3,
        "Technical_Test_Score": 72.5,
        "Certifications_Count": 2,
        "Hackathons_Participated": 1,
        "Soft_Skills_Score": 7,
        "Resume_Score": 8,
        "Skills": "Python, Java, SQL, REST API, Docker, Git",
        "Target_Role": "Backend Developer"
    }
    """
    
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400
    
    # Extract skills
    skills_text = data.get("Skills", "")
    target_role = data.get("Target_Role", "")
    
    if not skills_text:
        return jsonify({"error": "Skills field is required"}), 422
    
    # Extract skill features
    skill_features = extract_skill_features(skills_text)
    
    # Debug logging
    print("\n" + "="*50)
    print("PREDICTION REQUEST")
    print("="*50)
    print(f"Skills Text: {skills_text}")
    print(f"Extracted Features: {skill_features}")
    print(f"CGPA: {data.get('CGPA', 0)}")
    print(f"IQ: {data.get('IQ', 0)}")
    print(f"Technical Score: {data.get('Technical_Test_Score', 0)}")
    print(f"Projects: {data.get('Projects_Completed', 0)}")
    print(f"Internship: {data.get('Internship_Experience', 0)}")
    
    # Build feature vector
    feature_values = []
    for feat in FEATURE_NAMES:
        if feat in skill_features:
            feature_values.append(skill_features[feat])
        elif feat == "Internship_Experience":
            feature_values.append(int(data.get(feat, 0)))
        else:
            feature_values.append(float(data.get(feat, 0)))
    
    # Create DataFrame
    input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    
    # Scale
    input_scaled = scaler.transform(input_df)
    input_3d = input_scaled.reshape(1, 1, len(FEATURE_NAMES))
    
    # Predict with hybrid scoring
    raw_prob = float(model.predict(input_3d, verbose=0)[0][0])
    
    # Calculate manual score for realistic probabilities
    manual_score = 0.0
    
    # CGPA contribution (0-25 points)
    cgpa = float(data.get('CGPA', 0))
    if cgpa >= 8.5:
        manual_score += 25
    elif cgpa >= 7.5:
        manual_score += 20
    elif cgpa >= 6.5:
        manual_score += 15
    elif cgpa >= 5.5:
        manual_score += 10
    else:
        manual_score += 5
    
    # Technical Score contribution (0-25 points)
    tech_score = float(data.get('Technical_Test_Score', 0))
    manual_score += (tech_score / 100) * 25
    
    # Skills contribution (0-20 points)
    skills_score = (skill_features['Python_Skill'] * 5 + 
                   skill_features['DSA_Skill'] * 5 + 
                   skill_features['SQL_Skill'] * 5 + 
                   min(skill_features['Additional_Skills_Count'], 5))
    manual_score += skills_score
    
    # Internship (0-15 points)
    if int(data.get('Internship_Experience', 0)) == 1:
        manual_score += 15
    
    # Projects (0-10 points)
    projects = int(data.get('Projects_Completed', 0))
    manual_score += min(projects * 3, 10)
    
    # Communication (0-5 points)
    comm = int(data.get('Communication_Skills', 0))
    manual_score += (comm / 10) * 5
    
    # Convert to percentage (0-100)
    manual_percentage = min(manual_score, 100)
    
    # Blend model prediction with manual score (60% manual, 40% model)
    # This makes it more realistic and forgiving
    blended_percentage = (manual_percentage * 0.6) + (raw_prob * 100 * 0.4)
    percentage = round(blended_percentage, 2)
    prediction = "Placed" if percentage >= 50 else "Not Placed"
    
    # Debug logging
    print(f"\nRaw Model Probability: {raw_prob}")
    print(f"Manual Score: {manual_percentage}%")
    print(f"Blended Percentage: {percentage}%")
    print(f"Prediction: {prediction}")
    print(f"Feature Vector: {feature_values}")
    print("="*50 + "\n")
    
    # Role recommendations
    role_recommendations = recommend_roles(
        skills_text, 
        float(data.get("CGPA", 0)), 
        float(data.get("Technical_Test_Score", 0))
    )
    
    # AI-powered recommendations
    ai_recommendations = get_ai_recommendations(data, skills_text, target_role, prediction, percentage)
    
    response = {
        "probability": f"{percentage}%",
        "prediction": prediction,
        "target_role": target_role if target_role else "Not specified",
        "recommended_roles": role_recommendations,
        "recommendations": ai_recommendations,
        "skills_detected": skill_features
    }
    
    return jsonify(response), 200

@app.route("/roles", methods=["GET"])
def get_roles():
    """Get all available roles and their requirements"""
    roles_info = []
    for role, criteria in ROLE_DATABASE.items():
        roles_info.append({
            "role": role,
            "required_skills": criteria["required_skills"],
            "preferred_skills": criteria["preferred_skills"],
            "min_technical_score": criteria["min_technical_score"],
            "min_cgpa": criteria["min_cgpa"]
        })
    return jsonify({"roles": roles_info}), 200

@app.route("/chat", methods=["POST"])
def chat():
    """AI chat for career guidance"""
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data["message"]
    context = data.get("context", {})
    
    try:
        system_prompt = """You are a career counselor and placement expert for college students. 
Provide helpful, specific advice about:
- Career paths and role selection
- Skill development and learning resources
- Interview preparation
- Resume building
- Placement strategies
- Technical skill gaps

Keep responses concise (under 150 words), actionable, and encouraging."""

        messages = [{"role": "system", "content": system_prompt}]
        
        if context:
            context_msg = f"Student context: CGPA: {context.get('cgpa', 'N/A')}, Skills: {context.get('skills', 'N/A')}, Target Role: {context.get('target_role', 'N/A')}"
            messages.append({"role": "system", "content": context_msg})
        
        messages.append({"role": "user", "content": user_message})
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=300
        )
        
        ai_response = response.choices[0].message.content.strip()
        return jsonify({"response": ai_response}), 200
    
    except Exception as e:
        return jsonify({"error": f"AI chat failed: {str(e)}"}), 500

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "details": str(e)}), 500

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Campus Ease - Running on Render")
    print("=" * 55)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
