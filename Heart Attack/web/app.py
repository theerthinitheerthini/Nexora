# -*- coding: utf-8 -*-
# ============================================================
# CardioShield — AI-Powered Early Warning System for
# Cardiovascular Risk Assessment & Personalized Health Advisory
# ============================================================

import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import io

# =============================
# CONFIG
# =============================
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "model_pipeline.joblib")

MYSQL_USER = "root"
MYSQL_PASS = "Dora%4020"
MYSQL_HOST = "localhost"
MYSQL_DB   = "heart_attack_db"

app = Flask(__name__)
app.secret_key = "cardioshield_secret_2026"

app.config["SQLALCHEMY_DATABASE_URI"] = \
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}/{MYSQL_DB}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db           = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# =============================
# DATABASE MODELS
# =============================
class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120))
    email         = db.Column(db.String(180), unique=True)
    password_hash = db.Column(db.String(255))
    is_admin      = db.Column(db.Boolean, default=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


class LoginHistory(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"))
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    user       = db.relationship("User", backref=db.backref("login_history", lazy=True))


class Prediction(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("user.id"))
    inputs_json    = db.Column(db.Text)
    predicted_risk = db.Column(db.String(20))
    probability    = db.Column(db.Float)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()


# =============================
# LOAD ML MODEL
# =============================
pipeline = joblib.load(MODEL_PATH)


# =============================
# RISK SCORING ENGINE — 3 Levels
# LOW: 0-4 | MEDIUM: 5-9 | HIGH: 10+
# =============================
def calculate_risk_score(form):
    score = 0

    def safe_float(v):
        try: return float(v)
        except: return 0

    age         = safe_float(form.get("age"))
    bp          = safe_float(form.get("trestbps"))
    chol        = safe_float(form.get("chol"))
    thalach     = safe_float(form.get("thalach"))
    oldpeak     = safe_float(form.get("oldpeak"))
    smoker      = safe_float(form.get("Smoker"))
    stress      = safe_float(form.get("Stress_Level"))
    exercise    = safe_float(form.get("Exercise"))
    step_count  = safe_float(form.get("Step count_stress"))
    cp          = safe_float(form.get("cp"))
    wt_loss     = safe_float(form.get("Sudden_Weight_Loss"))

    if age > 60:        score += 2
    elif age > 45:      score += 1

    if bp > 180:        score += 3
    elif bp > 140:      score += 2
    elif bp > 120:      score += 1

    if chol > 300:      score += 3
    elif chol > 240:    score += 2
    elif chol > 200:    score += 1

    if thalach < 90:    score += 2

    if oldpeak > 3:     score += 3
    elif oldpeak > 2:   score += 2
    elif oldpeak > 1:   score += 1

    if smoker == 1:     score += 2

    if stress > 8:      score += 3
    elif stress > 6:    score += 2
    elif stress > 4:    score += 1

    if exercise < 1:    score += 2
    elif exercise < 3:  score += 1

    if step_count < 3000:   score += 2
    elif step_count < 6000: score += 1

    if cp == 3:   score += 3
    elif cp == 2: score += 2
    elif cp == 1: score += 1

    if wt_loss == 1: score += 2

    return score


# =============================
# ADVICE ENGINE — 3 Risk Levels
# =============================
def build_advice(risk):
    lifestyle = {
        "LOW": [
            "Maintain a balanced and nutritious diet daily.",
            "Walk or exercise for at least 30 minutes every day.",
            "Drink 8 glasses of water daily for hydration.",
            "Sleep 7 to 8 hours regularly each night.",
            "Limit salt and reduce sugar intake.",
            "Manage stress through meditation or yoga.",
            "Avoid smoking and passive smoke exposure.",
            "Maintain a healthy body weight consistently.",
            "Schedule an annual routine health checkup.",
            "Stay socially active and mentally positive."
        ],
        "MEDIUM": [
            "Reduce salt intake immediately and avoid processed foods.",
            "Avoid fried, oily and high-cholesterol foods.",
            "Exercise for at least 40 minutes every day.",
            "Monitor blood pressure at least once per week.",
            "Limit caffeine consumption and avoid alcohol.",
            "Increase dietary fiber intake through vegetables.",
            "Practice stress management and relaxation techniques.",
            "Improve your sleep schedule to 7-8 hours daily.",
            "Check cholesterol levels regularly with your doctor.",
            "Avoid sedentary lifestyle — stay physically active."
        ],
        "HIGH": [
            "Follow a strict low-salt and low-fat diet immediately.",
            "Stop smoking completely without any delay.",
            "Avoid alcohol consumption entirely.",
            "Limit physical activity — light walks only with caution.",
            "Check blood pressure every single day.",
            "Avoid junk food, heavy meals, and processed snacks.",
            "Reduce stress urgently — seek medical counselling.",
            "Monitor all symptoms carefully and track daily health.",
            "Stay well hydrated but avoid excess fluid intake.",
            "Seek immediate medical attention without delay."
        ]
    }

    doctor = {
        "LOW": [
            "Schedule a routine yearly health checkup with GP.",
            "Monitor cholesterol and blood sugar levels annually.",
            "Consult a doctor immediately if any symptoms appear.",
            "Family history cardiovascular screening recommended.",
            "Maintain regular visits to your general physician."
        ],
        "MEDIUM": [
            "Consult a physician within the next 30 days.",
            "Get a complete lipid profile blood test done.",
            "An ECG may be recommended — consult your doctor.",
            "Discuss a blood pressure management and diet plan.",
            "Consider a dietitian or cardiologist consultation."
        ],
        "HIGH": [
            "Consult a cardiologist within 72 hours urgently.",
            "ECG, stress test, and ECHO are required immediately.",
            "Medication review and adjustment with your doctor.",
            "Complete lipid profile and full blood work mandatory.",
            "Go to emergency immediately if chest pain occurs."
        ]
    }

    return lifestyle.get(risk, []), doctor.get(risk, [])


# =============================
# PDF REPORT GENERATOR
# =============================
def generate_pdf_report(user_name, risk, probability, tips, doctor, created_at):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=inch, leftMargin=inch,
                               topMargin=inch, bottomMargin=inch)
    styles  = getSampleStyleSheet()
    content = []

    # Title
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        fontSize=20, textColor=colors.HexColor('#0a1628'), spaceAfter=6
    )
    content.append(Paragraph("CardioShield — Cardiovascular Risk Assessment Report", title_style))
    content.append(Spacer(1, 0.1 * inch))

    sub_style = ParagraphStyle(
        'Sub', parent=styles['Normal'],
        fontSize=11, textColor=colors.grey
    )
    content.append(Paragraph(
        f"Generated on: {created_at} &nbsp;|&nbsp; Patient: {user_name}",
        sub_style
    ))
    content.append(Spacer(1, 0.3 * inch))

    # Risk Level Box
    risk_colors = {
        "LOW":    colors.HexColor('#d4edda'),
        "MEDIUM": colors.HexColor('#fff3cd'),
        "HIGH":   colors.HexColor('#f8d7da'),
    }
    risk_text_colors = {
        "LOW":    colors.HexColor('#155724'),
        "MEDIUM": colors.HexColor('#856404'),
        "HIGH":   colors.HexColor('#721c24'),
    }

    risk_data = [[
        Paragraph(f"<b>Risk Level: {risk}</b>", ParagraphStyle(
            'RiskText', fontSize=14,
            textColor=risk_text_colors.get(risk, colors.black)
        )),
        Paragraph(
            f"<b>Confidence: {probability}%</b>" if probability else "<b>Confidence: N/A</b>",
            ParagraphStyle('ProbText', fontSize=12, textColor=colors.HexColor('#1a2e4a'))
        )
    ]]

    risk_table = Table(risk_data, colWidths=[3.5 * inch, 3.5 * inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), risk_colors.get(risk, colors.white)),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#e8f0fe')),
        ('BOX',        (0, 0), (-1, -1), 1.5, colors.HexColor('#cccccc')),
        ('INNERGRID',  (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING',    (0, 0), (-1, -1), 12),
    ]))
    content.append(risk_table)
    content.append(Spacer(1, 0.3 * inch))

    section_style = ParagraphStyle(
        'Section', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor('#1d4ed8'),
        spaceBefore=12, spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet', parent=styles['Normal'],
        fontSize=10, leftIndent=15, spaceAfter=4,
        textColor=colors.HexColor('#333333')
    )

    content.append(Paragraph("Lifestyle Recommendations", section_style))
    for tip in tips:
        content.append(Paragraph(f"• {tip}", bullet_style))

    content.append(Spacer(1, 0.2 * inch))
    content.append(Paragraph("Doctor Consultation Suggestions", section_style))
    for d in doctor:
        content.append(Paragraph(f"• {d}", bullet_style))

    content.append(Spacer(1, 0.4 * inch))

    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontSize=9, textColor=colors.grey, alignment=1
    )
    content.append(Paragraph(
        "This report is generated by CardioShield — AI-Powered Early Warning System. "
        "Please consult a qualified medical professional for diagnosis and treatment.",
        footer_style
    ))

    doc.build(content)
    buffer.seek(0)
    return buffer


# =============================
# ROUTES
# =============================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form["name"]
        email    = request.form["email"]
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            flash("Email already exists. Please use a different email.", "error")
            return redirect(url_for("register"))

        user = User(
            name=name, email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully! Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password. Please try again.", "error")
            return redirect(url_for("login"))

        login_user(user)
        history = LoginHistory(user_id=user.id)
        db.session.add(history)
        db.session.commit()

        if user.is_admin:
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    pred_count = Prediction.query.filter_by(user_id=current_user.id).count()
    return render_template("dashboard.html", pred_count=pred_count)


@app.route("/history")
@login_required
def history():
    records = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", records=records)


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("Unauthorized access!", "error")
        return redirect(url_for("dashboard"))

    total_users       = User.query.count()
    total_predictions = Prediction.query.count()

    users_data = []
    for user in User.query.all():
        pred_count = Prediction.query.filter_by(user_id=user.id).count()
        last_login = LoginHistory.query.filter_by(
            user_id=user.id
        ).order_by(LoginHistory.login_time.desc()).first()
        users_data.append({
            "id": user.id, "name": user.name, "email": user.email,
            "pred_count": pred_count,
            "last_login": last_login.login_time if last_login else "Never"
        })

    recent_logins = LoginHistory.query\
        .order_by(LoginHistory.login_time.desc()).limit(10).all()

    prediction_history = db.session.query(Prediction, User)\
        .join(User, Prediction.user_id == User.id)\
        .order_by(Prediction.created_at.desc()).all()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        users_data=users_data,
        recent_logins=recent_logins,
        prediction_history=prediction_history
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    expected_cols = list(pipeline.named_steps["preprocess"].feature_names_in_)
    data_dict     = {col: np.nan for col in expected_cols}

    for col in expected_cols:
        val = request.form.get(col)
        if val:
            try:    data_dict[col] = float(val)
            except: data_dict[col] = val

    X_new   = pd.DataFrame([data_dict])
    ml_pred = pipeline.predict(X_new)[0]

    # 3-level risk scoring
    risk_score = calculate_risk_score(request.form)

    if risk_score <= 4:
        risk = "LOW"
    elif risk_score <= 9:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    probability = None
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs       = pipeline.predict_proba(X_new)[0]
        probability = round(np.max(probs) * 100, 2)

    tips, doctor = build_advice(risk)

    record = Prediction(
        user_id=current_user.id,
        inputs_json=json.dumps(request.form.to_dict()),
        predicted_risk=risk,
        probability=probability
    )
    db.session.add(record)
    db.session.commit()

    return render_template(
        "result.html",
        risk=risk,
        probability=probability,
        tips=tips,
        doctor=doctor,
        user_name=current_user.name,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        risk_score=risk_score
    )


@app.route("/download_report", methods=["POST"])
@login_required
def download_report():
    risk        = request.form.get("risk")
    probability = request.form.get("probability")
    created_at  = request.form.get("created_at")

    tips, doctor = build_advice(risk)

    pdf_buffer = generate_pdf_report(
        user_name=current_user.name,
        risk=risk,
        probability=probability,
        tips=tips,
        doctor=doctor,
        created_at=created_at
    )

    response = make_response(pdf_buffer.read())
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = \
        f"attachment; filename=CardioShield_Risk_Report_{current_user.name}.pdf"
    return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")