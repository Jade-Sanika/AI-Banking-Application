# AI-Banking-Application


# VYOM+: AI-Powered Banking Application

VYOM+ is an AI-powered web-based banking application designed to enhance the user experience of Union Bank's existing application by improving security, personalization, and usability using modern AI technologies and design practices.

---

## 🔥 Key Features

### 🔐 2FA Authentication
- Combines **Email + Password + Face Recognition** for secure login.
- Face embeddings are stored securely in an SQLite database.
- OTP is sent to registered mobile number after successful face match.

### 🧠 AI-Powered Personalized Chatbot
- Built using **Google Cloud Platform's Vertex AI**.
- Uses **RAG (Retrieval-Augmented Generation)** to respond to user queries by searching Union Bank's FAQs (PDFs stored in GCP Datastore).
- Supports:
  - Voice Input
  - Balance Check
  - Recent Transactions
  - Appointment Booking
  - Loan Options
  - Financial Insights
  - Ask any question

### 🏦 Dashboard Modules
Includes an interactive sidebar with the following sections:
- Accounts
- Loan Approval
- Transfers
- Payments
- Statements
- Investments
- Alerts
- Fraud Detection
- Ask Query
- Settings

Each section routes to its respective page with meaningful UI/UX.

### ✅ Loan Approval System
- Inputs: Loan Amount, Income, Employment Status, Purpose
- Outputs:
  - Loan Eligibility
  - Risk % Score
- Uses trained logic to simulate loan risk and approval.

### 🔍 Fraud Detection System
- Trained using custom dataset of 5000 transactions.
- Models used: Logistic Regression, Random Forest, **XGBoost (Best Accuracy)**.
- Classifies transaction as **Fraudulent** or **Legitimate**.

### 📊 Ask Query + Analytical Snapshot
- Accepts user queries and routes them to the relevant department.
- Allows **download of analytical snapshot** (Account + CIBIL Score) to assist officials.

### ⚙️ Settings
- Option to Logout and redirect to secure login page.

---

## 💡 Why VYOM+?

The legacy Union Bank application suffered from:
- Poor UI with inconsistent themes
- Weak Authentication
- Non-functional Chatbot
- Lack of AI features and personalization
- No system for query routing or analytical insights

VYOM+ addresses all these gaps with:
- Secure, face-recognition-based login
- Personalized AI chatbot
- Voice-enabled assistance
- Intuitive, modern UI/UX
- Loan and Fraud Analysis with ML Models
- Automated routing and analytics for queries

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Database**: SQLite
- **AI/ML**: Scikit-learn, XGBoost
- **Cloud**: Google Cloud Platform (Vertex AI, Datastore)
- **Security**: Face Recognition, OTP-based 2FA
- **Voice Input**: Web Speech API

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Flask
- face_recognition
- scikit-learn
- xgboost
- Google Cloud SDK (for chatbot)

### Installation

```bash
git clone https://github.com/Jade-Sanika/AI-Banking-Application.git
cd AI-Banking-Application
pip install -r requirements.txt
python app.py
