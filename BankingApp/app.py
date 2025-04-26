# Core Flask & Utilities
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from werkzeug.utils import secure_filename
from jinja2.exceptions import TemplateNotFound
import os
import re
import uuid
import string
import random
import json

# PDF & File Handling
from pdf_generator import generate_pdf
import mimetypes

# LangChain + Vertex AI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai import VertexAI
from langchain.memory import ConversationBufferMemory

# Environment Variables
from dotenv import load_dotenv

# Text-to-Speech
from gtts import gTTS
from pydub import AudioSegment

# Face Recognition
import face_recognition
import numpy as np

# Image Processing
from PIL import Image
import base64
import io

# Authentication & Security
import bcrypt

# Date & Time
from datetime import datetime, timedelta

# HTTP Requests
import requests

# SMS & Email Services
from twilio.rest import Client
from flask_mail import Mail, Message

# Google Cloud Translate
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account

# Machine Learning
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import pandas as pd
import joblib


from database import db, User, OTP, Appointment, Transaction, ChatMessage
load_dotenv()


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# Set Google Application Credentials for Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', r'C:\Users\sanik\OneDrive - Vidyalankar Institute of Technology\Desktop\BankingApp\mimetic-setup-456806-e4-0fd6ba61e7b5.json')


# # Configure email
# app.config['MAIL_SERVER'] = 'smtp.example.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your_email@example.com')
# app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your_email_password')
# app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@bankingapp.com')

# Configure Twilio for SMS
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Flask extensions
db.init_app(app)
# mail = Mail(app)

# Vertex AI configuration
MODEL = os.getenv('MODEL')
DATA_STORE_ID = os.getenv('DATA_STORE_ID')
DATA_STORE_LOCATION = os.getenv('DATA_STORE_LOCATION')
PROJECT_ID = os.getenv('PROJECT_ID')
SOURCE = eval(os.getenv("SOURCE", "True"))
AUDIO = eval(os.getenv("AUDIO", "True"))

# In-memory session-based chat history
user_chains = {}

# Get the credentials path from the environment
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Load credentials and initialize Translate client
credentials = service_account.Credentials.from_service_account_file(key_path)
translate_client = translate.Client(credentials=credentials)

# Initialize fraud detection model
# fraud_model = IsolationForest(
#     n_estimators=200,
#     max_samples='auto',
#     contamination=0.01,
#     max_features=1.0,
#     random_state=42,
#     n_jobs=-1
# ) 

# Load models
models = {
    'Logistic Regression': joblib.load('models/logistic_regression_model.pkl'),
    'Random Forest': joblib.load('models/random_forest_model.pkl'),
    'XGBoost': joblib.load('models/xgboost_model.pkl')
}

# Sample confusion matrices (you can update these dynamically if needed)
confusion_matrices = {
    'Logistic Regression': [[970, 30], [20, 80]],
    'Random Forest': [[990, 10], [5, 95]],
    'XGBoost': [[992, 8], [3, 97]]
}

# Initialize loan risk assessment model
loan_risk_model = RandomForestClassifier(n_estimators=100, max_depth=10)



# Create directories for storing PDFs
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'pdfs')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Create templates directory if not exists
templates_dir = os.path.join(os.getcwd(), 'templates')
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

# Define departments
DEPARTMENTS = {
    'accounts': 'Accounts Department - Manages savings, current accounts, and account creation details.',
    'loans': 'Loans Department - Handles loan applications, approvals, EMIs, and interest tracking.',
    'transactions': 'Transaction Department - Manages deposits, withdrawals, fund transfers, and transaction history.',
    'customer_support': 'Customer Support Department - Resolves user issues, FAQs, complaints, and service requests.',
    'auth_security': 'Authentication & Security Department - Manages login, registration, KYC, OTPs, and fraud detection.'
}

# Simple rule-based routing logic
def route_query(query):
    query = query.lower()
    
    # Keywords for each department
    accounts_keywords = ['account', 'savings', 'current', 'creation', 'open', 'close', 'balance']
    loans_keywords = ['loan', 'emi', 'interest', 'apply', 'approval', 'repayment', 'mortgage']
    transactions_keywords = ['deposit', 'withdraw', 'transfer', 'transaction', 'payment', 'history']
    support_keywords = ['help', 'support', 'issue', 'complaint', 'request', 'problem', 'assistance']
    security_keywords = ['login', 'password', 'register', 'kyc', 'verify', 'otp', 'fraud', 'security']
    
    # Count matches for each department
    scores = {
        'accounts': sum(1 for word in accounts_keywords if word in query),
        'loans': sum(1 for word in loans_keywords if word in query),
        'transactions': sum(1 for word in transactions_keywords if word in query),
        'customer_support': sum(1 for word in support_keywords if word in query),
        'auth_security': sum(1 for word in security_keywords if word in query)
    }
    
    # Get department with highest score
    max_score = max(scores.values())
    
    # If there's a clear winner
    if max_score > 0:
        for dept, score in scores.items():
            if score == max_score:
                return dept
    
    # Default to customer support if no matches
    return 'customer_support'

# Generate dummy user data
def generate_user_data():
    user_ids = ['USR' + str(random.randint(10000, 99999)) for _ in range(5)]
    names = ['John Smith', 'Jane Doe', 'Robert Johnson', 'Emily Davis', 'Michael Wilson']
    emails = ['john.smith@email.com', 'jane.doe@email.com', 'robert.j@email.com', 
              'emily.davis@email.com', 'mike.wilson@email.com']
    
    query = random.randint(0, 4)
    return {
        'user_id': user_ids[query],
        'name': names[query],
        'email': emails[query],
        'account_number': 'ACCT' + str(random.randint(1000000, 9999999)),
        'phone': '+1-' + str(random.randint(100, 999)) + '-' + str(random.randint(100, 999)) + '-' + str(random.randint(1000, 9999))
    }

# Initialize database
with app.app_context():
    db.create_all()


# Helper Functions for OTP
# def send_otp_via_email(email, otp):
#     """Send OTP via email"""
#     try:
#         msg = Message("Your Banking App OTP", recipients=[email])
#         msg.body = f"Your OTP for login is: {otp}. This OTP will expire in 5 minutes."
#         mail.send(msg)
#         return True
#     except Exception as e:
#         print(f"Error sending email: {str(e)}")
#         return False

# Define the system prompt template for the AI chatbot
system_prompt = """
You are a smart, helpful bank assistant chatbot designed to answer customer queries regarding banking services, accounts, transactions, and appointments.

Context: {context}
Chat History: {chat_history}
Question: {question}

Helpful answer:
"""

def set_system_prompt():
    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=['context', 'question', 'chat_history']
    )
    return prompt

def create_conversational_chain():
    retriever = VertexAISearchRetriever(
        project_id=PROJECT_ID,
        location_id=DATA_STORE_LOCATION,
        data_store_id=DATA_STORE_ID,
        get_extractive_answers=True,
        max_documents=10,
        max_extractive_segment_count=1,
        max_extractive_answer_count=5,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer')

    prompt = set_system_prompt()
    llm = VertexAI(model_name=MODEL)

    return ConversationalRetrievalChain.from_llm(
        condense_question_llm=llm,
        get_chat_history=lambda h: h,
        memory=memory,
        return_source_documents=True,
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )   

def text_to_speech(text):
    cleaned_text = re.sub(r'http\S+|www\S+|[^A-Za-z0-9\s.,!?;:]+', '', text)
    tts = gTTS(cleaned_text, lang='en', slow=False)
    tts.save("answer.mp3")
    sound = AudioSegment.from_file("answer.mp3")
    sound = sound.speedup(playback_speed=1.25)
    sound.export("answer.mp3", format="mp3")
    return "answer.mp3"



def send_otp_via_sms(phone_number, otp):
    """Send OTP via SMS using Twilio"""
    try:
        message = twilio_client.messages.create(
            body=f"Your Banking App OTP is: {otp}. This OTP will expire in 5 minutes.",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        return True
    except Exception as e:
        print(f"Error sending SMS: {str(e)}")
        return False





# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Extract all form data
        personal_data = {
            'first_name': request.form.get('first_name'),
            'middle_name': request.form.get('middle_name'),
            'last_name': request.form.get('last_name'),
            'date_of_birth': request.form.get('date_of_birth'),
            'gender': request.form.get('gender'),
            'mobile': request.form.get('mobile'),
            'fathers_name': request.form.get('fathers_name'),
            'mothers_name': request.form.get('mothers_name')
        }

        address_data = {
            'address_line1': request.form.get('address_line1'),
            'address_line2': request.form.get('address_line2'),
            'city': request.form.get('city'),
            'state': request.form.get('state'),
            'pincode': request.form.get('pincode')
        }

        identification_data = {
            'mock_aadhaar': request.form.get('mock_aadhaar'),
            'pan_number': request.form.get('pan_number')
        }

        account_data = {
            'account_type': request.form.get('account_type'),
            'email': request.form.get('email'),
            'password': request.form.get('password'),
            'preferred_language': request.form.get('preferred_language', 'english')
        }

        # Combine all data
        form_data = {**personal_data, **address_data, **identification_data, **account_data}
        image_data = request.form.get('face_image')

        # Validate required fields
        required_fields = ['first_name', 'last_name', 'date_of_birth', 'gender', 'mobile',
                           'address_line1', 'city', 'state', 'pincode', 'mock_aadhaar',
                           'pan_number', 'account_type', 'email', 'password']
        
        for field in required_fields:
            if not form_data.get(field):
                return jsonify({"message": f"Missing required field: {field}"}), 400

        # Check password match
        if form_data['password'] != request.form.get('confirm_password'):
            return jsonify({"message": "Passwords do not match"}), 400

        # Process face image
        try:
            # Split base64 string properly
            header, data = image_data.split(',', 1)
            image_bytes = base64.b64decode(data)
            
            # Convert to RGB explicitly
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_array = np.array(image)

            # Extract face encoding
            face_encodings = face_recognition.face_encodings(image_array)
            if not face_encodings:
                return jsonify({"message": "No face detected, try again!"}), 400

            face_embedding = json.dumps(face_encodings[0].tolist())
        except Exception as e:
            return jsonify({"message": f"Image processing error: {str(e)}"}), 500

        # Check existing user
        if User.query.filter_by(email=form_data['email']).first() or User.query.filter_by(mock_aadhaar=form_data['mock_aadhaar']).first():
            return jsonify({"message": "User already registered with this email or Aadhaar!"}), 400

        # Hash password
        hashed_password = bcrypt.hashpw(form_data['password'].encode(), bcrypt.gensalt()).decode()

        # Create new user
        new_user = User(
            **personal_data,
            **address_data,
            **identification_data,
            account_type=form_data['account_type'],
            email=form_data['email'],
            password=hashed_password,
            face_embedding=face_embedding,
            preferred_language=form_data['preferred_language']
        )

        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "Registration successful!", "redirect": url_for('login')}), 200

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.json
        email = data.get("email")
        password = data.get("password")
        image_data = data.get("image")

        if not email or not password or not image_data:
            return jsonify({"message": "Missing required fields!"}), 400

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"message": "User not found!"}), 400

        # Check password
        if not bcrypt.checkpw(password.encode(), user.password.encode()):
            return jsonify({"message": "Incorrect password!"}), 400

        # Convert base64 image to numpy array
        try:
            # Remove the base64 header if it exists
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_array = np.array(image)

            # Extract face encoding
            face_encodings = face_recognition.face_encodings(image_array)
            if not face_encodings:
                return jsonify({"message": "No face detected, try again!"}), 400

            # Compare face embeddings
            stored_embedding = np.array(json.loads(user.face_embedding))
            match = face_recognition.compare_faces([stored_embedding], face_encodings[0])

            if match[0]:
                # Face recognized, now generate OTP for the next step
                otp_code = OTP.generate_otp(user.id)
                
                # Send OTP via email and SMS
                sms_sent = send_otp_via_sms(f"+91{user.mobile}", otp_code)
                
                if sms_sent:
                    # Store user ID in session for OTP verification
                    session["pending_user_id"] = user.id
                    return jsonify({
                        "success": True,
                        "message": "Face recognized! Please enter the OTP sent to your phone.",
                        "redirect": url_for("verify_otp")
                    })
                else:
                    return jsonify({"message": "Failed to send OTP via SMS. Please try again."}), 500
            else:
                return jsonify({"message": "Face does not match!"}), 400
        except Exception as e:
            return jsonify({"message": f"Image processing error: {str(e)}"}), 500

    return render_template("login.html")


@app.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    if "pending_user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        user_id = session["pending_user_id"]

        # Support both JSON and form data
        if request.is_json:
            data = request.get_json()
            otp_code = data.get("otp_code")
        else:
            otp_code = request.form.get("otp_code")

        print(f"[DEBUG] Submitted OTP: {otp_code} for user_id: {user_id}")

        if not otp_code:
            return jsonify({"success": False, "message": "OTP is required!"}), 400

        if OTP.verify_otp(user_id, otp_code):
            # OTP verified, complete login
            user = User.query.get(user_id)
            session["user_id"] = user.id
            session["user_email"] = user.email
            session["preferred_language"] = user.preferred_language
            session.pop("pending_user_id", None)  # Remove pending user ID

            return jsonify({
                "success": True,
                "message": "Login successful!",
                "redirect": url_for("dashboard")
            })
        else:
            return jsonify({
                "success": False,
                "message": "Invalid or expired OTP. Please try again."
            }), 400

    return render_template("verify_otp.html")



@app.route("/resend_otp", methods=["POST"])
def resend_otp():
    if "pending_user_id" not in session:
        return jsonify({"message": "No pending login session."}), 400
    
    user_id = session["pending_user_id"]
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({"message": "User not found!"}), 400
    
    # Generate new OTP
    otp_code = OTP.generate_otp(user_id)
    
    # Send OTP via email and SMS
    # email_sent = send_otp_via_email(user.email, otp_code)
    sms_sent = send_otp_via_sms(f"+91{user.mobile}", otp_code)
    
    if sms_sent:
        return jsonify({
            "success": True,
            "message": "OTP resent successfully!"
        })
    else:
        return jsonify({"message": "Failed to resend OTP. Please try again."}), 500


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))
    
    return render_template("dashboard.html", user=user.first_name)


# @app.route("/chatbot", methods=["GET", "POST"])
# def chatbot():
#     if "user_id" not in session:
#         return redirect(url_for("login"))
    
#     user = User.query.get(session["user_id"])
#     if not user:
#         session.clear()
#         return redirect(url_for("login"))
    
#     if request.method == "POST":
#         data = request.json
#         user_message = data.get("message")
#         language = data.get("language", session.get("preferred_language", "english"))
        
#         if not user_message:
#             return jsonify({"message": "Empty message!"}), 400
        
#         # Save user message
#         chat_message = ChatMessage(
#             user_id=user.id,
#             content=user_message,
#             is_user=True,
#             language=language
#         )
#         db.session.add(chat_message)
#         db.session.commit()
        
#         # Process the message and generate a response
#         response_data = process_chatbot_message(user, user_message, language)
        
#         # Save bot response
#         chat_message = ChatMessage(
#             user_id=user.id,
#             content=response_data["text"],
#             is_user=False,
#             language=language
#         )
#         db.session.add(chat_message)
#         db.session.commit()
        
#         return jsonify(response_data)
    
#     # Get chat history
#     chat_history = ChatMessage.query.filter_by(user_id=user.id).order_by(ChatMessage.timestamp).all()
#     return render_template("chatbot.html", user=user, chat_history=chat_history)


# @app.route("/ai_chat", methods=["POST"])
# def ai_chat():
#     if "user_id" not in session:
#         return redirect(url_for("login"))
    
#     data = request.json
#     query = data.get("query")
#     session_id = data.get("session_id", str(uuid.uuid4()))

#     # Reuse or create chain
#     if session_id not in user_chains:
#         user_chains[session_id] = create_conversational_chain()
#     chain = user_chains[session_id]

#     response = chain({"question": query})
#     answer = response["answer"]

#     audio_url = None
#     if AUDIO:
#         audio_file = text_to_speech(answer)
#         audio_url = "/audio"

#     sources_text = ""
#     if SOURCE:
#         docs = response.get("source_documents", [])
#         for doc in docs:
#             doc = doc.dict()
#             if 'page_content' in doc:
#                 sources_text += f"- {doc['page_content'][:200]}...\n"

#     # Save the conversation
#     user = User.query.get(session["user_id"])
    
#     # Save user message
#     chat_message = ChatMessage(
#         user_id=user.id,
#         content=query,
#         is_user=True,
#         language=session.get("preferred_language", "english")
#     )
#     db.session.add(chat_message)
    
#     # Save bot response
#     chat_message = ChatMessage(
#         user_id=user.id,
#         content=answer,
#         is_user=False,
#         language=session.get("preferred_language", "english")
#     )
#     db.session.add(chat_message)
#     db.session.commit()

#     return jsonify({
#         "answer": answer,
#         "audio_url": audio_url,
#         "sources": sources_text.strip() if sources_text else None
#     })

# @app.route("/audio")
# def serve_audio():
#     return send_file("answer.mp3", mimetype="audio/mpeg")



# def process_chatbot_message(user, message, language):
#     """Process the user's message and generate a response"""
#     # Translate message to English if needed
#     original_language = language
#     english_message = message
    
#     if language != "english":
#         try:
#             translation = translate_client.translate(message, target_language="en")
#             english_message = translation["translatedText"]
#         except Exception as e:
#             print(f"Translation error: {str(e)}")
    
#     # TODO: Process with a real NLP model or API
#     # For now, use simple keyword matching
#     response = {
#         "text": "",
#         "type": "text",
#         "components": []
#     }
    
#     # Simple intent detection
#     message_lower = english_message.lower()
    
#     if any(word in message_lower for word in ["hi", "hello", "hey", "greetings"]):
#         response["text"] = f"Hello {user.first_name}! How can I assist you today? You can ask about your account, make appointments, check transactions, or apply for services."
    
#     elif any(word in message_lower for word in ["appointment", "book", "schedule", "meeting"]):
#         response["type"] = "appointment"
#         response["text"] = "I can help you book an appointment. Please select an appointment type and date:"
#         response["components"] = [
#             {
#                 "type": "appointment_form",
#                 "options": [
#                     {"value": "account_opening", "label": "Account Opening"},
#                     {"value": "loan_consultation", "label": "Loan Consultation"},
#                     {"value": "wealth_management", "label": "Wealth Management"},
#                     {"value": "issue_resolution", "label": "Issue Resolution"}
#                 ]
#             }
#         ]
    
#     elif any(word in message_lower for word in ["balance", "account", "money", "funds"]):
#         # Mock balance information
#         response["type"] = "account_info"
#         response["text"] = f"Your current account balance is ₹{random.randint(10000, 100000)}.00"
#         response["components"] = [
#             {
#                 "type": "balance_card",
#                 "account_number": f"XXXX{random.randint(1000, 9999)}",
#                 "balance": f"₹{random.randint(10000, 100000)}.00",
#                 "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
#             }
#         ]
    
#     elif any(word in message_lower for word in ["transaction", "history", "payment", "transfer"]):
#         # Mock transaction data
#         transactions = [
#             {
#                 "date": (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d"),
#                 "description": f"{'Credit' if i % 2 == 0 else 'Debit'} - {random.choice(['Salary', 'Shopping', 'Utilities', 'Rent', 'Food'])}",
#                 "amount": f"{'₹' + str(random.randint(100, 5000)) + '.00' if i % 2 == 0 else '-₹' + str(random.randint(100, 1000)) + '.00'}",
#                 "balance": f"₹{random.randint(10000, 100000)}.00"
#             } for i in range(1, 6)
#         ]
        
#         response["type"] = "transactions"
#         response["text"] = "Here are your recent transactions:"
#         response["components"] = [
#             {
#                 "type": "transaction_list",
#                 "transactions": transactions
#             }
#         ]
    
#     elif any(word in message_lower for word in ["loan", "borrow", "credit", "finance"]):
#         response["type"] = "loan_options"
#         response["text"] = "We offer various loan products. Would you like to explore your options or apply for a loan?"
#         response["components"] = [
#             {
#                 "type": "loan_form",
#                 "options": [
#                     {"value": "personal", "label": "Personal Loan"},
#                     {"value": "home", "label": "Home Loan"},
#                     {"value": "car", "label": "Car Loan"},
#                     {"value": "education", "label": "Education Loan"}
#                 ]
#             }
#         ]
    
#     else:
#         response["text"] = "I'm here to help with your banking needs. You can ask about your account, schedule appointments, check transactions, or apply for our services."
    
#     # Translate response back to original language if needed
#     if original_language != "english":
#         try:
#             translation = translate_client.translate(response["text"], target_language=original_language)
#             response["text"] = translation["translatedText"]
#         except Exception as e:
#             print(f"Translation error: {str(e)}")
    
#     return response


# @app.route("/book_appointment", methods=["POST"])
# def book_appointment():
#     if "user_id" not in session:
#         return jsonify({"message": "Please login first!"}), 401
    
#     data = request.json
#     appointment_type = data.get("appointment_type")
#     appointment_date = data.get("appointment_date")
#     notes = data.get("notes", "")
    
#     if not appointment_type or not appointment_date:
#         return jsonify({"message": "Appointment type and date are required!"}), 400
    
#     try:
#         # Parse date string to datetime
#         appointment_datetime = datetime.strptime(appointment_date, "%Y-%m-%dT%H:%M")
        
#         # Generate ticket number
#         ticket_number = Appointment.generate_ticket_number()
        
#         # Create appointment
#         new_appointment = Appointment(
#             user_id=session["user_id"],
#             appointment_type=appointment_type,
#             appointment_date=appointment_datetime,
#             notes=notes,
#             ticket_number=ticket_number
#         )
        
#         db.session.add(new_appointment)
#         db.session.commit()
        
#         # Return confirmation
#         return jsonify({
#             "success": True,
#             "message": f"Appointment booked successfully! Your ticket number is {ticket_number}.",
#             "appointment": {
#                 "id": new_appointment.id,
#                 "type": new_appointment.appointment_type,
#                 "date": new_appointment.appointment_date.strftime("%Y-%m-%d %H:%M"),
#                 "ticket_number": new_appointment.ticket_number
#             }
#         })
    
#     except Exception as e:
#         return jsonify({"message": f"Error booking appointment: {str(e)}"}), 500


# @app.route("/check_appointment_status", methods=["POST"])
# def check_appointment_status():
#     if "user_id" not in session:
#         return jsonify({"message": "Please login first!"}), 401
    
#     data = request.json
#     ticket_number = data.get("ticket_number")
    
#     if not ticket_number:
#         return jsonify({"message": "Ticket number is required!"}), 400
    
#     appointment = Appointment.query.filter_by(user_id=session["user_id"], ticket_number=ticket_number).first()
    
#     if not appointment:
#         return jsonify({"message": "Appointment not found!"}), 404
    
#     return jsonify({
#         "success": True,
#         "appointment": {
#             "id": appointment.id,
#             "type": appointment.appointment_type,
#             "date": appointment.appointment_date.strftime("%Y-%m-%d %H:%M"),
#             "status": appointment.status,
#             "notes": appointment.notes,
#             "ticket_number": appointment.ticket_number
#         }
#     })


# @app.route("/financial_insights", methods=["GET"])
# def financial_insights():
#     if "user_id" not in session:
#         return jsonify({"message": "Please login first!"}), 401
    
#     user = User.query.get(session["user_id"])
    
#     # Mock financial data
#     income = random.randint(30000, 100000)
#     expenses = random.randint(20000, income)
#     savings = income - expenses
#     savings_percentage = (savings / income) * 100
    
#     # Mock transaction categories
#     categories = {
#         "Housing": random.randint(5000, 15000),
#         "Food": random.randint(3000, 8000),
#         "Transportation": random.randint(2000, 6000),
#         "Utilities": random.randint(1000, 4000),
#         "Entertainment": random.randint(1000, 5000),
#         "Healthcare": random.randint(500, 3000),
#         "Others": random.randint(500, 3000)
#     }
    
#     # Generate insights
#     insights = [
#         f"You're saving {savings_percentage:.1f}% of your income.",
#         "Your top spending category is " + max(categories.items(), key=lambda x: x[1])[0] + ".",
#         "Your spending is " + ("below" if expenses < 0.7 * income else "above") + " the recommended budget for your income level."
#     ]
    
#     return jsonify({
#         "success": True,
#         "financial_summary": {
#             "income": income,
#             "expenses": expenses,
#             "savings": savings,
#             "savings_percentage": savings_percentage
#         },
#         "spending_categories": categories,
#         "insights": insights
#     })


# @app.route("/fraud_detection", methods=["POST"])
# def fraud_detection():
#     if "user_id" not in session:
#         return jsonify({"message": "Please login first!"}), 401
    
#     data = request.json
#     transaction_data = data.get("transaction_data")
    
#     if not transaction_data:
#         return jsonify({"message": "Transaction data is required!"}), 400
    
#     # In a real application, this would use the Isolation Forest model
#     # For now, simulating fraud detection with mock data
#     transaction_features = np.array([
#         [
#             float(tx.get('amount', 0)),
#             float(tx.get('hour_of_day', 12)),
#             float(tx.get('day_of_week', 3)),
#             float(tx.get('distance_from_home', 0)),
#             float(tx.get('frequency_last_24h', 1))
#         ] for tx in transaction_data
#     ])
    
#     # Detect anomalies using Isolation Forest
#     if len(transaction_features) > 0:
#         # In production, use a pre-trained model
#         if len(transaction_features) > 0:
#             fraud_scores = fraud_model.decision_function(transaction_features)
#             fraud_predictions = fraud_model.predict(transaction_features)
#             # Convert predictions (-1=anomaly, 1=normal)
#             fraud_predictions = [pred == -1 for pred in fraud_predictions]
        
#         # Add fraud scores to transaction data
#         for i, tx in enumerate(transaction_data):
#             tx['fraud_score'] = float(fraud_scores[i])
#             tx['is_suspicious'] = bool(fraud_predictions[i])
    
#     # Graph analysis for detecting fraud networks would happen here
#     # This would involve Louvain community detection, Tarjan's algorithm, etc.
#     # For demo purposes, we'll just return the mock results
    
#     return jsonify({
#         "success": True,
#         "transaction_analysis": transaction_data,
#         "fraud_network_detected": False,
#         "risk_level": "low"
#     })


@app.route("/loan_approval")
def loan_approval():
    return render_template("loan_approval.html")

@app.route("/credit_risk_assessment", methods=["POST"])
def credit_risk_assessment():
    if "user_id" not in session:
        return render_template("loan_approval.html", error="Please login first!")

    # Get form data
    name = request.form.get("name")
    amount = request.form.get("amount")
    income = request.form.get("income")
    purpose = request.form.get("purpose")
    employment = request.form.get("employment")

    if not all([name, amount, income, purpose, employment]):
        return render_template("loan_approval.html", error="All fields are required!")

    # Mock risk assessment
    credit_score = random.randint(600, 850)

    if credit_score >= 750:
        risk_level = "low"
        approval_probability = round(random.uniform(0.85, 0.98), 2)
        recommended_interest_rate = round(random.uniform(7.0, 9.0), 2)
    elif credit_score >= 680:
        risk_level = "medium"
        approval_probability = round(random.uniform(0.65, 0.84), 2)
        recommended_interest_rate = round(random.uniform(9.1, 12.0), 2)
    else:
        risk_level = "high"
        approval_probability = round(random.uniform(0.30, 0.64), 2)
        recommended_interest_rate = round(random.uniform(12.1, 16.0), 2)

    return render_template("loan_approval.html", result={
        "name": name,
        "credit_score": credit_score,
        "risk_level": risk_level,
        "approval_probability": approval_probability,
        "recommended_interest_rate": recommended_interest_rate,
        "factors": [
            "Credit history length",
            "Payment history",
            "Debt-to-income ratio",
            "Employment stability"
        ]
    })


# @app.route("/branch_routing", methods=["POST"])
# def branch_routing():
#     """Automated routing system for branch visits and staff notifications"""
#     if "user_id" not in session:
#         return jsonify({"message": "Please login first!"}), 401
    
#     data = request.json
#     service_type = data.get("service_type")
#     preferred_branch = data.get("branch_id")
#     preferred_time = data.get("preferred_time")
    
#     if not service_type or not preferred_branch:
#         return jsonify({"message": "Service type and branch are required!"}), 400
    
#     # Mock branch data for demonstration
#     branches = {
#         "BR001": {
#             "name": "Main Branch",
#             "address": "123 Main Street, City",
#             "current_wait_time": random.randint(5, 30),
#             "staff_available": random.randint(3, 8)
#         },
#         "BR002": {
#             "name": "North Branch",
#             "address": "456 North Avenue, City",
#             "current_wait_time": random.randint(5, 30),
#             "staff_available": random.randint(3, 8)
#         },
#         "BR003": {
#             "name": "East Branch",
#             "address": "789 East Boulevard, City",
#             "current_wait_time": random.randint(5, 30),
#             "staff_available": random.randint(3, 8)
#         }
#     }
    
#     # Get selected branch info
#     branch = branches.get(preferred_branch)
#     if not branch:
#         return jsonify({"message": "Invalid branch selected!"}), 400
    
#     # Generate token number
#     token_number = f"{preferred_branch}-{random.randint(100, 999)}"
    
#     # Calculate estimated wait time
#     estimated_wait = branch["current_wait_time"]
    
#     # Prioritization logic would happen here in a real system
#     customer_priority = "Regular"  # Could be VIP, Premium, etc. based on account type
    
#     # In a real system, send notification to branch staff
#     # For demo, we'll just simulate this
#     staff_notification = {
#         "token": token_number,
#         "service": service_type,
#         "customer_id": session["user_id"],
#         "priority": customer_priority,
#         "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
#     }
    
#     return jsonify({
#         "success": True,
#         "routing_info": {
#             "token_number": token_number,
#             "branch": branch["name"],
#             "address": branch["address"],
#             "estimated_wait_time": estimated_wait,
#             "preferred_time": preferred_time,
#             "service_type": service_type
#         },
#         "notification_sent": True
#     })


# @app.route("/change_language", methods=["POST"])
# def change_language():
#     """Change user's preferred language"""
#     if "user_id" not in session:
#         return jsonify({"message": "Please login first!"}), 401
    
#     data = request.json
#     language = data.get("language")
    
#     if not language:
#         return jsonify({"message": "Language is required!"}), 400
    
#     # Update user's preferred language
#     user = User.query.get(session["user_id"])
#     user.preferred_language = language
#     db.session.commit()
    
#     # Update session
#     session["preferred_language"] = language
    
#     return jsonify({
#         "success": True,
#         "message": "Language preference updated!",
#         "language": language
#     })


# @app.route("/chatbot_history", methods=["GET"])
# def chatbot_history():
#     """Get user's chat history"""
#     if "user_id" not in session:
#         return jsonify({"message": "Please login first!"}), 401
    
#     # Get chat history
#     chat_history = ChatMessage.query.filter_by(user_id=session["user_id"]).order_by(ChatMessage.timestamp).all()
    
#     # Format history for response
#     formatted_history = [
#         {
#             "id": msg.id,
#             "content": msg.content,
#             "is_user": msg.is_user,
#             "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#             "language": msg.language
#         } for msg in chat_history
#     ]
    
#     return jsonify({
#         "success": True,
#         "chat_history": formatted_history
#     })


@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query")
    session_id = data.get("session_id", str(uuid.uuid4()))

    if session_id not in user_chains:
        user_chains[session_id] = create_conversational_chain()
    chain = user_chains[session_id]

    response = chain({"question": query})
    answer = response["answer"]

    audio_url = None
    if AUDIO:
        text_to_speech(answer)
        audio_url = "/audio"

    sources_text = ""
    if SOURCE:
        docs = response.get("source_documents", [])
        for doc in docs:
            doc = doc.dict()
            if 'page_content' in doc:
                sources_text += f"- {doc['page_content'][:200]}...\n"

    return jsonify({
        "answer": answer,
        "audio_url": audio_url,
        "sources": sources_text.strip() if sources_text else None
    })

@app.route("/audio")
def serve_audio():
    return send_file("answer.mp3", mimetype="audio/mpeg")     



@app.route('/fraud', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'User_ID': request.form['User_ID'],
            'Amount': float(request.form['Amount']),
            'Location': request.form['Location'],
            'Transaction_Type': request.form['Transaction_Type'],
            'Previous_Transaction_Minutes_Gap': int(request.form['Previous_Transaction_Time']),
            'Hour': int(request.form['Hour'])
        }

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        results = {}
        for model_name, model in models.items():
            y_pred = model.predict(df_input)[0]
            y_prob = model.predict_proba(df_input)[0][1]
            results[model_name] = {
                'prediction': 'Fraud' if y_pred == 1 else 'Legit',
                'probability': round(y_prob * 100, 2)
            }

        return render_template('fraud.html', input_data=input_data, results=results, confusion_matrices=confusion_matrices)

    return render_template('fraud.html')


@app.route('/query', methods=['GET'])
def query():
    return render_template('query.html')

@app.route('/submit_query', methods=['POST'])
def submit_query():
    if request.method == 'POST':
        query_text = request.form['query']
        
        # Store in session
        session['query'] = query_text
        
        # Route query to department
        department = route_query(query_text)
        session['department'] = department
        
        return redirect(url_for('departments'))
    
    return redirect(url_for('query'))

@app.route('/departments')
def departments():
    query = session.get('query', 'No query found')
    department = session.get('department', 'customer_support')
    
    # List all departments with the selected one highlighted
    return render_template('departments.html', 
                          query=query, 
                          selected_dept=department,
                          departments=DEPARTMENTS)

@app.route('/download_pdf/<department>')
def download_pdf(department):
    try:
        query = session.get('query', 'No query found')
        
        if department not in DEPARTMENTS:
            department = 'customer_support'
        
        # Generate user data
        user_data = generate_user_data()
        
        # Generate PDF or text file
        file_path = generate_pdf(query, department, DEPARTMENTS[department], user_data)
        
        # Check if file exists
        if os.path.exists(file_path):
            # Get file mimetype
            file_ext = os.path.splitext(file_path)[1]
            if file_ext == '.pdf':
                mimetype = 'application/pdf'
                download_name = f"query_{department}.pdf"
            else:
                mimetype = 'text/plain'
                download_name = f"query_{department}.txt"
                
            # Send the file
            return send_file(
                file_path, 
                as_attachment=True, 
                download_name=download_name,
                mimetype=mimetype
            )
        else:
            # Flash message if file not found
            flash('Error generating file. Please try again.', 'error')
            return redirect(url_for('departments'))
            
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        flash('Error downloading file. Please try again.', 'error')
        return redirect(url_for('departments'))


@app.route("/account")
def account():
    return render_template("account.html")

@app.route("/transfers")
def transfers():
    return render_template("transfers.html")


@app.route("/payments")
def payments():
    return render_template("payments.html")

@app.route("/statements")
def statements():
    return render_template("statements.html")

@app.route("/investments")
def investments():
    return render_template("investments.html")

@app.route("/alerts")
def alerts():
    return render_template("alerts.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")




@app.route("/logout")
def logout():
    """Log out the user"""
    session.clear()
    return redirect(url_for("home"))


# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error_code=404, message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error_code=500, message="Server error"), 500


if __name__ == "__main__":
    app.run(debug=True)