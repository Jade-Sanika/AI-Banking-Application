from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
import random
import string

db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Personal Information
    first_name = db.Column(db.String(50), nullable=False)
    middle_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    mobile = db.Column(db.String(10), nullable=False)
    fathers_name = db.Column(db.String(100))
    mothers_name = db.Column(db.String(100))
    
    # Address Information
    address_line1 = db.Column(db.String(100), nullable=False)
    address_line2 = db.Column(db.String(100))
    city = db.Column(db.String(50), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    pincode = db.Column(db.String(6), nullable=False)
    
    # Identification Details
    mock_aadhaar = db.Column(db.String(12), nullable=False)
    pan_number = db.Column(db.String(10), nullable=False)
    
    # Account Information
    account_type = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    preferred_language = db.Column(db.String(20), default='english')
    
    # Face Authentication
    face_embedding = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    appointments = db.relationship('Appointment', backref='user', lazy=True)
    transactions = db.relationship('Transaction', backref='user', lazy=True)
    
    def verify_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

class OTP(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    otp_code = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    verified = db.Column(db.Boolean, default=False)
    
    @staticmethod
    def generate_otp(user_id, expire_minutes=5):
        # Generate a 6-digit OTP
        otp_code = ''.join(random.choices(string.digits, k=6))
        
        # Calculate expiration time
        expires_at = datetime.utcnow() + timedelta(minutes=expire_minutes)
        
        # Create new OTP record
        new_otp = OTP(
            user_id=user_id,
            otp_code=otp_code,
            expires_at=expires_at
        )
        
        # Add to database
        db.session.add(new_otp)
        db.session.commit()

        print(f"[DEBUG] OTP generated for user_id={user_id}: {otp_code}, expires at {expires_at}")
        
        return otp_code
    
    @staticmethod
    def verify_otp(user_id, otp_code):
        print(f"[DEBUG] Verifying OTP '{otp_code}' for user_id={user_id}")
        # Find the most recent OTP for this user that's not expired or already verified
        otp = OTP.query.filter_by(
            user_id=user_id,
            otp_code=otp_code,
            verified=False
        ).filter(OTP.expires_at > datetime.utcnow()).order_by(OTP.created_at.desc()).first()
        
        if otp:
            otp.verified = True
            db.session.commit()
            print(f"[DEBUG] OTP verification successful for user_id={user_id}")
            return True
        else:
            print(f"[DEBUG] OTP verification failed: Either incorrect, already used, or expired.")
        
        return False

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    appointment_type = db.Column(db.String(50), nullable=False)
    appointment_date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='scheduled')
    notes = db.Column(db.Text)
    ticket_number = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    @staticmethod
    def generate_ticket_number():
        # Generate a unique ticket number
        prefix = "BNK"
        random_digits = ''.join(random.choices(string.digits, k=6))
        return f"{prefix}{random_digits}"

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200))
    transaction_date = db.Column(db.DateTime, default=datetime.utcnow)
    reference_number = db.Column(db.String(20), nullable=False)
    
    @staticmethod
    def generate_reference_number():
        # Generate a unique reference number for transactions
        prefix = "TXN"
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M')
        random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        return f"{prefix}{timestamp}{random_chars}"

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)  # True if sent by user, False if sent by bot
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    language = db.Column(db.String(20), default='english')