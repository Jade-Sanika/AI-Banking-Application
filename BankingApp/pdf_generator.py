# pdf_generator.py
from fpdf import FPDF
import os
import datetime
import random

def generate_pdf(query, department, department_description, user_data):
    # Create PDF directory if not exists
    pdf_dir = os.path.join(os.getcwd(), 'pdfs')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(pdf_dir, f"query_{department}_{timestamp}.pdf")
    
    # Add hardcoded account balance - use "Rs." instead of ₹ symbol to avoid encoding issues
    account_balance = f"Rs. {random.randint(10000, 999999):,}"
    cibil_score = random.randint(650, 850)
    
    # Create PDF with error handling for each step
    pdf = FPDF()
    pdf.add_page()
    
    # Set up fonts - use standard fonts that are guaranteed to be available
    # Skip DejaVu font which might not be available and cause errors
    pdf.set_font('Arial', 'B', 16)
    
    # Header
    pdf.cell(0, 10, "Banking Query System - Report", 0, 1, "C")
    pdf.line(10, 20, 200, 20)
    pdf.ln(10)
    
    # Department Info
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Routed to: {department_description}", 0, 1)
    pdf.ln(5)
    
    # User Information Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "User Information:", 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(40, 8, "User ID:", 0, 0)
    pdf.cell(0, 8, user_data['user_id'], 0, 1)
    
    pdf.cell(40, 8, "Name:", 0, 0)
    pdf.cell(0, 8, user_data['name'], 0, 1)
    
    pdf.cell(40, 8, "Email:", 0, 0)
    pdf.cell(0, 8, user_data['email'], 0, 1)
    
    pdf.cell(40, 8, "Account Number:", 0, 0)
    pdf.cell(0, 8, user_data['account_number'], 0, 1)
    
    pdf.cell(40, 8, "Phone:", 0, 0)
    pdf.cell(0, 8, user_data['phone'], 0, 1)
    
    # Add account balance and CIBIL score
    pdf.cell(40, 8, "Account Balance:", 0, 0)
    pdf.cell(0, 8, account_balance, 0, 1)
    
    pdf.cell(40, 8, "CIBIL Score:", 0, 0)
    
    # Determine CIBIL score color based on range
    if cibil_score >= 750:
        score_text = f"{cibil_score} (Excellent)"
    elif cibil_score >= 700:
        score_text = f"{cibil_score} (Good)"
    elif cibil_score >= 650: 
        score_text = f"{cibil_score} (Fair)"
    else:
        score_text = f"{cibil_score} (Needs Improvement)"
        
    pdf.cell(0, 8, score_text, 0, 1)
    
    pdf.ln(10)
    
    # Query Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Query Details:", 0, 1)
    
    pdf.set_font('Arial', '', 12)
    
    # Handle potential encoding issues by ensuring the text is properly encoded
    # Clean the query text to avoid encoding errors
    safe_query = ''.join(c if ord(c) < 128 else ' ' for c in query)
    
    # Use multi_cell for better text wrapping with long queries
    pdf.multi_cell(0, 8, f"Query: {safe_query}")
    
    pdf.ln(10)
    
    # Date and Time
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    
    try:
        # Save PDF with explicit exception handling
        pdf.output(filename)
        return filename
    except Exception as e:
        # If saving fails, try again with a different approach
        print(f"Error saving PDF: {str(e)}")
        
        # Create a new PDF with even simpler content
        retry_pdf = FPDF()
        retry_pdf.add_page()
        retry_pdf.set_font('Courier', '', 12)  # Use the most basic font
        
        retry_pdf.cell(0, 10, "Banking Query Report", 0, 1, "C")
        retry_pdf.ln(5)
        retry_pdf.cell(0, 10, f"Department: {department}", 0, 1)
        retry_pdf.ln(5)
        retry_pdf.cell(0, 10, f"User: {user_data['name']}", 0, 1)
        retry_pdf.ln(5)
        retry_pdf.cell(0, 10, "Query: See system for details", 0, 1)
        retry_pdf.ln(5)
        retry_pdf.cell(0, 10, f"Generated: {timestamp}", 0, 1)
        
        # Try to save with a different name
        retry_filename = os.path.join(pdf_dir, f"query_{department}_{timestamp}_retry.pdf")
        try:
            retry_pdf.output(retry_filename)
            return retry_filename
        except Exception as e2:
            print(f"Second attempt failed: {str(e2)}")
            # If all PDF generation attempts fail, raise the error to be handled upstream
            raise Exception("PDF generation failed after multiple attempts")