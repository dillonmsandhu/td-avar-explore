import subprocess
import os
import base64

def email_pdf(pdf_filename='figures/is-ppo-training.pdf', recipient='ds541@cs.duke.edu'):
    """
    Simple function to email PDF using sendmail with proper MIME encoding
    This method will definitely work since you have sendmail available.
    
    Args:
        pdf_filename: Path to the PDF file
        recipient: Email address (defaults to your Duke email)
    """
    
    # Check if file exists
    if not os.path.exists(pdf_filename):
        print(f"❌ File {pdf_filename} not found!")
        print("💡 Make sure to save your plot first:")
        print(f"   fig.savefig('{pdf_filename}', bbox_inches='tight', pad_inches=0.02)")
        return False
    
    file_size = os.path.getsize(pdf_filename) / 1024  # KB
    print(f"📁 Found {pdf_filename} ({file_size:.1f} KB)")
    
    if file_size > 10000:  # 10MB limit
        print("⚠️ File is quite large, might be rejected by email server")
    
    try:
        # Read and encode PDF
        with open(pdf_filename, 'rb') as f:
            pdf_data = f.read()
        pdf_b64 = base64.b64encode(pdf_data).decode()
        
        # Split base64 into 76-character lines (RFC requirement)
        pdf_b64_lines = [pdf_b64[i:i+76] for i in range(0, len(pdf_b64), 76)]
        pdf_b64_formatted = '\n'.join(pdf_b64_lines)
        
        # Create proper MIME email
        boundary = "boundary_123_pdf_attachment"
        subject = f"Jupyter Plot: {os.path.basename(pdf_filename)}"
        
        email_content = f"""To: {recipient}
From: {recipient}
Subject: {subject}
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="{boundary}"

This is a multi-part message in MIME format.

--{boundary}
Content-Type: text/plain; charset=UTF-8

Hi!

Your plot from the Jupyter notebook is attached.

File: {os.path.basename(pdf_filename)}
Size: {file_size:.1f} KB

Generated from your notebook.

Best regards,
Your Jupyter Notebook 🐍

--{boundary}
Content-Type: application/pdf
Content-Transfer-Encoding: base64
Content-Disposition: attachment; filename="{os.path.basename(pdf_filename)}"

{pdf_b64_formatted}
--{boundary}--
"""
        
        print(f"📤 Sending to {recipient} using sendmail...")
        
        # Send via sendmail
        cmd = ['/sbin/sendmail', recipient]
        result = subprocess.run(cmd, input=email_content, text=True, capture_output=True)
        
        if result.returncode == 0:
            print("✅ Email sent successfully with attachment!")
            print(f"📧 Check your inbox at {recipient}")
            return True
        else:
            print(f"❌ sendmail failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_email(recipient='ds541@cs.duke.edu'):
    """Send a simple test email without attachment"""
    try:
        email_content = f"""To: {recipient}
From: {recipient}
Subject: Test from Jupyter Notebook

This is a test email from your Jupyter notebook to verify mail is working.

If you receive this, the mail system is functional!
"""
        
        print(f"📤 Sending test email to {recipient}...")
        cmd = ['/sbin/sendmail', recipient]
        result = subprocess.run(cmd, input=email_content, text=True, capture_output=True)
        
        if result.returncode == 0:
            print("✅ Test email sent! Check your inbox.")
            return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

# Quick usage:
# 1. Test basic email first:
# test_email()

# 2. Save your plot and email it:
# fig.savefig('figures/is-ppo-training.pdf', bbox_inches="tight", pad_inches=0.02)
# email_pdf()

print("📧 Email functions ready!")
print("Try: test_email() first, then email_pdf_simple()")