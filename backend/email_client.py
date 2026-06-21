"""
Transactional email via Resend (https://resend.com).
Falls back to a stdout log when RESEND_API_KEY is not set — safe for local dev.

Required env vars:
  RESEND_API_KEY   re_...
  FROM_EMAIL       noreply@yourdomain.com  (defaults to onboarding@resend.dev for dev)
  APP_BASE_URL     https://yourapp.up.railway.app  (defaults to http://localhost:8000)
"""

import os

_RESEND_KEY  = os.environ.get("RESEND_API_KEY")
_FROM_EMAIL  = os.environ.get("FROM_EMAIL", "onboarding@resend.dev")
_BASE_URL    = os.environ.get("APP_BASE_URL", "http://localhost:8000")


def _send(to: str, subject: str, html: str) -> bool:
    if not _RESEND_KEY:
        print(f"[email] (no RESEND_API_KEY) To={to} Subject={subject}")
        return True
    try:
        import resend
        resend.api_key = _RESEND_KEY
        resend.Emails.send({
            "from":    _FROM_EMAIL,
            "to":      [to],
            "subject": subject,
            "html":    html,
        })
        return True
    except Exception as e:
        print(f"[email] Resend error: {e}")
        return False


def send_verification_email(to: str, name: str, token: str) -> bool:
    link = f"{_BASE_URL}/api/auth/verify-email?token={token}"
    html = f"""
    <div style="font-family:sans-serif;max-width:500px;margin:auto;padding:32px">
      <h2 style="color:#58a6ff;margin-bottom:8px">Verify your SleepSense AI account</h2>
      <p style="color:#444">Hi {name},</p>
      <p style="color:#444">Click the button below to verify your email and activate your account.</p>
      <a href="{link}"
         style="display:inline-block;margin:20px 0;padding:12px 24px;background:#58a6ff;
                color:#0d1117;border-radius:8px;font-weight:700;text-decoration:none">
        Verify Email →
      </a>
      <p style="color:#888;font-size:12px">This link expires in 24 hours. If you didn't sign up, ignore this email.</p>
    </div>"""
    return _send(to, "Verify your SleepSense AI email", html)


def send_password_reset_email(to: str, name: str, token: str) -> bool:
    link = f"{_BASE_URL}/?reset={token}"
    html = f"""
    <div style="font-family:sans-serif;max-width:500px;margin:auto;padding:32px">
      <h2 style="color:#58a6ff;margin-bottom:8px">Reset your SleepSense AI password</h2>
      <p style="color:#444">Hi {name},</p>
      <p style="color:#444">We received a request to reset your password. Click the button below:</p>
      <a href="{link}"
         style="display:inline-block;margin:20px 0;padding:12px 24px;background:#58a6ff;
                color:#0d1117;border-radius:8px;font-weight:700;text-decoration:none">
        Reset Password →
      </a>
      <p style="color:#888;font-size:12px">This link expires in 1 hour. If you didn't request a reset, ignore this email.</p>
    </div>"""
    return _send(to, "Reset your SleepSense AI password", html)
