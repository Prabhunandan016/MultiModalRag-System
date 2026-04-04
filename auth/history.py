import html
import logging
from auth.supabase_client import get_supabase

logger = logging.getLogger(__name__)


def _parse_error(e):
    msg = str(e).lower()
    if "rate limit" in msg or "429" in msg or "too many" in msg:
        return "Too many attempts. Please wait a few minutes and try again."
    if "already registered" in msg or "already exists" in msg or "user already" in msg:
        return "This email is already registered. Please sign in instead."
    if "invalid login" in msg or "invalid email" in msg or "invalid password" in msg:
        return "Invalid email or password."
    if "email not confirmed" in msg:
        return "Please confirm your email before signing in."
    if "password" in msg and ("short" in msg or "weak" in msg or "characters" in msg):
        return "Password must be at least 6 characters."
    if "network" in msg or "connection" in msg or "timeout" in msg:
        return "Network error. Please check your connection and try again."
    # Sanitize error message to prevent XSS
    return f"Error: {html.escape(str(e))}"


def sign_up(email, password):
    try:
        if not email or "@" not in email or "." not in email.split("@")[-1]:
            return False, "Please enter a valid email address."
        if len(password) < 6:
            return False, "Password must be at least 6 characters."
        sb = get_supabase()
        res = sb.auth.sign_up({"email": email.strip().lower(), "password": password})
        if res.user:
            return True, "Account created! You can now sign in."
        return False, "Sign up failed. Please try again."
    except BaseException as e:
        logger.warning("Sign up failed: %s", type(e).__name__)
        return False, _parse_error(e)


def sign_in(email, password):
    try:
        if not email or not password:
            return None, "Please enter your email and password."
        sb = get_supabase()
        res = sb.auth.sign_in_with_password({"email": email.strip().lower(), "password": password})
        return res.user, res.session
    except BaseException as e:
        logger.warning("Sign in failed: %s", type(e).__name__)
        return None, _parse_error(e)


def sign_out():
    try:
        sb = get_supabase()
        sb.auth.sign_out()
    except BaseException as e:
        logger.warning("Sign out error: %s", type(e).__name__)


def save_history(user_id, question, answer, source_label, access_token):
    if not user_id or not access_token:
        return
    try:
        sb = get_supabase()
        sb.postgrest.auth(access_token)
        sb.table("history").insert({
            "user_id": str(user_id),
            "question": question[:2000] if question else "",
            "answer": answer[:5000] if answer else "",
            "source_label": (source_label or "")[:200]
        }).execute()
    except BaseException as e:
        logger.warning("Failed to save history: %s", type(e).__name__)


def fetch_history(user_id, access_token, limit=30):
    if not user_id or not access_token:
        return []
    try:
        sb = get_supabase()
        sb.postgrest.auth(access_token)
        res = sb.table("history") \
            .select("*") \
            .eq("user_id", str(user_id)) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return res.data or []
    except BaseException as e:
        logger.warning("Failed to fetch history: %s", type(e).__name__)
        return []
