from auth.supabase_client import get_supabase


def sign_up(email, password):
    sb = get_supabase()
    res = sb.auth.sign_up({"email": email, "password": password})
    if res.user:
        return True, "Account created! You can now log in."
    return False, "Sign up failed. Try again."


def sign_in(email, password):
    sb = get_supabase()
    try:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        return res.user, res.session
    except Exception as e:
        return None, str(e)


def sign_out():
    sb = get_supabase()
    sb.auth.sign_out()


def save_history(user_id, question, answer, source_label, access_token):
    try:
        sb = get_supabase()
        sb.postgrest.auth(access_token)
        sb.table("history").insert({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "source_label": source_label or ""
        }).execute()
    except Exception:
        pass


def fetch_history(user_id, access_token, limit=30):
    try:
        sb = get_supabase()
        sb.postgrest.auth(access_token)
        res = sb.table("history") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return res.data or []
    except Exception:
        return []
