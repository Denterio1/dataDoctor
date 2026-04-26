"""
auth.py — User Authentication & Database (Supabase)

Simplified version without OAuth.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
load_dotenv()


def _get_client():
    """Get Supabase client."""
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def sync_user_to_db(user: Any) -> None:
    """Sync user to public.users table."""
    client = _get_client()
    if not client or not user:
        return

    try:
        # Get metadata
        user_id = user.get("id")
        email = user.get("email")
        name = user.get("name") or (email.split("@")[0] if email else "User")
        plan = user.get("plan", "professional")

        client.table("users").upsert({
            "id": user_id,
            "email": email,
            "name": name,
            "last_seen": datetime.now().isoformat(),
            "plan": plan
        }).execute()
    except Exception as e:
        print(f"Warning: Could not sync user to DB: {e}")


def sign_up(email: str, password: str, name: str = "") -> dict[str, Any] | str:
    """Register a new user (Legacy)."""
    return "Registration is disabled in this version."


def sign_in(email: str, password: str) -> dict[str, Any] | str:
    """Authenticate a user (Legacy)."""
    return "Login is disabled in this version."


def register_or_login(email: str, name: str = "") -> dict[str, Any] | None:
    """Legacy support for email-only flow."""
    return {
        "id": "guest_user",
        "email": email or "guest@example.com",
        "name": name or "Guest",
        "plan": "professional"
    }


def log_session(
    email:     str,
    file_name: str,
    file_size: int,
    action:    str,
    ml_score:  int = 0,
) -> None:
    """Log a user session/action."""
    client = _get_client()
    if not client:
        return

    try:
        # Get user ID
        result = client.table("users").select("id").eq("email", email).execute()
        if not result.data:
            return

        user_id = result.data[0]["id"]

        client.table("sessions").insert({
            "user_id":    user_id,
            "file_name":  file_name,
            "file_size":  file_size,
            "action":     action,
            "ml_score":   ml_score,
            "created_at": datetime.now().isoformat(),
        }).execute()
    except Exception:
        pass


def get_user_stats(email: str) -> dict[str, Any]:
    """Get usage statistics for a user."""
    return {
        "name":            "Guest",
        "plan":            "professional",
        "total_sessions":  0,
        "files_analysed":  0,
        "avg_ml_score":    0,
        "last_seen":       datetime.now().isoformat()[:19].replace("T", " "),
        "member_since":    datetime.now().isoformat()[:10],
    }


def is_supabase_configured() -> bool:
    """Check if Supabase is properly configured."""
    return bool(
        os.environ.get("SUPABASE_URL") and
        os.environ.get("SUPABASE_KEY")
    )
