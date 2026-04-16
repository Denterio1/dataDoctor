"""
auth.py — User Authentication & Database (Supabase)

Handles:
    - User registration and login
    - Session tracking
    - Usage analytics
    - Privacy-first design (no data stored, only metadata)
"""

from __future__ import annotations
import os
import hashlib
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


def register_or_login(email: str, name: str = "") -> dict[str, Any] | None:
    """
    Register a new user or update last_seen for existing user.

    Args:
        email: User email address.
        name:  Display name (optional).

    Returns:
        User dict or None if failed.
    """
    client = _get_client()
    if not client:
        return None

    try:
        # Check if user exists
        result = client.table("users").select("*").eq("email", email).execute()

        if result.data:
            # Update last_seen
            client.table("users").update({
                "last_seen": datetime.now().isoformat(),
            }).eq("email", email).execute()
            return result.data[0]
        else:
            # Create new user
            new_user = {
                "email":      email,
                "name":       name or email.split("@")[0],
                "created_at": datetime.now().isoformat(),
                "last_seen":  datetime.now().isoformat(),
                "plan":       "free",
            }
            result = client.table("users").insert(new_user).execute()
            return result.data[0] if result.data else None
    except Exception:
        return None


def log_session(
    email:     str,
    file_name: str,
    file_size: int,
    action:    str,
    ml_score:  int = 0,
) -> None:
    """
    Log a user session/action.

    Args:
        email:     User email.
        file_name: Name of the file analysed.
        file_size: File size in bytes.
        action:    Action performed (inspect, clean, ml, etc).
        ml_score:  ML Readiness Score if computed.
    """
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
    """
    Get usage statistics for a user.

    Returns:
        {
            "total_sessions": int,
            "files_analysed": int,
            "avg_ml_score":   float,
            "last_seen":      str,
            "plan":           str,
        }
    """
    client = _get_client()
    if not client:
        return {}

    try:
        user_result = client.table("users").select("*").eq("email", email).execute()
        if not user_result.data:
            return {}

        user    = user_result.data[0]
        user_id = user["id"]

        sessions_result = client.table("sessions").select("*").eq("user_id", user_id).execute()
        sessions = sessions_result.data or []

        ml_scores = [s["ml_score"] for s in sessions if s.get("ml_score", 0) > 0]

        return {
            "name":            user.get("name", ""),
            "plan":            user.get("plan", "free"),
            "total_sessions":  len(sessions),
            "files_analysed":  len(set(s["file_name"] for s in sessions)),
            "avg_ml_score":    round(sum(ml_scores) / len(ml_scores), 1) if ml_scores else 0,
            "last_seen":       user.get("last_seen", "")[:19].replace("T", " "),
            "member_since":    user.get("created_at", "")[:10],
        }
    except Exception:
        return {}


def get_all_users() -> list[dict]:
    """Get all users (admin only)."""
    client = _get_client()
    if not client:
        return []

    try:
        result = client.table("users").select("*").order("created_at", desc=True).execute()
        return result.data or []
    except Exception:
        return []


def is_supabase_configured() -> bool:
    """Check if Supabase is properly configured."""
    return bool(
        os.environ.get("SUPABASE_URL") and
        os.environ.get("SUPABASE_KEY")
    )