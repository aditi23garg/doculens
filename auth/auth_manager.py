"""Register / login logic. Both return (user, error); exactly one is None."""
import re
from typing import Optional

from auth.password_utils import MAX_PASSWORD_BYTES, hash_password, verify_password
from database import mongo_client as db

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _public(user: dict) -> dict:
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


def register(name: str, email: str, password: str) -> tuple[Optional[dict], Optional[str]]:
    name, email = name.strip(), email.strip().lower()
    if not name:
        return None, "Enter your name."
    if not _EMAIL_RE.match(email):
        return None, "Enter a valid email address."
    if len(password) < 8:
        return None, "Use a password with at least 8 characters."
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return None, f"Use a password under {MAX_PASSWORD_BYTES} bytes."
    if db.get_user_by_email(email):
        return None, "An account with this email already exists. Sign in instead."
    user = db.create_user(name, email, hash_password(password))
    if user is None:  # lost a race on the unique index
        return None, "An account with this email already exists. Sign in instead."
    return _public(user), None


def login(email: str, password: str) -> tuple[Optional[dict], Optional[str]]:
    user = db.get_user_by_email(email.strip().lower())
    if not user or not verify_password(password, user["password_hash"]):
        return None, "Email or password is incorrect."
    return _public(user), None
