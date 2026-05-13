"""Streamlit UI for the auth screens (Sign In / Sign Up / Forgot Password)."""

from __future__ import annotations

import re

import streamlit as st

from . import auth_db


_AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

/* Force-hide the top white strip / Streamlit chrome */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }
header { display: none !important; }
#MainMenu { display: none !important; }
.stApp > header { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Belt-and-braces for other chrome that occasionally leaks */
footer {display: none !important;}
[data-testid="stToolbarActions"] {display: none !important;}
[data-testid="stDeployButton"], [data-testid="stAppDeployButton"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
[data-testid="stSidebar"] {display: none !important;}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stAppViewBlockContainer"] {padding-top: 0 !important;}

/* Kill the empty white box above the auth card (column / vertical-block padding) */
[data-testid="column"] > div:first-child {
    background: transparent !important;
    padding: 0 !important;
}
[data-testid="stVerticalBlock"] > div:first-child {
    background: transparent !important;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #f8f9fa !important;
    color: #1a1a1a !important;
    font-family: 'DM Sans', sans-serif !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
}

/* The middle column itself becomes the auth card — no manual <div> bookends.
   The .auth-card-anchor is a hidden marker we render once inside the column. */
.auth-card-anchor { display: none !important; }
[data-testid="column"]:has(.auth-card-anchor) {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08) !important;
    padding: 2.25rem 2rem 2rem 2rem !important;
    color: #1a1a1a !important;
}
.auth-brand {
    text-align: center;
    margin-bottom: 1.25rem;
}
.auth-brand .logo {
    font-size: 2.25rem;
    line-height: 1;
}
.auth-brand h1 {
    margin: 0.25rem 0 0.25rem 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a1a;
    font-family: 'DM Sans', sans-serif;
}
/* Suppress the global section.main h1::before blue dot on the auth title. */
.auth-brand h1::before { content: none !important; }
.auth-brand p {
    margin: 0;
    color: #64748b;
    font-size: 0.9rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2.25rem;
    border-bottom: 1px solid #e5e7eb;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
    padding-left: 0.25rem !important;
    padding-right: 0.25rem !important;
}
.stTabs [aria-selected="true"] {
    color: #1D4ED8 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #1D4ED8 !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextInput input {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput label, .stForm label {
    color: #1a1a1a !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Primary form-submit button (Sign in / Sign up / Reset) */
.stForm [data-testid="stFormSubmitButton"] button {
    background: #1D4ED8 !important;
    color: #ffffff !important;
    border: 0 !important;
    border-radius: 8px !important;
    padding: 0.65rem 1rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    transition: background 120ms ease;
}
.stForm [data-testid="stFormSubmitButton"] button:hover {
    background: #1e40af !important;
}

/* Secondary outlined button (e.g. "Try Demo") — scoped to the auth card
   column so it only affects buttons rendered on this screen. */
[data-testid="column"]:has(.auth-card-anchor) .stButton > button {
    background: #ffffff !important;
    color: #1D4ED8 !important;
    border: 1px solid #1D4ED8 !important;
    border-radius: 8px !important;
    padding: 0.65rem 1rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    transition: background 120ms ease, color 120ms ease;
}
[data-testid="column"]:has(.auth-card-anchor) .stButton > button:hover {
    background: #F0F4FF !important;
    color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

/* Divider */
.auth-divider {
    display: flex;
    align-items: center;
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    margin: 1.1rem 0 0.6rem 0;
    font-family: 'DM Sans', sans-serif;
}
.auth-divider::before, .auth-divider::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #e5e7eb;
}
.auth-divider:not(:empty)::before { margin-right: .75em; }
.auth-divider:not(:empty)::after { margin-left: .75em; }

/* Password strength bar */
.pw-strength {
    height: 6px;
    border-radius: 4px;
    background: #e5e7eb;
    overflow: hidden;
    margin-top: 0.35rem;
}
.pw-strength > div {
    height: 100%;
    transition: width 200ms ease, background 200ms ease;
}
.pw-strength-label {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.25rem;
    font-family: 'DM Sans', sans-serif;
}
</style>
"""


def _password_strength(password: str) -> tuple[int, str, str]:
    """Return (score 0-5, label, hex color)."""
    if not password:
        return 0, "", "#e5e7eb"
    score = 0
    if len(password) >= 8:
        score += 1
    if len(password) >= 12:
        score += 1
    if re.search(r"[A-Z]", password) and re.search(r"[a-z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[^A-Za-z0-9]", password):
        score += 1
    label, color = {
        0: ("Too short", "#ef4444"),
        1: ("Weak", "#ef4444"),
        2: ("Fair", "#f59e0b"),
        3: ("Good", "#3b82f6"),
        4: ("Strong", "#10b981"),
        5: ("Very strong", "#059669"),
    }[score]
    return score, label, color


def _render_strength_bar(password: str) -> None:
    score, label, color = _password_strength(password)
    pct = int((score / 5) * 100)
    st.markdown(
        f"""
        <div class="pw-strength"><div style="width:{pct}%;background:{color};"></div></div>
        <div class="pw-strength-label">{label}</div>
        """,
        unsafe_allow_html=True,
    )


def _validate_signup(full_name: str, email: str, username: str, password: str, confirm: str) -> tuple[bool, str]:
    if not all([full_name, email, username, password, confirm]):
        return False, "Please fill in every field."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Please enter a valid email address."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if password != confirm:
        return False, "Passwords do not match."
    return True, ""


def _render_sign_in() -> None:
    with st.form("auth_signin_form", clear_on_submit=False):
        username = st.text_input("Username", key="signin_username", placeholder="your.username")
        password = st.text_input("Password", key="signin_password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In")
        if submitted:
            user = auth_db.verify_password(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.username = user["username"]
                st.success(f"Welcome back, {user['full_name'] or user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # "or" divider between Sign In and the Try Demo shortcut.
    st.markdown('<div class="auth-divider">or</div>', unsafe_allow_html=True)

    # Demo entry — no sign-up required. Outlined secondary style is
    # provided by the auth-card-scoped CSS in _AUTH_CSS.
    if st.button(
        "Try Demo — No sign up needed",
        key="auth_try_demo_btn",
        use_container_width=True,
    ):
        st.session_state.authenticated = True
        st.session_state.username = "demo"
        st.rerun()


def _render_sign_up() -> None:
    pw_value = st.session_state.get("signup_password", "")

    with st.form("auth_signup_form", clear_on_submit=False):
        full_name = st.text_input("Full name", key="signup_full_name")
        email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
        username = st.text_input("Username", key="signup_username")
        password = st.text_input("Password", key="signup_password", type="password")
        if password:
            _render_strength_bar(password)
        confirm = st.text_input("Confirm password", key="signup_confirm", type="password")
        submitted = st.form_submit_button("Create Account")
        if submitted:
            ok, msg = _validate_signup(full_name, email, username, password, confirm)
            if not ok:
                st.error(msg)
            else:
                created, message = auth_db.create_user(full_name, email, username, password)
                if created:
                    st.success(message + " You can now sign in.")
                else:
                    st.error(message)

    # Re-render strength outside the form too if there was a submit error and
    # the user is still typing (Streamlit reruns on every keystroke).
    if not pw_value and st.session_state.get("signup_password"):
        _render_strength_bar(st.session_state["signup_password"])


def _render_forgot_password() -> None:
    with st.form("auth_forgot_form", clear_on_submit=False):
        email = st.text_input("Account email", key="forgot_email", placeholder="you@example.com")
        submitted = st.form_submit_button("Send Reset Instructions")
        if submitted:
            # Always create a token if the email exists; show the same generic
            # message either way to avoid leaking whether an email is registered.
            auth_db.create_reset_token(email)
            st.success("If this email is registered, reset instructions have been sent.")


def render_auth_page() -> None:
    """Render the unified Sign In / Sign Up / Forgot Password screen."""
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            """
            <span class="auth-card-anchor"></span>
            <div class="auth-brand">
                <div class="logo">🏥</div>
                <h1>MediAid AI</h1>
                <p>Your evidence-based medical companion</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        signin_tab, signup_tab, forgot_tab = st.tabs(["Sign In", "Sign Up", "Forgot Password"])
        with signin_tab:
            _render_sign_in()
        with signup_tab:
            _render_sign_up()
        with forgot_tab:
            _render_forgot_password()
