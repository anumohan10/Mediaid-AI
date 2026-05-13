"""Google OAuth placeholder.

The real OAuth dance isn't wired up yet — this just renders a styled button so
the auth screen has the expected affordance. Clicking it surfaces a
"coming soon" toast.
"""

from __future__ import annotations

import streamlit as st

_GOOGLE_BUTTON_CSS = """
<style>
.google-btn-wrapper { margin-top: 0.5rem; }
.google-btn-wrapper .stButton > button {
    width: 100%;
    background: #ffffff !important;
    color: #1f2937 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 0.65rem 1rem !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.5rem !important;
    transition: background 120ms ease, box-shadow 120ms ease;
}
.google-btn-wrapper .stButton > button:hover {
    background: #f9fafb !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.google-btn-wrapper .stButton > button p::before {
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 8px;
    vertical-align: -3px;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'><path fill='%23FFC107' d='M43.6 20.5H42V20H24v8h11.3c-1.6 4.7-6.1 8-11.3 8-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3l5.7-5.7C34.5 6.5 29.5 4.5 24 4.5 12.7 4.5 3.5 13.7 3.5 25S12.7 45.5 24 45.5 44.5 36.3 44.5 25c0-1.6-.2-3.1-.9-4.5z'/><path fill='%23FF3D00' d='M6.3 14.7l6.6 4.8C14.5 16 18.9 13 24 13c3.1 0 5.8 1.2 7.9 3l5.7-5.7C34.5 6.5 29.5 4.5 24 4.5 16.4 4.5 9.8 8.6 6.3 14.7z'/><path fill='%234CAF50' d='M24 45.5c5.4 0 10.3-2 13.9-5.3l-6.4-5.3C29.6 36.3 26.9 37 24 37c-5.2 0-9.6-3.3-11.3-7.9l-6.6 5.1C9.6 41 16.3 45.5 24 45.5z'/><path fill='%231976D2' d='M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.3 4.3-4.3 5.7l6.4 5.3C40.7 36 44.5 30.9 44.5 25c0-1.6-.2-3.1-.9-4.5z'/></svg>");
    background-size: contain;
    background-repeat: no-repeat;
}
</style>
"""


def render_google_button(key: str = "google_oauth_btn") -> None:
    """Render the 'Continue with Google' button. Clicking shows a toast."""
    st.markdown(_GOOGLE_BUTTON_CSS, unsafe_allow_html=True)
    st.markdown('<div class="google-btn-wrapper">', unsafe_allow_html=True)
    if st.button("Continue with Google", key=key, use_container_width=True):
        st.info("Google OAuth coming soon")
    st.markdown("</div>", unsafe_allow_html=True)
