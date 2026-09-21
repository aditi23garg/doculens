"""Global CSS. Paper-and-ink palette; the highlighter yellow is reserved for citations.

get_css(dark) renders the same rules against a light or dark palette. Streamlit's own
[theme] in .streamlit/config.toml only takes effect on a full server restart, so a runtime
toggle (see ui/sidebar.py) can't use it - instead every color is a CSS variable, and this
function just swaps which palette's values go into :root on each rerun.
"""

_LIGHT = {
    "ink": "#16233B",
    "paper": "#F6F7F9",
    "surface": "#FFFFFF",
    "surface-2": "#F0F2F5",
    "rule": "#DCE0E8",
    "muted": "#5B6678",
    "marker": "#FFE27A",
    "marker-soft": "#FFF6CF",
    "marker-text": "#16233B",
    "shadow": "0 1px 2px rgba(22,35,59,.06)",
}

_DARK = {
    "ink": "#E8ECF4",
    "paper": "#10131C",
    "surface": "#1A2030",
    "surface-2": "#212840",
    "rule": "#2C3347",
    "muted": "#96A1B8",
    "marker": "#FFD866",
    "marker-soft": "#2E2A18",
    "marker-text": "#16233B",
    "shadow": "0 1px 3px rgba(0,0,0,.35)",
}


def get_css(dark: bool = False) -> str:
    p = _DARK if dark else _LIGHT
    vars_css = "\n".join(f"  --{k}: {v};" for k, v in p.items())
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

:root {{
{vars_css}
}}

html, body, .stApp, [class*="css"] {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; color: var(--ink); font-size: 17px; }}
.stApp {{ background: var(--paper); }}
h1, h2, h3 {{ font-family: 'Newsreader', Georgia, serif; font-weight: 500; letter-spacing: -0.01em; color: var(--ink); }}
h2 {{ font-size: 2.1rem !important; margin-bottom: .3rem !important; }}
h3 {{ font-size: 1.4rem !important; }}
p, li, label, .stMarkdown {{ font-size: 1.02rem; }}
footer, #MainMenu {{ visibility: hidden; }}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--rule); }}
[data-testid="stSidebar"] * {{ color: var(--ink); }}
.brand {{ display: flex; align-items: center; gap: 10px; font-family: 'Newsreader', Georgia, serif;
         font-size: 1.7rem; font-weight: 600; margin: 4px 0 14px; }}
.brand-mark {{ width: 24px; height: 16px; background: var(--marker); border-radius: 3px;
              box-shadow: inset 0 -3px 0 rgba(0,0,0,.18); display: inline-block; }}
.muted {{ color: var(--muted); font-size: .92rem; }}
[data-testid="stCaptionContainer"] p {{ font-size: 1rem !important; }}
.theme-row {{ display: flex; align-items: center; justify-content: space-between;
             margin: -6px 0 14px; }}

/* Buttons */
.stButton > button, [data-testid="stBaseButton-secondary"] {{
  border-radius: 6px; border: 1px solid var(--rule); background: var(--surface);
  color: var(--ink); font-weight: 500; font-size: 1rem; }}
.stButton > button:hover {{ border-color: var(--ink); color: var(--ink); }}
[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {{
  background: var(--ink); color: var(--paper); border: 1px solid var(--ink); }}
[data-testid="stBaseButton-primary"]:hover {{ opacity: .88; color: var(--paper); }}
:focus-visible {{ outline: 2px solid var(--ink) !important; outline-offset: 2px; }}

/* Text inputs, select boxes, chat input, radios */
.stTextInput input, .stTextArea textarea, [data-testid="stChatInput"] textarea,
[data-baseweb="select"] > div, [data-baseweb="base-input"] {{
  background: var(--surface) !important; color: var(--ink) !important;
  border-color: var(--rule) !important; font-size: 1.02rem !important; }}
[data-testid="stChatInput"] {{ background: var(--surface); border: 1px solid var(--rule); }}
[data-baseweb="select"] span, [data-baseweb="menu"], [data-baseweb="popover"] {{
  background: var(--surface) !important; color: var(--ink) !important; }}
::placeholder {{ color: var(--muted) !important; opacity: 1; }}
[data-testid="stWidgetLabel"] p {{ font-size: 1rem; color: var(--ink); }}
[role="radiogroup"] label p {{ font-size: 1.02rem; }}

/* Alerts */
[data-testid="stAlert"] {{ background: var(--surface-2); border: 1px solid var(--rule); color: var(--ink); }}

/* Expanders */
[data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }}
[data-testid="stExpander"] summary {{ font-size: 1rem; color: var(--ink); }}

/* Tables (Search details) */
[data-testid="stTable"] table, .stDataFrame {{ background: var(--surface); color: var(--ink); font-size: .95rem; }}
[data-testid="stTable"] th {{ background: var(--surface-2) !important; color: var(--ink) !important; }}
[data-testid="stTable"] td {{ border-color: var(--rule) !important; }}

hr, [data-testid="stDivider"] {{ border-color: var(--rule) !important; }}

/* Answers */
[data-testid="stChatMessage"] {{ background: transparent; border: none; padding: .7rem 0; }}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{
  font-family: 'Newsreader', Georgia, serif; font-size: 1.22rem; line-height: 1.68; }}
.cite {{ display: inline-block; background: var(--marker); color: var(--marker-text);
        font: 600 .76rem/1.5 'IBM Plex Sans', sans-serif; padding: 0 .4em; margin: 0 .12em;
        border-radius: 3px; vertical-align: baseline; }}
.note {{ border: 1px dashed var(--rule); background: var(--surface); color: var(--muted);
        border-radius: 6px; padding: .8rem 1rem; font-family: 'IBM Plex Sans', sans-serif; font-size: 1.02rem; }}
.src-head {{ font-weight: 600; font-size: .95rem; margin: .8rem 0 .35rem; color: var(--ink); }}
.src-score {{ color: var(--muted); font-weight: 400; margin-left: .4rem; }}
.excerpt {{ border-left: 3px solid var(--marker); background: var(--marker-soft); color: var(--ink);
           padding: .7rem .95rem; border-radius: 0 5px 5px 0;
           font-family: 'Newsreader', Georgia, serif; font-size: 1.08rem; line-height: 1.6; white-space: pre-wrap; }}

/* Ask page header */
.ask-caption {{ font-size: 1.02rem; color: var(--muted); margin-bottom: 1.1rem; }}

/* Documents */
.doc-name {{ font-weight: 600; word-break: break-word; font-size: 1.05rem; }}
.doc-meta {{ color: var(--muted); font-size: .9rem; }}
.empty {{ border: 1px dashed var(--rule); background: var(--surface); border-radius: 10px;
         padding: 2rem 1.8rem; margin-top: .8rem; box-shadow: var(--shadow); }}
.empty h3 {{ margin: 0 0 .4rem; }}

/* Login */
.login-title {{ font-family: 'Newsreader', Georgia, serif; font-size: 2.5rem; font-weight: 600; margin: 1.5rem 0 .2rem; }}
.login-sub {{ color: var(--muted); margin-bottom: 1.2rem; max-width: 36ch; font-size: 1.05rem; }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""
