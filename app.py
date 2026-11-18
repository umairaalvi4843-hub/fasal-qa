
import sys
import html
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag_answer import generate_answer

st.set_page_config(
    page_title="Fasal QA | Farmer Companion",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- SESSION STATE ----------------

defaults = {
    "question": "",
    "answer": None,
    "history": [],
    "selected_scheme": "All schemes",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- PASTEL DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
    --cream: #FAF9F4;
    --ink: #293D31;
    --muted: #586C5D;
    --sage: #E5EDDF;
    --green: #456C4D;
    --border: #DCE5D7;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] {
    display: none;
}

#MainMenu,
footer {
    visibility: hidden;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

.block-container {
    max-width: 1450px;
    padding: 2rem 2rem 3rem;
}

h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--ink) !important;
    font-weight: 800 !important;
}

p, label, li, .stMarkdown {
    color: var(--ink);
    font-size: 15px;
    line-height: 1.65;
}

/* Left panel */

.left-panel {
    background: #F0F2E9;
    border: 1px solid #E0E6D9;
    border-radius: 25px;
    padding: 23px 18px;
}

.brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 27px;
    font-weight: 800;
    color: #365B3F;
}

.brand-subtitle {
    font-size: 13px;
    color: #617363;
    margin-top: 2px;
}

.side-heading {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 17px;
    font-weight: 800;
    color: #304A36;
    margin: 24px 0 10px;
}

.side-note {
    color: #586C5D;
    font-size: 13px;
    margin-bottom: 12px;
}

/* Right panel */

.hero {
    background: linear-gradient(120deg, #E5EFDF, #F4EEDB);
    border: 1px solid #DDE8D6;
    border-radius: 25px;
    padding: 27px;
    margin-bottom: 22px;
}

.hero-tag {
    color: #527454;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.3px;
}

.hero h1 {
    color: #294B35 !important;
    font-size: clamp(27px, 3vw, 39px) !important;
    margin: 9px 0;
}

.hero p {
    color: #435F49;
    font-size: 16px;
    max-width: 650px;
}

.hero-art {
    font-size: 65px;
    text-align: center;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #304A36;
    margin: 20px 0 4px;
}

.section-subtitle {
    font-size: 14px;
    color: #596D5D;
    margin-bottom: 15px;
}

/* Cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 19px;
    border-color: #DCE5D7;
    background: #FFFFFF;
}

div[data-testid="stButton"] > button {
    min-height: 49px;
    width: 100%;
    border-radius: 14px;
    border: 1.5px solid #D7E2D2;
    background: #FFFFFF;
    color: #304C37;
    font-weight: 700;
    font-size: 14px;
    transition: all 0.15s ease;
}

div[data-testid="stButton"] > button:hover {
    background: #E7F0E2;
    border-color: #8EAD8D;
    color: #294B35;
    transform: translateY(-1px);
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: #456C4D;
    border: none;
    color: #FFFFFF;
    min-height: 54px;
    font-size: 16px;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #31573A;
    color: #FFFFFF;
}

/* Inputs */

.stTextArea textarea {
    background: #FFFFFF;
    color: #263D2E;
    border: 1.5px solid #D8E3D4;
    border-radius: 16px;
    font-size: 16px;
    line-height: 1.6;
    padding: 15px;
}

.stTextArea textarea::placeholder {
    color: #718174;
    opacity: 1;
}

.stTextArea textarea:focus {
    border-color: #6F956F;
    box-shadow: 0 0 0 2px #E4EEE0;
}

div[data-baseweb="select"] > div {
    background: #FFFFFF;
    border-radius: 12px;
    color: #293D31;
    border-color: #D8E3D4;
}

div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E1E7DC;
    border-radius: 15px;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

.answer-panel {
    background: #FFFFFF;
    border: 1px solid #DDE7D9;
    border-radius: 20px;
    padding: 21px;
    margin: 14px 0;
}

.answer-tag {
    color: #47734F;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.answer-text {
    background: #EDF3E9;
    border-radius: 17px;
    padding: 22px;
    color: #293D31;
    line-height: 1.8;
    font-size: 16px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.footer {
    background: #E8EEDF;
    color: #405C45;
    border-radius: 19px;
    text-align: center;
    padding: 20px;
    margin-top: 25px;
    font-size: 13px;
}

hr {
    border-color: #E2E7DC;
}

@media(max-width: 850px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .left-panel {
        padding: 18px;
    }

    .hero {
        padding: 21px;
    }

    .hero-art {
        font-size: 45px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- SCHEME CONTENT ----------------

schemes = {
    "PM-KISAN": {
        "icon": "🌾",
        "subtitle": "Income support",
        "description": "Benefits, instalments and eligibility",
        "color": "#F8EACB",
        "question": "How much annual financial assistance does PM-KISAN provide?",
    },
    "PMFBY": {
        "icon": "🌦️",
        "subtitle": "Crop protection",
        "description": "Crop losses, premiums and claims",
        "color": "#DDEDE5",
        "question": "What crop losses are covered under PMFBY?",
    },
    "KCC": {
        "icon": "🏦",
        "subtitle": "Farmer credit",
        "description": "Credit, eligibility and loan limits",
        "color": "#F4DFD8",
        "question": "Who is eligible to apply for a Kisan Credit Card?",
    },
}


# ---------------- ENGLISH QUESTIONS ----------------

quick_questions = [
    ("🌾", "How much assistance does PM-KISAN provide?", "PM-KISAN"),
    ("🌧️", "What happens if crops are damaged by heavy rainfall?", "PMFBY"),
    ("🏦", "Can tenant farmers apply for KCC?", "KCC"),
    ("📋", "How can farmers submit a crop insurance application?", "PMFBY"),
    ("💰", "How are KCC loan limits calculated?", "KCC"),
    ("📅", "When are PM-KISAN instalments paid?", "PM-KISAN"),
]


# ---------------- HINDI QUESTIONS ----------------

hindi_questions = [
    ("🌾", "पीएम-किसान योजना के तहत किसानों को सालाना कितनी आर्थिक सहायता मिलती है?", "PM-KISAN"),
    ("📅", "पीएम-किसान की किस्तें किसानों के बैंक खाते में कब जमा की जाती हैं?", "PM-KISAN"),
    ("🌱", "पीएम-किसान योजना का लाभ लेने के लिए किसान की पात्रता क्या है?", "PM-KISAN"),
    ("🌧️", "भारी बारिश से फसल खराब होने पर पीएमएफबीवाई के तहत मुआवजा कैसे मिलता है?", "PMFBY"),
    ("🌾", "पीएमएफबीवाई के तहत फसल कटाई के बाद होने वाले नुकसान का कवरेज कितने दिनों तक मिलता है?", "PMFBY"),
    ("🐛", "क्या कीटों और बीमारियों से फसल को हुए नुकसान का बीमा कवर मिलता है?", "PMFBY"),
    ("🏦", "क्या बटाईदार और मौखिक पट्टेदार किसान किसान क्रेडिट कार्ड के लिए आवेदन कर सकते हैं?", "KCC"),
    ("💰", "किसान क्रेडिट कार्ड के तहत ऋण की सीमा कैसे निर्धारित की जाती है?", "KCC"),
    ("📋", "किसान क्रेडिट कार्ड के लिए आवेदन करते समय कौन-कौन से दस्तावेज़ आवश्यक होते हैं?", "KCC"),
]


# ---------------- HINGLISH QUESTIONS ----------------

hinglish_questions = [
    ("🌾", "PM-KISAN ke tahat farmers ko har saal kitni financial assistance milti hai?", "PM-KISAN"),
    ("📅", "PM-KISAN ki instalments farmers ke bank account mein kab aati hain?", "PM-KISAN"),
    ("🌱", "PM-KISAN scheme ka benefit lene ke liye farmer ki eligibility kya hai?", "PM-KISAN"),
    ("🌧️", "Agar heavy rainfall ki wajah se crop damage ho jaye toh PMFBY ke tahat claim kaise milta hai?", "PMFBY"),
    ("🌾", "PMFBY mein harvesting ke baad crop loss ka coverage kitne din tak milta hai?", "PMFBY"),
    ("🐛", "Kya pests aur diseases ki wajah se crop damage hone par insurance cover milta hai?", "PMFBY"),
    ("🏦", "Kya tenant farmers aur sharecroppers Kisan Credit Card ke liye apply kar sakte hain?", "KCC"),
    ("💰", "KCC ke under farmer ka loan limit kaise decide hota hai?", "KCC"),
    ("📋", "Kisan Credit Card apply karne ke liye kaunse documents required hote hain?", "KCC"),
]


# ---------------- CALLBACKS ----------------

def select_question(question, scheme):
    st.session_state.question = question
    st.session_state.selected_scheme = scheme
    st.session_state.answer = None


def clear_answer():
    st.session_state.answer = None


def clear_history():
    st.session_state.history = []


def submit_question():
    question = st.session_state.question.strip()

    if not question:
        st.session_state.answer = {
            "question": "",
            "answer": "Please type a question before requesting an answer.",
            "mode": "",
            "passages": [],
        }
        return

    try:
        with st.spinner("🌱 Searching official scheme documents..."):
            response = generate_answer(question)

        result = {
            "question": question,
            "answer": response.get("answer", "No answer generated."),
            "mode": response.get("mode", ""),
            "passages": response.get("retrieved_passages", []),
        }

        st.session_state.answer = result
        st.session_state.history.insert(0, result)
        st.session_state.history = st.session_state.history[:10]

    except Exception as error:
        st.session_state.answer = {
            "question": question,
            "answer": f"Unable to generate an answer: {error}",
            "mode": "",
            "passages": [],
        }


# ================= MAIN TWO-COLUMN LAYOUT =================

left, right = st.columns([1.05, 2.45], gap="large")


# ================= LEFT COLUMN =================

with left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="brand">🌱 Fasal QA</div>'
        '<div class="brand-subtitle">Your farmer information companion</div>',
        unsafe_allow_html=True,
    )

    # Scheme cards

    st.markdown(
        '<div class="side-heading">Explore schemes</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-note">Choose a scheme to start asking questions.</div>',
        unsafe_allow_html=True,
    )

    for name, details in schemes.items():
        st.markdown(
            f"""
            <div style="
                background:{details['color']};
                border-radius:16px;
                padding:15px;
                margin:12px 0 8px;
                border:1px solid #E1E5D9;">
                <div style="font-size:30px;">{details['icon']}</div>
                <div style="font-size:18px;font-weight:800;color:#304A36;">{name}</div>
                <div style="font-size:13px;font-weight:700;color:#405C45;">{details['subtitle']}</div>
                <div style="font-size:12px;color:#586C5D;">{details['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            f"Explore {name} →",
            key=f"scheme_{name}",
            use_container_width=True,
            on_click=select_question,
            args=(details["question"], name),
        )

    # English questions

    st.markdown(
        '<div class="side-heading">Popular questions · English</div>',
        unsafe_allow_html=True,
    )

    for index, (icon, question, scheme) in enumerate(quick_questions):
        st.button(
            f"{icon}  {question}",
            key=f"quick_{index}",
            use_container_width=True,
            on_click=select_question,
            args=(question, scheme),
        )

    # Hindi questions

    st.markdown(
        '<div class="side-heading">🇮🇳 हिंदी में सवाल पूछें</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-note">किसी सवाल पर क्लिक करें। वह प्रश्न बॉक्स में आ जाएगा।</div>',
        unsafe_allow_html=True,
    )

    for index, (icon, question, scheme) in enumerate(hindi_questions):
        st.button(
            f"{icon}  {question}",
            key=f"hindi_quick_{index}",
            use_container_width=True,
            on_click=select_question,
            args=(question, scheme),
        )

    # Hinglish questions

    st.markdown(
        '<div class="side-heading">💬 Hinglish mein questions</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-note">Roman English mein likhe questions par click karein.</div>',
        unsafe_allow_html=True,
    )

    for index, (icon, question, scheme) in enumerate(hinglish_questions):
        st.button(
            f"{icon}  {question}",
            key=f"hinglish_quick_{index}",
            use_container_width=True,
            on_click=select_question,
            args=(question, scheme),
        )

    # Language note

    st.markdown(
        """
        <div style="
            background:#E7EDE1;
            padding:15px;
            border-radius:15px;
            margin-top:20px;
            color:#405C45;
            font-size:13px;">
            <b>🌱 Ask in your language</b><br>
            English, हिन्दी or Hinglish.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ================= RIGHT COLUMN =================

with right:

    # Hero

    hero_left, hero_right = st.columns([3, 1])

    with hero_left:
        st.markdown("""
        <div class="hero">
            <div class="hero-tag">YOUR DIGITAL FARMING COMPANION</div>
            <h1>Namaste, Kisan! 🌱</h1>
            <p>
                Your farming schemes, explained simply.
                Ask questions in English, Hindi or Hinglish and
                discover information from official scheme documents.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        st.markdown("""
        <div class="hero" style="height:150px;display:flex;align-items:center;justify-content:center;">
            <div class="hero-art">🌱<br>🌾☀️</div>
        </div>
        """, unsafe_allow_html=True)

    # Question input

    st.markdown(
        '<div class="section-title">💬 How can we help you?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Type your question below. Ask naturally, just as you would ask another person.</div>',
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Choose a scheme",
        ["All schemes", "PM-KISAN", "PMFBY", "KCC"],
        key="selected_scheme",
    )

    st.text_area(
        "Your question",
        key="question",
        height=125,
        placeholder="Example: PM-KISAN ke tahat kitne paise milte hain?",
    )

    st.button(
        "Get my answer  🌱",
        key="submit_question",
        type="primary",
        use_container_width=True,
        on_click=submit_question,
    )

    # Answer section

    if st.session_state.answer:
        result = st.session_state.answer

        st.markdown("---")
        st.markdown(
            '<div class="section-title">Your answer</div>',
            unsafe_allow_html=True,
        )

        safe_question = html.escape(result["question"])

        if safe_question:
            st.markdown(
                f"""
                <div class="answer-panel">
                    <div class="answer-tag">YOUR QUESTION</div>
                    <h3>{safe_question}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        safe_answer = html.escape(result["answer"])

        st.markdown(
            f'<div class="answer-text">{safe_answer}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📄 View retrieved supporting passages"):
            if result["passages"]:
                for passage in result["passages"]:
                    st.write(passage)
            else:
                st.write("No passage references returned.")

            st.caption(
                "Retrieved passages are evidence supplied to the answer generator."
            )

        st.button(
            "Clear answer",
            key="clear_answer",
            on_click=clear_answer,
        )

    # Recent questions

    if st.session_state.history:
        st.markdown("---")
        st.markdown(
            '<div class="section-title">🕘 Recent questions</div>',
            unsafe_allow_html=True,
        )

        for index, item in enumerate(st.session_state.history[:5]):
            with st.expander(item["question"]):
                st.write(item["answer"])

        st.button(
            "Clear history",
            key="clear_history",
            on_click=clear_history,
        )

    # Footer

    st.markdown("""
    <div class="footer">
        <div style="font-size:25px;">🌱</div>
        <b>Fasal QA</b><br>
        Making agricultural scheme information easier to understand.
        <br><br>
        Independent educational project · Not an official government portal
    </div>
    """, unsafe_allow_html=True)
