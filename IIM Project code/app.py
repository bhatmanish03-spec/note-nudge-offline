import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Table, MetaData
from sqlalchemy.orm import sessionmaker
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ---------- Configuration ----------
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./notes_nudge.db")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # optional
DEFAULT_SCHEDULE_DAYS = [1, 3, 7, 14]

# ---------- DB setup ----------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine)
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String, unique=True, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow),
)

documents = Table(
    "documents", metadata,
    Column("id", Integer, primary_key=True),
    Column("user_email", String, nullable=False),
    Column("title", String),
    Column("raw_text", Text),
    Column("created_at", DateTime, default=datetime.utcnow),
)

cards = Table(
    "cards", metadata,
    Column("id", Integer, primary_key=True),
    Column("document_id", Integer),
    Column("topic", String),
    Column("summary", Text),
    Column("bullets", JSON),
    Column("question", Text),
    Column("answer", Text),
    Column("created_at", DateTime, default=datetime.utcnow),
)

schedules = Table(
    "schedules", metadata,
    Column("id", Integer, primary_key=True),
    Column("card_id", Integer),
    Column("user_email", String),
    Column("due_date", DateTime),
    Column("interval_days", Integer),
    Column("status", String, default="pending"),
    Column("last_reviewed", DateTime, nullable=True),
)

metadata.create_all(engine)

# ---------- Utilities ----------
def extract_text_from_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        texts = []
        for p in reader.pages:
            texts.append(p.extract_text() or "")
        return "\n\n".join(texts)
    except:
        return ""

def mock_generate_cards(text: str) -> List[Dict[str, Any]]:
    # A simple mock generator (used if no OpenAI key is provided)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = " ".join(lines)
    sentences = [s.strip() for s in joined.replace("?", ".").split(".") if s.strip()]
    cards_list = []
    for i, s in enumerate(sentences[:6]):
        cards_list.append({
            "topic": f"Topic {i+1}",
            "summary": s[:150],
            "bullets": [
                (s[:80] + "..."),
                "Key point: remember this",
                "Example: try a short practice"
            ],
            "question": "What is the main idea of the above?",
            "answer": s[:150]
        })
    return cards_list

def call_openai_generate_cards(text: str) -> List[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return mock_generate_cards(text)

    import openai
    openai.api_key = OPENAI_API_KEY

    prompt = f"""
    Split the following notes into topics.
    For each topic generate:
    - topic
    - summary (1 line)
    - bullets (3 bullets)
    - question
    - answer

    Return JSON list only.
    Notes:
    {text}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return clean JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        raw = response.choices[0].message.content
        start = raw.find("[")
        end = raw.rfind("]") + 1
        data = json.loads(raw[start:end])
        return data
    except:
        return mock_generate_cards(text)

def save_document_and_cards(user_email: str, title: str, raw_text: str, generated_cards: List[Dict[str, Any]]):
    conn = engine.connect()
    result = conn.execute(documents.insert().values(
        user_email=user_email,
        title=title,
        raw_text=raw_text
    ))
    doc_id = result.inserted_primary_key[0]

    for c in generated_cards:
        res = conn.execute(cards.insert().values(
            document_id=doc_id,
            topic=c["topic"],
            summary=c["summary"],
            bullets=json.dumps(c["bullets"]),
            question=c["question"],
            answer=c["answer"],
        ))
        card_id = res.inserted_primary_key[0]

        for days in DEFAULT_SCHEDULE_DAYS:
            due = datetime.utcnow() + timedelta(days=days)
            conn.execute(schedules.insert().values(
                card_id=card_id,
                user_email=user_email,
                due_date=due,
                interval_days=days
            ))
    conn.close()

def fetch_due_cards(user_email):
    conn = engine.connect()
    now = datetime.utcnow()
    res = conn.execute(
        schedules.select()
        .where(schedules.c.user_email == user_email)
        .where(schedules.c.due_date <= now)
        .where(schedules.c.status == "pending")
    )
    rows = res.fetchall()
    card_list = []
    for r in rows:
        card_row = conn.execute(cards.select().where(cards.c.id == r.card_id)).fetchone()
        card_list.append({"schedule_id": r.id, "card": dict(card_row)})
    conn.close()
    return card_list

def mark_reviewed(schedule_id: int, correct: bool):
    conn = engine.connect()
    sched = conn.execute(schedules.select().where(schedules.c.id == schedule_id)).fetchone()
    if not sched:
        conn.close()
        return

    last_interval = sched.interval_days
    next_interval = last_interval * 2 if correct else 1
    next_due = datetime.utcnow() + timedelta(days=next_interval)

    conn.execute(schedules.update().where(schedules.c.id == schedule_id).values(
        status="done", last_reviewed=datetime.utcnow()
    ))

    conn.execute(schedules.insert().values(
        card_id=sched.card_id,
        user_email=sched.user_email,
        due_date=next_due,
        interval_days=next_interval,
        status="pending"
    ))

    conn.close()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Note→Nudge MVP", layout="centered")
st.title("Note→Nudge — Streamlit MVP")

st.sidebar.header("Demo User")
user_email = st.sidebar.text_input("Demo user email", value="student@example.com")

st.header("Step 1 — Enter Notes")
text_input = st.text_area("Paste notes here", height=200)

if st.button("Use Example Notes"):
    text_input = """
Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food.
Occurs in chloroplasts.
Two stages: Light-dependent reactions & Calvin cycle.
"""

st.header("Step 2 — Generate Cards")
if st.button("Generate Cards"):
    if not text_input.strip():
        st.error("Please enter notes first.")
    else:
        with st.spinner("Generating cards..."):
            generated = call_openai_generate_cards(text_input)
            st.session_state["generated"] = generated
        st.success("Generated cards!")

if "generated" in st.session_state:
    st.header("Generated Cards Preview")
    for i, c in enumerate(st.session_state["generated"]):
        st.subheader(f"{i+1}. {c['topic']}")
        st.write(c["summary"])
        for b in c["bullets"]:
            st.write("• " + b)

        st.write("Question: " + c["question"])
        with st.expander("Answer"):
            st.write(c["answer"])

    if st.button("Save Cards & Create Schedule"):
        save_document_and_cards(
            user_email=user_email,
            title=f"Notes {datetime.utcnow()}",
            raw_text=text_input,
            generated_cards=st.session_state["generated"]
        )
        st.success("Saved & scheduled!")

st.header("Step 3 — Today’s Due Cards")
if st.button("Show Due Cards"):
    due_cards = fetch_due_cards(user_email)
    st.session_state["due_cards"] = due_cards

if "due_cards" in st.session_state:
    for item in st.session_state["due_cards"]:
        card = item["card"]
        sid = item["schedule_id"]

        st.subheader(card["topic"])
        st.write(card["summary"])

        st.write("• " + "\n• ".join(json.loads(card["bullets"])))

        st.write("Q: " + card["question"])

        col1, col2 = st.columns(2)
        if col1.button(f"Correct ({sid})"):
            mark_reviewed(sid, True)
            st.experimental_rerun()

        if col2.button(f"Wrong ({sid})"):
            mark_reviewed(sid, False)
            st.experimental_rerun()