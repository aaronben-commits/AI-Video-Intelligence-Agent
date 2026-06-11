import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Core pipeline imports
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")


# ---------------------------
# Backend Pipeline Function
# ---------------------------
def run_pipeline(source: str, language: str = "english") -> dict:
    chunks = process_input(source)

    transcript = transcribe_all(chunks, language)

    title = generate_title(transcript)
    summary = summarize(transcript)

    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .sub-text {
            text-align: center;
            color: gray;
            font-size: 18px;
        }
        .card {
            padding: 20px;
            border-radius: 12px;
            background-color: #1E1E1E;
            margin-bottom: 15px;
            border: 1px solid #333;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Header
# ---------------------------
st.markdown('<div class="main-title">🎥 AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Upload meeting audio/video or paste YouTube link for smart transcription, summaries & Q&A</div>',
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("Choose Language", ["english", "hinglish"])

    st.markdown("### 📌 Supported Inputs")
    st.markdown("- YouTube URL")
    st.markdown("- MP3 / WAV / MP4 local files")

    st.markdown("### 💡 Features")
    st.markdown("- Smart transcription")
    st.markdown("- AI summary")
    st.markdown("- Action items")
    st.markdown("- Key decisions")
    st.markdown("- Open questions")
    st.markdown("- Meeting Q&A")


# ---------------------------
# Input Section
# ---------------------------
input_mode = st.radio(
    "Choose Input Type:",
    ["YouTube URL", "Upload Local File"],
    horizontal=True,
)

source = None

if input_mode == "YouTube URL":
    source = st.text_input("🔗 Enter YouTube URL")
else:
    uploaded_file = st.file_uploader(
        "📂 Upload Audio/Video File",
        type=["mp3", "wav", "mp4", "m4a"],
    )

    if uploaded_file:
        temp_path = Path("temp_uploads")
        temp_path.mkdir(exist_ok=True)

        file_path = temp_path / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        source = str(file_path)


# ---------------------------
# Session State
# ---------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------
# Process Button
# ---------------------------
if st.button("🚀 Run AI Assistant", use_container_width=True):
    if not source:
        st.warning("Please provide a valid input source.")
    else:
        with st.spinner("Processing... This may take a few minutes ⏳"):
            try:
                st.session_state.result = run_pipeline(source, language)
                st.success("✅ Processing Complete!")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ---------------------------
# Results Section
# ---------------------------
if st.session_state.result:
    result = st.session_state.result

    st.divider()

    st.subheader(f"📌 Title: {result['title']}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Summary")
        st.info(result["summary"])

    with col2:
        st.markdown("### ✅ Action Items")
        st.success(result["action_items"])

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🔑 Key Decisions")
        st.warning(result["key_decisions"])

    with col4:
        st.markdown("### ❓ Open Questions")
        st.error(result["open_questions"])

    # Transcript Section
    with st.expander("📝 Full Transcript"):
        st.write(result["transcript"])

    # Download Button
    st.download_button(
        label="📥 Download Transcript",
        data=result["transcript"],
        file_name="transcript.txt",
        mime="text/plain",
    )

    # ---------------------------
    # RAG Chat Section
    # ---------------------------
    st.divider()
    st.subheader("💬 Chat with your meeting")

    user_question = st.text_input("Ask a question about this meeting/video:")

    if st.button("Ask Question"):
        if user_question.strip():
            answer = ask_question(result["rag_chain"], user_question)

            st.session_state.chat_history.append(
                {"question": user_question, "answer": answer}
            )

    # Display Chat History
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {chat['question']}")
        st.markdown(f"**🤖 Assistant:** {chat['answer']}")
        st.divider()