"""
capstone_streamlit.py — Streamlit UI for MediCare Hospital Assistant.

Launch: streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid
from medicare_assistant.graph import build_graph, ask

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediCare Hospital Assistant",
    page_icon="🏥",
    layout="centered",
)

# ── Cache expensive resources (loaded ONCE per session) ────────────────────────
@st.cache_resource
def load_agent():
    """Build the LangGraph app, embedder, and ChromaDB collection once."""
    app, embedder, collection = build_graph()
    return app


# ── Session state init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

app = load_agent()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("MediCare Hospital")
    st.caption("Hyderabad — 350-bed multi-specialty hospital")
    st.divider()
    st.markdown("**Topics I can help with:**")
    topics = [
        "OPD timings & schedules",
        "Appointment booking",
        "Doctor directory",
        "Consultation fees",
        "Insurance & cashless",
        "Emergency services",
        "Pharmacy",
        "Lab & diagnostics",
        "Health packages",
        "Admission & discharge",
    ]
    for t in topics:
        st.markdown(f"• {t}")
    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}…`")

# ── Main header ────────────────────────────────────────────────────────────────
st.title("🏥 MediCare Hospital Assistant")
st.caption(
    "I can answer questions about our services. For medical advice, please consult a doctor. "
    "Emergency: **040-99999999**"
)

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me about OPD timings, appointments, fees, insurance…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Looking that up for you…"):
            result = ask(app, prompt, thread_id=st.session_state.thread_id)
            answer = result.get("answer", "I'm sorry, I couldn't process that. Please try again.")

            # Show debug info (expandable)
            with st.expander("Agent trace", expanded=False):
                st.write(f"**Route:** `{result.get('route', 'N/A')}`")
                st.write(f"**Faithfulness:** `{result.get('faithfulness', 'N/A')}`")
                st.write(f"**Sources:** {result.get('sources', [])}")
                if result.get("eval_retries", 0) > 1:
                    st.warning(f"Answer was retried {result['eval_retries'] - 1} time(s).")

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
