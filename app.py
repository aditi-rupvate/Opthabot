import os
import re
import uuid
import time
import base64
import streamlit as st
from fpdf import FPDF
import fitz # PyMuPDF
from langchain.agents import AgentExecutor, create_react_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.tools import StructuredTool
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from langchain.memory import ConversationBufferMemory

# --- 1. Configuration ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_DEFAULT_API_KEY_HERE")
FAISS_INDEX_PATH = "oxford_handbook_kb"
TEMP_STORAGE_PATH = "temp_user_docs"
CHEATSHEET_PATH = "downloads"
os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
os.makedirs(CHEATSHEET_PATH, exist_ok=True)

# --- DISCLAIMER ---
disclaimer_text = "— Note: This output is for academic purposes only and must not be used for clinical diagnosis."

# --- Backend Components ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# --- Text-to-Speech: mobile-safe (returns base64) ---
def text_to_audio_b64(text: str, tld: str) -> str | None:
    try:
        tts = gTTS(text=text, lang='en', tld=tld, slow=False)
        audio_filename = os.path.join(CHEATSHEET_PATH, f"response_{uuid.uuid4()}.mp3")
        tts.save(audio_filename)
        with open(audio_filename, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        return b64
    except Exception as e:
        st.warning(f"Could not generate audio response: {e}")
        return None

def render_audio_player_b64(audio_b64: str):
    # This audio player will autoplay because it's triggered after a user gesture.
    audio_html = f"""
    <audio autoplay style="display:none;">
      <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# --- PDF Generation Class & Function ---
class PDF(FPDF):
    def __init__(self, topic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topic = topic
    def header(self):
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Ophthalmology Cheatsheet: {self.topic.title()}", 0, 0, 'L'); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font("DejaVu", "", 8); self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, 'C')

def create_formatted_pdf(text_content: str, topic: str) -> str:
    pdf = PDF(topic)
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
    except RuntimeError:
        st.error("Could not find 'DejaVuSans.ttf' or 'DejaVuSans-Bold.ttf'."); return ""
    pdf.alias_nb_pages(); pdf.add_page(); pdf.set_margins(15, 15, 15); pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_font("DejaVu", "B", 20); pdf.set_text_color(40, 40, 40); pdf.multi_cell(0, 10, f"Cheatsheet: {topic.title()}", 0, 'C'); pdf.ln(2)
    pdf.set_draw_color(200, 200, 200); pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y()); pdf.ln(10)
    pdf.set_text_color(50, 50, 50)
    for line in text_content.split('\n'):
        line = line.strip()
        if not line: continue
        if line.startswith('## '):
            pdf.set_font("DejaVu", "B", 14); pdf.set_text_color(0, 80, 150)
            pdf.multi_cell(0, 7, line.replace('## ', ''), 0, 'L'); pdf.set_text_color(50, 50, 50); pdf.ln(2)
        elif line.startswith('- '):
            pdf.set_font("DejaVu", "", 11); pdf.set_x(20)
            pdf.multi_cell(0, 7, f"• {line.replace('- ', '', 1)}"); pdf.ln(1)
        else:
            pdf.set_font("DejaVu", "", 11); pdf.multi_cell(0, 7, line.strip())
    pdf.ln(5); pdf.set_draw_color(200, 200, 200); pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y()); pdf.ln(4)
    pdf.set_font("DejaVu", "", 8); pdf.set_text_color(120, 120, 120); pdf.multi_cell(0, 6, disclaimer_text)
    filename = f"{re.sub(r'[^a-zA-Z0-9]', '_', topic).lower()}_cheatsheet.pdf"
    pdf.output(os.path.join(CHEATSHEET_PATH, filename))
    return filename

# --- Main Query Logic ---
def get_bot_response(query: str, chat_history: list, session_id: str = None):
    if session_id:
        db_path = os.path.join(TEMP_STORAGE_PATH, session_id)
        if not os.path.exists(db_path): return "Error: Your document session has expired.", None
    else:
        db_path = FAISS_INDEX_PATH
        if not os.path.exists(db_path): return "Error: Default knowledge base not available.", None
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def tool_chain(query: str, template: str):
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(query)])
        prompt = PromptTemplate.from_template(template)
        # Using .invoke().content to get the string output directly from the LLM
        return (prompt | llm ).invoke({"context": context, "question": query}).content

    def cheatsheet_generator_func(topic: str) -> str:
        cheatsheet_text = tool_chain(topic, "Create a detailed cheat sheet for {question} using '##' for headings and '-' for list items.\nContext: {context}\nCheat Sheet:")
        pdf_filename = create_formatted_pdf(cheatsheet_text, topic)
        return f"PDF_GENERATED::{pdf_filename}::{cheatsheet_text}"

    tools = [
        StructuredTool.from_function(lambda q: tool_chain(q, "Context: {context}\n\nQuestion: {question}\n\nAnswer:"), name="QuestionAnswerTool", description="Use for direct, specific questions."),
        StructuredTool.from_function(lambda t: tool_chain(t, "Provide a comprehensive explanation for {question}.\n\nContext: {context}\nResponse:"), name="ConceptExplainerTool", description="Use for summaries or explanations in the chat."),
        StructuredTool.from_function(cheatsheet_generator_func, name="CheatsheetGeneratorTool", description="Use ONLY when explicitly asked for a downloadable PDF or 'cheat sheet'.")
    ]
    
    prompt = hub.pull("hwchase17/react-chat").partial(system_message="You are an expert ophthalmology assistant. Your purpose is to answer questions strictly related to ophthalmology or the provided documents. If the user asks a question that is outside of this scope, you must politely decline and state that you can only answer questions about ophthalmology.")
    agent = create_react_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='output')
    for msg in chat_history:
        if "user" in msg: memory.chat_memory.add_user_message(msg["user"])
        else: memory.chat_memory.add_ai_message(re.sub(r'<.*?>', '', msg["bot"]))

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False, handle_parsing_errors=True)
    response = agent_executor.invoke({"input": query})
    final_answer = response.get('output', "I couldn't find an answer.")
    pdf_filename = None
    if "PDF_GENERATED::" in final_answer:
        _, pdf_filename, final_answer = final_answer.split("::", 2)
    return final_answer, pdf_filename

# --- Streamlit UI Setup ---
st.set_page_config(layout="centered", page_title="Ophthalmology AI")
THEME_LIGHT = {"bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd", "text": "#191b22"}
THEME_DARK = {"bg": "#18181c", "bar": "#202126", "bot": "#232733", "user": "#22577a", "text": "#f3f5f8"}
for key, val in [("theme", "dark"), ("chat_history", []), ("session_id", None), 
                 ("active_doc_name", None), ("voice_enabled", False), 
                 ("input_accent", 'en-US'), ("output_accent", 'com'), 
                 ("session_started", False), ("audio_to_play", None), ("processing", False)]:
    if key not in st.session_state: st.session_state[key] = val
THEME = THEME_DARK if st.session_state.theme == "dark" else THEME_LIGHT

# --- "Initial Interaction" Gate for Mobile Audio ---
if not st.session_state.session_started:
    st.title("Ophthalmology AI Assistant")
    st.markdown("Tap the button below to start your session. This one-time action enables voice features on mobile devices.")
    if st.button("Start Session", use_container_width=True, type="primary"):
        st.session_state.session_started = True
        st.rerun()
else:
    # --- Main Application UI ---
    with st.sidebar:
        st.header("Settings")
        if st.toggle("Dark Mode", value=st.session_state.theme == "dark"): st.session_state.theme = "dark"
        else: st.session_state.theme = "light"
        st.divider(); st.header("Voice Settings")
        st.session_state.voice_enabled = st.toggle("Enable Voice Chat", value=st.session_state.voice_enabled)
        if st.session_state.voice_enabled:
            accent_map = {'American (US)': 'en-US', 'British (UK)': 'en-GB', 'Indian': 'en-IN'}
            output_map = {'American (US)': 'com', 'British (UK)': 'co.uk', 'Indian': 'co.in'}
            selected_input = st.selectbox("Your Accent", list(accent_map.keys()))
            selected_output = st.selectbox("Assistant's Accent", list(output_map.keys()))
            st.session_state.input_accent, st.session_state.output_accent = accent_map[selected_input], output_map[selected_output]

    st.markdown(f"""<style>.stApp{{background:{THEME['bg']};color:{THEME['text']}}}.topbar-custom{{background:{THEME['bar']};color:{THEME['text']};border-radius:16px;padding:1.3em 1.2em 1.15em 2.1em;margin-bottom:1.6em;font-size:1.55rem;font-weight:800}}div[data-testid="stChatMessage"]{{background-color:transparent;}}.note-text{{color:#888;font-size:0.9rem}}</style>""", unsafe_allow_html=True)
    st.markdown("<div class='topbar-custom'>Ophthalmology AI Assistant</div>", unsafe_allow_html=True)

    # --- ARCHITECTURALLY CORRECT CHAT FLOW ---
    # 1. Handle bot response generation if the last message was from the user and we are not already processing.
    if st.session_state.chat_history and "user" in st.session_state.chat_history[-1] and not st.session_state.processing:
        st.session_state.processing = True
        user_prompt = st.session_state.chat_history[-1]["user"]
        with st.spinner("Thinking..."):
            answer, pdf_filename = get_bot_response(user_prompt, st.session_state.chat_history, st.session_state.get("session_id"))
            full_answer_html = f"{answer}<br><span class='note-text'>{disclaimer_text}</span>"
            bot_message = {"bot": full_answer_html, "pdf_filename": pdf_filename}
            if st.session_state.voice_enabled:
                clean_text = re.sub(r'<.*?>', '', answer).replace('`', '').replace('*', '')
                spoken_text = clean_text + " Is there anything else I can help with?"
                audio_b64 = text_to_audio_b64(spoken_text, st.session_state.output_accent)
                if audio_b64:
                    st.session_state.audio_to_play = audio_b64
            st.session_state.chat_history.append(bot_message)
        st.session_state.processing = False
        st.rerun()

    # 2. Display the entire chat history
    for message in st.session_state.chat_history:
        role = "user" if "user" in message else "bot"
        with st.chat_message(role, avatar="👤" if role == "user" else "🧑‍⚕️"):
            st.markdown(message[role], unsafe_allow_html=True)
            if role == 'bot' and message.get('pdf_filename'):
                pdf_path = os.path.join(CHEATSHEET_PATH, message['pdf_filename'])
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 Download Cheatsheet", f, message['pdf_filename'], "application/pdf", key=f"dl_{message['pdf_filename']}_{uuid.uuid4()}")

    # 3. Play any pending audio at a stable point in the script run
    if st.session_state.audio_to_play:
        render_audio_player_b64(st.session_state.audio_to_play)
        st.session_state.audio_to_play = None

    # 4. Get the next user input, only if not processing
    if not st.session_state.processing:
        if st.session_state.voice_enabled:
            user_prompt = speech_to_text(language=st.session_state.input_accent, use_container_width=True, just_once=True, key=f'STT_{len(st.session_state.chat_history)}')
        else:
            user_prompt = st.chat_input("Type your question here...")

        if user_prompt:
            st.session_state.chat_history.append({"user": user_prompt})
            st.rerun()

    # 5. Document uploader at the bottom
    st.divider()
    with st.expander("Upload a Custom Document"):
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document..."):
                session_id = str(uuid.uuid4())
                temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id); os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                doc = fitz.open(file_path)
                texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text("".join(page.get_text() for page in doc))
                doc.close()
                FAISS.from_texts(texts, embeddings).save_local(temp_dir)
                st.session_state.session_id = session_id
                st.session_state.active_doc_name = uploaded_file.name
                st.session_state.chat_history = [{"bot": f"Ready for questions about **{uploaded_file.name}**."}]
                st.rerun()

    if st.session_state.active_doc_name:
        st.info(f"Active Document: **{st.session_state['active_doc_name']}**")
        if st.button("Clear Document & Revert to Default"):
            st.session_state.session_id = None
            st.session_state.active_doc_name = None
            st.session_state.chat_history.append({"bot": "Reverted to default knowledge base."})
            st.rerun()
