import os
import re
import uuid
import base64
import json
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
import fitz  # PyMuPDF
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.tools import StructuredTool
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from langchain.memory import ConversationBufferMemory

# --- 1) Configuration ---
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

# --- Optional server TTS (downloadable MP3; not required for speaking) ---
def text_to_audio_b64(text: str, tld: str) -> str | None:
    try:
        tts = gTTS(text=text, lang='en', tld=tld, slow=False)
        audio_filename = os.path.join(CHEATSHEET_PATH, f"response_{uuid.uuid4()}.mp3")
        tts.save(audio_filename)
        with open(audio_filename, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        try:
            os.remove(audio_filename)
        except OSError:
            pass
        return b64
    except Exception as e:
        st.toast(f"Server TTS issue (browser voice will speak): {e}", icon="⚠️")
        return None

# --- Client voice components (no extra deps) ---

def render_voice_unlock_button():
    """
    Real HTML button; its click handler runs entirely in the browser so it counts as a user gesture.
    It warms up AudioContext (optional) + speechSynthesis, then sets localStorage.voice_ready='1'.
    """
    components.html(
        """
        <button id="voiceUnlockBtn"
                style="padding:10px 14px;border-radius:10px;border:1px solid #444;
                       background:#2b2f3a;color:#fff;font-weight:700;cursor:pointer;">
            🔊 Enable Voice (one-time)
        </button>
        <script>
        (function(){
          const btn = document.getElementById('voiceUnlockBtn');
          if (!btn) return;

          // If already enabled, update UI on load
          if (localStorage.getItem('voice_ready') === '1') {
            btn.textContent = "✅ Voice Enabled";
            btn.disabled = true;
            btn.style.opacity = "0.7";
          }

          const AC = window.AudioContext || window.webkitAudioContext;

          async function unlockAudio(){
            try{
              // 1) WebAudio warmup (future-proof)
              if (AC){
                window._voiceCtx = window._voiceCtx || new AC();
                if (window._voiceCtx.state !== 'running'){
                  try { await window._voiceCtx.resume(); } catch (e) {}
                }
                try {
                  const ctx = window._voiceCtx;
                  const buf = ctx.createBuffer(1, 22050, 44100);
                  const src = ctx.createBufferSource();
                  src.buffer = buf; src.connect(ctx.destination); src.start(0);
                } catch (e) {}
              }

              // 2) SpeechSynthesis warmup with a silent utterance
              const ss = window.speechSynthesis;
              if (ss) {
                const u = new SpeechSynthesisUtterance("ok");
                u.volume = 0; // silent
                ss.cancel();
                ss.speak(u);
              }

              // 3) Persist
              localStorage.setItem('voice_ready', '1');

              // 4) Update UI
              btn.textContent = "✅ Voice Enabled";
              btn.disabled = true;
              btn.style.opacity = "0.7";
            }catch(e){
              console.warn("Unlock failed", e);
            }
          }

          btn.addEventListener('click', unlockAudio, { passive: true });
        })();
        </script>
        """,
        height=60,
    )

def speak_via_browser(text: str, voice_hint: str):
    """
    Hidden component that auto-speaks via speechSynthesis if localStorage.voice_ready == '1'.
    Uses token replacement (no f-strings), so JS braces don’t break Python.
    """
    safe_text = json.dumps(text)
    safe_hint = json.dumps(voice_hint or "")
    html = """
    <div style="display:none"></div>
    <script>
    (function(){
      if (localStorage.getItem('voice_ready') !== '1') return;

      const text = __TEXT__;
      const hint = __HINT__;
      const ss = window.speechSynthesis;
      if (!ss) return;

      function pickVoice(h) {
        const vs = ss.getVoices ? ss.getVoices() : [];
        if (vs && vs.length) {
          if (h) {
            const hit = vs.find(v => (v.lang||"").toLowerCase().includes(h.toLowerCase()));
            if (hit) return hit;
          }
          const en = vs.find(v => /^en[-_]/i.test(v.lang||""));
          return en || vs[0];
        }
        return null;
      }

      function speakNow(){
        try {
          ss.cancel();
          const u = new SpeechSynthesisUtterance(text);
          const v = pickVoice(hint);
          if (v) u.voice = v;
          u.rate = 1.0; u.pitch = 1.0; u.volume = 1.0;
          ss.speak(u);
        } catch(e) {}
      }

      if (!ss.getVoices || ss.getVoices().length === 0) {
        // Safari/iOS sometimes populates voices asynchronously
        let spoke = false;
        const once = () => { if (!spoke) { speakNow(); spoke = true; } };
        ss.addEventListener('voiceschanged', once, { once: true });
        setTimeout(once, 600);
      } else {
        speakNow();
      }
    })();
    </script>
    """
    html = html.replace("__TEXT__", safe_text).replace("__HINT__", safe_hint)
    components.html(html, height=0)

# --- PDF Generation Class ---
class PDF(FPDF):
    def __init__(self, topic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topic = topic
    def header(self):
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Ophthalmology Cheatsheet: {self.topic.title()}", 0, 0, 'L')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, 'C')

# --- PDF Function ---
def create_formatted_pdf(text_content: str, topic: str) -> str:
    pdf = PDF(topic)
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
    except RuntimeError:
        st.error("Could not find 'DejaVuSans.ttf' or 'DejaVuSans-Bold.ttf'.")
        return ""
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 10, f"Cheatsheet: {topic.title()}", 0, 'C')
    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y())
    pdf.ln(10)
    line_height = 7
    pdf.set_text_color(50, 50, 50)
    for line in text_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('## '):
            pdf.set_font("DejaVu", "B", 14)
            pdf.set_text_color(0, 80, 150)
            pdf.multi_cell(0, line_height, line.replace('## ', ''), 0, 'L')
            pdf.set_text_color(50, 50, 50)
            pdf.ln(2)
        elif line.startswith('- '):
            pdf.set_font("DejaVu", "", 11)
            pdf.set_x(20)
            pdf.multi_cell(0, line_height, f"• {line.replace('- ', '', 1)}")
            pdf.ln(1)
        else:
            pdf.set_font("DejaVu", "", 11)
            pdf.multi_cell(0, line_height, line)
    pdf.ln(5)
    pdf.set_draw_color(200, 200, 200)
    x = pdf.get_x()
    pdf.line(x, pdf.get_y(), x + 180, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, "Note: This content is for academic purposes only and must not be used for clinical diagnosis.")
    clean_topic = re.sub(r'[\W_]+', '_', topic).lower()
    filename = f"{clean_topic}_cheatsheet.pdf"
    filepath = os.path.join(CHEATSHEET_PATH, filename)
    pdf.output(filepath)
    return filename

# --- Main Query Logic (RAG/Agent) ---
def handle_query_logic(query: str, session_id: str = None):
    if session_id:
        temp_db_path = os.path.join(TEMP_STORAGE_PATH, session_id)
        if not os.path.exists(temp_db_path):
            return "Error: Your document session has expired.", None
        db = FAISS.load_local(temp_db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(FAISS_INDEX_PATH):
            return "Error: Default knowledge base not available.", None
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def question_answer_func(q: str) -> str:
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return chain.invoke(q)['result']

    def concept_explainer_func(topic: str) -> str:
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
        prompt_template = PromptTemplate.from_template(
            "Provide a comprehensive explanation or summary for {topic}.\n\nContext: {context}\nResponse:"
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        return chain.run(topic=topic, context=context)

    def cheatsheet_generator_func(topic: str) -> str:
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
        prompt_template = PromptTemplate.from_template(
            "Create a detailed cheat sheet for {topic} using '##' for headings and '-' for list items.\nContext: {context}\nCheat Sheet:"
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        cheatsheet_text = chain.run(topic=topic, context=context)
        pdf_filename = create_formatted_pdf(cheatsheet_text, topic)
        return f"PDF_GENERATED::{pdf_filename}::{cheatsheet_text}"

    tools = [
        StructuredTool.from_function(func=question_answer_func, name="QuestionAnswerTool", description="Use for direct, specific questions."),
        StructuredTool.from_function(func=concept_explainer_func, name="ConceptExplainerTool", description="Use for summaries or explanations in the chat."),
        StructuredTool.from_function(func=cheatsheet_generator_func, name="CheatsheetGeneratorTool", description="Use ONLY when explicitly asked for a downloadable PDF or 'cheat sheet'.")
    ]

    base_prompt = hub.pull("hwchase17/react-chat")
    system_instruction = (
        "You are an expert ophthalmology assistant. Your purpose is to answer questions strictly related to "
        "ophthalmology or the provided documents. If the user asks a question that is outside of this scope, you must "
        "politely decline and state that you can only answer questions about ophthalmology."
    )
    prompt = base_prompt.partial(system_message=system_instruction)
    agent = create_react_agent(llm, tools, prompt)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for msg in st.session_state.chat_history:
        if "user" in msg:
            memory.chat_memory.add_user_message(msg["user"])
        else:
            clean_bot_message = re.sub(r'<.*?>', '', msg["bot"])
            memory.chat_memory.add_ai_message(clean_bot_message)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    response = agent_executor.invoke({"input": query})
    final_answer = response.get('output', "I couldn't find an answer.")
    pdf_filename = None
    if 'intermediate_steps' in response:
        for _, observation in response['intermediate_steps']:
            if isinstance(observation, str) and observation.startswith("PDF_GENERATED::"):
                try:
                    pdf_filename = observation.split("::")[1]
                except IndexError:
                    pass
    return final_answer, pdf_filename

# --- Streamlit UI ---
st.set_page_config(layout="centered")

LIGHT = {"bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd", "text": "#191b22", "input": "#e8edf2", "border": "#d4dde7", "expander": "#f4f7fb"}
DARK  = {"bg": "#18181c", "bar": "#202126", "bot": "#232733", "user": "#22577a", "text": "#f3f5f8", "input": "#242730", "border": "#26282f", "expander": "#24272e"}

# State
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "session_id" not in st.session_state: st.session_state.session_id = None
if "active_doc_name" not in st.session_state: st.session_state.active_doc_name = None
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = True
if "input_accent" not in st.session_state: st.session_state.input_accent = 'en-GB'   # browser voice hint
if "output_accent" not in st.session_state: st.session_state.output_accent = 'co.uk' # gTTS TLD (optional)
THEME = DARK if st.session_state.theme == "dark" else LIGHT

# Topbar + styles
st.markdown(f"""
<style>
  .stApp {{ background: {THEME['bg']}; color: {THEME['text']}; }}
  .topbar {{ background: {THEME['bar']}; border-radius: 16px; padding: 1.0em 1.2em; 
             margin-bottom: 1.0em; box-shadow: 0 2px 12px rgba(44,46,66,0.06); }}
  .title   {{ font-size: 2.2rem; font-weight: 800; letter-spacing: .02em; }}
</style>
<div class="topbar"><span class="title">Ophthalmology AI Assistant</span></div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    is_dark = st.toggle("Dark Mode", value=(st.session_state.theme == "dark"))
    st.session_state.theme = "dark" if is_dark else "light"
with c2:
    st.session_state.voice_enabled = st.toggle("Enable Voice Chat", value=st.session_state.voice_enabled)

# Voice unlock + accent selectors
if st.session_state.voice_enabled:
    st.warning("Tap the button once to enable voice.", icon="🗣️")
    render_voice_unlock_button()

    input_accent_options = {
        'American (US)': 'en-US', 'British (UK)': 'en-GB', 'Indian': 'en-IN',
        'Australian': 'en-AU', 'Canadian': 'en-CA', 'South African': 'en-ZA'
    }
    output_accent_options = {'American (US)': 'com', 'British (UK)': 'co.uk', 'Indian': 'co.in'}

    selected_input_label = st.selectbox(
        "Your Accent (for browser voice)", list(input_accent_options.keys()),
        index=list(input_accent_options.values()).index(st.session_state.input_accent)
    )
    st.session_state.input_accent = input_accent_options[selected_input_label]

    selected_output_label = st.selectbox(
        "Assistant's Accent (for MP3 download)", list(output_accent_options.keys()),
        index=list(output_accent_options.values()).index(st.session_state.output_accent)
    )
    st.session_state.output_accent = output_accent_options[selected_output_label]

# Chat history
for entry in st.session_state.chat_history:
    if "user" in entry:
        st.markdown(
            f"<div style='background:{THEME['user']};color:{THEME['text']};border:1px solid {THEME['border']};"
            f"border-radius:14px;padding:10px 14px;margin:6px 0;text-align:right'>{entry['user']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background:{THEME['bot']};color:{THEME['text']};border:1px solid {THEME['border']};"
            f"border-radius:14px;padding:10px 14px;margin:6px 0'>{entry['bot']}</div>",
            unsafe_allow_html=True
        )
        if entry.get("pdf_filename"):
            pdf_path = os.path.join(CHEATSHEET_PATH, entry["pdf_filename"])
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📥 Download Cheatsheet", pdf_file.read(), entry["pdf_filename"],
                                       "application/pdf", key=f"dl_{entry['pdf_filename']}_{uuid.uuid4()}")

# Upload expander
with st.expander("Upload a Custom Document"):
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            session_id = str(uuid.uuid4())
            temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id); os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as buffer: buffer.write(uploaded_file.getbuffer())
            doc = fitz.open(file_path)
            texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(
                "".join(page.get_text() for page in doc)
            )
            doc.close()
            FAISS.from_texts(texts, embeddings).save_local(temp_dir)
            st.session_state.session_id = session_id
            st.session_state.active_doc_name = uploaded_file.name
            st.session_state.chat_history.append({"bot": f"Ready for questions about **{uploaded_file.name}**.<br><span style='color:#787878'>{disclaimer_text}</span>"})
            st.rerun()

if st.session_state.active_doc_name:
    st.info(f"Active Document: **{st.session_state['active_doc_name']}**")
    if st.button("Clear Document & Revert to Default"):
        st.session_state.session_id = None
        st.session_state.active_doc_name = None
        st.session_state.chat_history.append({"bot": f"Reverted to default knowledge base.<br><span style='color:#787878'>{disclaimer_text}</span>"})
        st.rerun()

# Input
if st.session_state.voice_enabled:
    user_prompt = speech_to_text(language=st.session_state.input_accent, use_container_width=True, just_once=True, key='STT')
else:
    user_prompt = st.chat_input("Type your question here...")

# On message
if user_prompt:
    st.markdown(
        f"<div style='background:{THEME['user']};color:{THEME['text']};border:1px solid {THEME['border']};"
        f"border-radius:14px;padding:10px 14px;margin:6px 0;text-align:right'>{user_prompt}</div>",
        unsafe_allow_html=True
    )
    st.session_state.chat_history.append({"user": user_prompt})

    with st.spinner("Thinking..."):
        answer, pdf_filename = handle_query_logic(user_prompt, st.session_state.get("session_id"))
        clean_text = re.sub(r'<.*?>', '', answer)
        raw_answer_text = clean_text.replace('`', '').replace('*', '')
        full_answer_html = f"{answer}<br><span style='color:#787878'>{disclaimer_text}</span>"

        st.markdown(
            f"<div style='background:{THEME['bot']};color:{THEME['text']};border:1px solid {THEME['border']};"
            f"border-radius:14px;padding:10px 14px;margin:6px 0'>{full_answer_html}</div>",
            unsafe_allow_html=True
        )

        # Speak automatically via browser voice (after one-time unlock)
        if st.session_state.voice_enabled:
            speak_via_browser(raw_answer_text + " Is there anything else I can help with?", st.session_state.input_accent)

        # Optional: offer MP3 download when a cheatsheet was made
        if pdf_filename:
            pdf_path = os.path.join(CHEATSHEET_PATH, pdf_filename)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📥 Download Cheatsheet", pdf_file.read(), pdf_filename,
                                       "application/pdf", key=f"dl_{pdf_filename}_{uuid.uuid4()}")

        st.session_state.chat_history.append({"bot": full_answer_html, "pdf_filename": pdf_filename})
