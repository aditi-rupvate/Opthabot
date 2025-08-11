import os
import re
import uuid
import base64
import json
import streamlit as st
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

# NEW: JS executor (runs in top page context)
try:
    from streamlit_javascript import st_javascript
except Exception:
    st_javascript = None

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

# --- Text-to-Speech (server -> base64 MP3) (optional) ---
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
        st.toast(f"Server TTS issue (browser will speak instead): {e}", icon="⚠️")
        return None

# --- Browser TTS bridge (using streamlit-javascript) ---
def js_init_runtime():
    if not st_javascript:
        st.error("Missing dependency: streamlit-javascript. Add it to requirements.txt and redeploy.")
        return False
    # Prepare a small runtime in the page: unlock and speak helpers
    st_javascript("""
    (function(){
      window.TTS = window.TTS || {};
      TTS._unlocked = TTS._unlocked || false;

      TTS.unlock = async function(){
        // A user gesture must call this (via Streamlit button)
        try {
          // Warm up speechSynthesis with a short utterance
          const u = new SpeechSynthesisUtterance("Voice enabled");
          u.volume = 0; // silent blip to satisfy some browsers
          speechSynthesis.cancel();
          speechSynthesis.speak(u);
          TTS._unlocked = true;
          return true;
        } catch(e) {
          return false;
        }
      };

      TTS.speak = async function(text, voiceHint){
        try {
          if (!TTS._unlocked) return false;
          const ss = window.speechSynthesis;
          if (!ss) return false;
          function pickVoice(want){
            const voices = ss.getVoices ? ss.getVoices() : [];
            if (voices && voices.length){
              if (want){
                const hit = voices.find(v => v.lang && v.lang.toLowerCase().includes(want.toLowerCase()));
                if (hit) return hit;
              }
              const en = voices.find(v => v.lang && /^en[-_]/i.test(v.lang));
              return en || voices[0];
            }
            return null;
          }
          // Wait a tick for voices list if needed
          if (!ss.getVoices || ss.getVoices().length === 0){
            await new Promise(r => setTimeout(r, 400));
          }
          ss.cancel();
          const u = new SpeechSynthesisUtterance(text);
          const v = pickVoice(voiceHint);
          if (v) u.voice = v;
          u.rate = 1.0; u.pitch = 1.0; u.volume = 1.0;
          ss.speak(u);
          return true;
        } catch(e){
          return false;
        }
      };
    })();
    """)
    return True

def js_unlock():
    if not st_javascript: return False
    ok = st_javascript("await window.TTS?.unlock?.();")
    return bool(ok)

def js_speak(text: str, voice_hint: str):
    if not st_javascript: return False
    # Pass safe JSON strings
    payload = json.dumps({"text": text, "hint": voice_hint or ""})
    ok = st_javascript(f"""
      (async () => {{
        const p = {payload};
        return await window.TTS?.speak?.(p.text, p.hint);
      }})()
    """)
    return bool(ok)

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

# --- Main Query Logic ---
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

    def question_answer_func(query: str) -> str:
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return chain.invoke(query)['result']

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
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = False
if "input_accent" not in st.session_state: st.session_state.input_accent = 'en-GB'   # voice hint for browser TTS
if "output_accent" not in st.session_state: st.session_state.output_accent = 'co.uk' # gTTS TLD (optional)
if "session_started" not in st.session_state: st.session_state.session_started = False
if "voice_ready" not in st.session_state: st.session_state.voice_ready = False

THEME = DARK if st.session_state.theme == "dark" else LIGHT

# Init JS runtime once
if st_javascript:
    js_init_runtime()

# --- Initial interaction gate (first user tap enables audio) ---
st.markdown(f"""
<style>
    .stApp {{ background: {THEME['bg']}; color: {THEME['text']}; }}
</style>
""", unsafe_allow_html=True)

st.title("Ophthalmology AI Assistant")

col1, col2 = st.columns(2)
with col1:
    if st.toggle("Dark Mode", value=(st.session_state.theme == "dark")):
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
with col2:
    st.session_state.voice_enabled = st.toggle("Enable Voice Chat", value=st.session_state.voice_enabled)

# One-time voice unlock button (must be tapped/clicked once in this page)
if st.session_state.voice_enabled:
    if not st.session_state.voice_ready:
        if st.button("🔊 Enable Voice (one-time)"):
            ok = js_unlock()
            st.session_state.voice_ready = bool(ok)
            if ok:
                st.success("Voice enabled. I’ll speak answers automatically.")
            else:
                st.warning("Couldn’t enable voice automatically. Try again, or check browser autoplay settings.")
    else:
        st.info("Voice is enabled — I’ll speak every answer.")

# Chat history render
for entry in st.session_state.chat_history:
    if "user" in entry:
        st.markdown(f"<div style='background:{THEME['user']};padding:10px;border-radius:10px;margin:6px 0;text-align:right'>{entry['user']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:{THEME['bot']};padding:10px;border-radius:10px;margin:6px 0'>{entry['bot']}</div>", unsafe_allow_html=True)
        if entry.get("pdf_filename"):
            pdf_path = os.path.join(CHEATSHEET_PATH, entry["pdf_filename"])
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📥 Download Cheatsheet", pdf_file.read(), entry["pdf_filename"], "application/pdf", key=f"dl_{entry['pdf_filename']}_{uuid.uuid4()}")

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
            texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text("".join(page.get_text() for page in doc))
            doc.close()
            FAISS.from_texts(texts, embeddings).save_local(temp_dir)
            st.session_state.session_id = session_id; st.session_state.active_doc_name = uploaded_file.name
            st.session_state.chat_history.append({"bot": f"Ready for questions about **{uploaded_file.name}**.<br><span style='color:#787878'>{disclaimer_text}</span>"})
            st.rerun()

if st.session_state.active_doc_name:
    st.info(f"Active Document: **{st.session_state['active_doc_name']}**")
    if st.button("Clear Document & Revert to Default"):
        st.session_state.session_id = None; st.session_state.active_doc_name = None
        st.session_state.chat_history.append({"bot": f"Reverted to default knowledge base.<br><span style='color:#787878'>{disclaimer_text}</span>"})
        st.rerun()

# Input
if st.session_state.voice_enabled:
    user_prompt = speech_to_text(language=st.session_state.input_accent, use_container_width=True, just_once=True, key='STT')
else:
    user_prompt = st.chat_input("Type your question here...")

# On message
if user_prompt:
    st.markdown(f"<div style='background:{THEME['user']};padding:10px;border-radius:10px;margin:6px 0;text-align:right'>{user_prompt}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"user": user_prompt})

    with st.spinner("Thinking..."):
        answer, pdf_filename = handle_query_logic(user_prompt, st.session_state.get("session_id"))
        clean_text = re.sub(r'<.*?>', '', answer)
        raw_answer_text = clean_text.replace('`', '').replace('*', '')
        full_answer_html = f"{answer}<br><span style='color:#787878'>{disclaimer_text}</span>"
        st.markdown(f"<div style='background:{THEME['bot']};padding:10px;border-radius:10px;margin:6px 0'>{full_answer_html}</div>", unsafe_allow_html=True)

        # Voice: speak automatically after unlock
        if st.session_state.voice_enabled and st.session_state.voice_ready:
            # Browser speech — reliable after first tap
            spoke = js_speak(raw_answer_text + " Is there anything else I can help with?", st.session_state.input_accent)
            # Optional MP3 generation (downloadable). Not needed to speak.
            if not spoke:
                mp3_b64 = text_to_audio_b64(raw_answer_text, st.session_state.output_accent)
                if mp3_b64:
                    # show a small audio player as fallback
                    st.audio(base64.b64decode(mp3_b64), format="audio/mp3")

        if pdf_filename:
            pdf_path = os.path.join(CHEATSHEET_PATH, pdf_filename)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📥 Download Cheatsheet", pdf_file.read(), pdf_filename, "application/pdf", key=f"dl_{pdf_filename}_{uuid.uuid4()}")

        st.session_state.chat_history.append({"bot": full_answer_html, "pdf_filename": pdf_filename})
