import os
import re
import uuid
import time
import base64
import streamlit as st
from fpdf import FPDF
import fitz # PyMuPDF
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

# --- 1. Configuration ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_DEFAULT_API_KEY_HERE")
FAISS_INDEX_PATH = "oxford_handbook_kb"
TEMP_STORAGE_PATH = "temp_user_docs"
CHEATSHEET_PATH = "downloads"
os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
os.makedirs(CHEATSHEET_PATH, exist_ok=True)

# --- DISCLAIMER ---
disclaimer_text = "— Note: This output is for academic purposes only and must not be used for clinical diagnosis."

# === Ophthalmology gate (add-only) ===========================================
# Allow ophthalmology only (diagnoses, anatomy, tests, surgery, meds, devices, guidelines)
OPHTH_KEYWORDS = [
    # core
    "eye", "ocular", "ophthalmology", "ophthalmic", "vision", "visual acuity", "refraction",
    "cornea", "conjunctiva", "sclera", "anterior chamber", "iris", "pupil", "lens",
    "vitreous", "retina", "macula", "fovea", "optic nerve", "optic disc",
    # diseases / findings
    "cataract", "glaucoma", "amd", "age-related macular degeneration", "diabetic retinopathy",
    "dr", "csr", "central serous", "uveitis", "keratoconus", "dry eye", "meibomian",
    "blepharitis", "strabismus", "amblyopia", "endophthalmitis", "retinal detachment",
    "rhegmatogenous", "retinitis pigmentosa", "toxoplasmosis", "cmv retinitis",
    # tests / devices / surgery
    "slit lamp", "gonioscopy", "tonometry", "intraocular pressure", "iop",
    "oct", "optical coherence tomography", "perimetry", "visual field",
    "fundus", "ophthalmoscopy", "fluorescein angiography", "ultrasound b-scan",
    "lasik", "prk", "iol", "intraocular lens", "phaco", "vitrectomy", "trabeculectomy",
    # meds
    "latanoprost", "timolol", "brimonidine", "dorzolamide", "pilocarpine",
    "prednisolone", "moxifloxacin", "cyclopentolate", "tropicamide",
    # clinic symptoms
    "red eye", "floaters", "flashes", "metamorphopsia", "diplopia", "photophobia",
    "eye trauma", "chemical injury", "contact lens", "orthokeratology"
]

# Tiny small-talk whitelist (only pleasantries, not general chat)
GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|yo)\b",
    r"^\s*good (morning|afternoon|evening)\b",
    r"^\s*(how are you|how's it going|what's up)\b",
    r"^\s*(thank you|thanks)\b",
    r"^\s*(bye|goodbye|see you)\b"
]

def _is_greeting(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in GREETING_PATTERNS)

def _is_ophthalmology(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in OPHTH_KEYWORDS)

DOMAIN_REFUSAL = (
    "I'm specialised in ophthalmology (eye care) and basic greetings only. "
    "Please ask an eye-related question."
)
# ============================================================================ 

# --- Backend Components ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# ---------------- DYNAMIC FOLLOW-UP: helper chain ---------------------------
def generate_teaching_followup(user_q: str, explanation: str) -> str:
    """
    Produce ONE short, dynamic follow-up line:
      - either a comprehension check, an offer to simplify, or a bite-sized self-check question.
    """
    tmpl = PromptTemplate.from_template(
        """You are an ophthalmology tutor.
Choose ONE best follow-up style for the learner based on the question and explanation:
- Comprehension check question tailored to the content, OR
- Offer to simplify with an easier explanation/analogy, OR
- One short self-check question (MCQ or short answer); do NOT reveal answers.

Constraints:
- 25 words max
- No headers/emojis/preamble; just the line.

User question: {user_q}
Explanation: {explanation}

Follow-up:"""
    )
    chain = LLMChain(llm=llm, prompt=tmpl)
    out = chain.run(user_q=user_q, explanation=explanation).strip()
    # guardrails: keep it single line & short
    return re.sub(r"\s+", " ", out.split("\n")[0])[:300]

# --- Text-to-Speech: mobile-safe (returns base64 + renders HTML audio) ---
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
        st.warning(f"Could not generate audio response: {e}")
        return None

def render_audio_player_b64(audio_b64: str):
    # Mobile-safe autoplay with fallback on first user gesture (iOS/Android)
    audio_id = f"audio_{uuid.uuid4().hex}"
    audio_html = f"""
    <audio id="{audio_id}" autoplay playsinline preload="auto"
           style="width:0;height:0;visibility:hidden;" controlslist="nodownload noplaybackrate"
           src="data:audio/mpeg;base64,{audio_b64}">
      <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
    </audio>
    <script>
      (function() {{
        const a = document.getElementById("{audio_id}");
        if (!a) return;
        function tryPlay() {{
          const p = a.play();
          if (p && typeof p.then === "function") {{
            p.catch(() => {{
              // If autoplay is blocked, unlock on the next user gesture
              const unlock = () => {{
                a.play().catch(() => {{ /* ignore */ }});
                document.removeEventListener('touchstart', unlock, true);
                document.removeEventListener('click', unlock, true);
              }};
              document.addEventListener('touchstart', unlock, true);
              document.addEventListener('click', unlock, true);
            }});
          }}
        }}
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
          tryPlay();
        }} else {{
          document.addEventListener('DOMContentLoaded', tryPlay, {{ once: true }});
        }}
      }})();
    </script>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

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
        line = strip = line.strip()
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
    # --- Domain gate (ophthalmology-only + brief greetings) ---
    if _is_greeting(query):
        return "Hello! I’m your ophthalmology-only assistant. How can I help with eyes/vision today?", None
    if not _is_ophthalmology(query):
        return DOMAIN_REFUSAL, None
    # --- end gate ---

    # --- Choose KB (default or uploaded) ---
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

    # --- Chains for tools ---
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

    # System prompt toggles depth/style but dynamic follow-up is always on now
    teaching_mode = st.session_state.get("teaching_mode", False)
    if teaching_mode:
        system_instruction = (
            "You are an expert ophthalmology tutor. "
            "Rules: "
            "1) Answer strictly ophthalmology questions. "
            "2) Explain clearly in short steps, define jargon in-line, and use concise examples or analogies when helpful. "
            "3) Prefer bullet points and short paragraphs; keep focus and avoid fluff. "
            "4) Calibrate depth to a postgraduate student. "
            "5) At the end, do NOT repeat the answer—just include one dynamic follow-up line that either "
            "   (a) checks understanding, (b) offers to simplify, or (c) asks a tiny self-check question. "
            "6) Never leave the ophthalmology domain."
        )
    else:
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

    # ---------------- DYNAMIC FOLLOW-UP: now applied to ALL answers ----------
    if isinstance(final_answer, str) and final_answer.strip():
        try:
            follow = generate_teaching_followup(query, final_answer)
            if follow:
                final_answer = f"{final_answer}\n\n{follow}"
        except Exception:
            pass
    # ------------------------------------------------------------------------

    return final_answer, pdf_filename

# --- Streamlit UI ---
st.set_page_config(layout="centered")

LIGHT = {"bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd", "text": "#191b22", "input": "#e8edf2", "border": "#d4dde7", "expander": "#f4f7fb"}
DARK = {"bg": "#18181c", "bar": "#202126", "bot": "#232733", "user": "#22577a", "text": "#f3f5f8", "input": "#242730", "border": "#26282f", "expander": "#24272e"}

# Initialize session state variables
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "session_id" not in st.session_state: st.session_state.session_id = None
if "active_doc_name" not in st.session_state: st.session_state.active_doc_name = None
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = False
if "input_accent" not in st.session_state: st.session_state.input_accent = 'en-US'
if "output_accent" not in st.session_state: st.session_state.output_accent = 'com'
if "teaching_mode" not in st.session_state: st.session_state.teaching_mode = False

THEME = DARK if st.session_state.theme == "dark" else LIGHT

# --- Main Application UI (no first page) ---
with st.sidebar:
    st.header("Settings")
    is_dark_on = st.session_state.theme == "dark"
    toggled = st.toggle("Dark Mode", value=is_dark_on, key="theme_toggle", help="Switch themes")
    if toggled != is_dark_on:
        st.session_state.theme = "dark" if toggled else "light"
        st.rerun()

    st.divider()
    # Teaching Mode still controls style/depth; follow-up is always on
    st.session_state.teaching_mode = st.toggle(
        "Teaching Mode (tutor-style answers + clearer steps)",
        value=st.session_state.teaching_mode,
        help="When on, explanations are stepwise with definitions and examples. Dynamic follow-up is always enabled."
    )

    st.header("Voice Settings")
    st.session_state.voice_enabled = st.toggle("Enable Voice Chat", value=st.session_state.voice_enabled, help="Enable voice input and spoken responses.")

    if st.session_state.voice_enabled:
        input_accent_options = {
            'American (US)': 'en-US', 'British (UK)': 'en-GB', 'Indian': 'en-IN',
            'Australian': 'en-AU', 'Canadian': 'en-CA', 'South African': 'en-ZA'
        }
        current_accent_index = 0
        try:
            current_accent_index = list(input_accent_options.values()).index(st.session_state.input_accent)
        except ValueError:
            pass
        selected_input_label = st.selectbox("Your Accent (for input)", options=list(input_accent_options.keys()), index=current_accent_index)
        st.session_state.input_accent = input_accent_options[selected_input_label]
        
        output_accent_options = {'American (US)': 'com', 'British (UK)': 'co.uk', 'Indian': 'co.in'}
        st.session_state.output_accent = output_accent_options[st.selectbox(
            "Assistant's Accent (for output)",
            options=list(output_accent_options.keys()),
            index=list(output_accent_options.values()).index(st.session_state.output_accent)
        )]

st.markdown(f"""
<style>
    .stApp {{ background: {THEME['bg']}; color: {THEME['text']}; }}
    .topbar-custom {{ background: {THEME['bar']}; border-radius: 16px; padding: 1.3em 1.2em 1.15em 2.1em; margin-bottom: 1.6em; box-shadow: 0 2px 12px 0 rgba(44,46,66,0.06); font-size: 1.55rem; font-weight: 800; letter-spacing: .02em; }}
    .msg-user {{ background: {THEME['user']}; color: {THEME['text']}; border-radius: 16px 16px 4px 20px; margin-bottom: 0.3em; padding: 1em 1.35em; width: fit-content; max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-left: auto; margin-right: 0; text-align: right; box-shadow: 0 1px 12px 0 rgba(55,96,148,0.05); word-break: break-word; }}
    .msg-bot {{ background: {THEME['bot']}; color: {THEME['text']}; border-radius: 16px 16px 20px 4px; margin-bottom: 0.7em; padding: 1.08em 1.23em 1em 1.18em; width: fit-content; max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-right: auto; margin-left: 0; text-align: left; box-shadow: 0 1px 12px 0 rgba(44,46,66,0.05); word-break: break-word; }}
    [data-testid="stExpander"] {{ border-color: {THEME['border']}; background: {THEME['expander']}; }}
    .stButton>button, .stDownloadButton>button {{ border: 1px solid {THEME['border']}; }}
    .note-text {{ color: #787878; font-size: 0.9rem; }}
    @media only screen and (max-width: 768px) {{ .topbar-custom {{ font-size: 1.2rem; padding: 1em; text-align: center; }} .msg-user, .msg-bot {{ font-size: 0.95rem; max-width: 95%; }} }}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='topbar-custom'>Ophtha Bot : AI Chatbot for Postgrad Ophthalmology Students</div>", unsafe_allow_html=True)

for entry in st.session_state.chat_history:
    if "user" in entry:
        st.markdown(f"<div class='msg-user'>{entry['user']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='msg-bot'>{entry['bot']}</div>", unsafe_allow_html=True)
        if entry.get("pdf_filename"):
            pdf_path = os.path.join(CHEATSHEET_PATH, entry["pdf_filename"])
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📥 Download Cheatsheet", pdf_file.read(), entry["pdf_filename"], "application/pdf", key=f"dl_{entry['pdf_filename']}_{uuid.uuid4()}")

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
            st.session_state.chat_history.append({"bot": f"Ready for questions about **{uploaded_file.name}**.<br><span class='note-text'>{disclaimer_text}</span>"})
            st.rerun()

if st.session_state.active_doc_name:
    st.info(f"Active Document: **{st.session_state['active_doc_name']}**")
    if st.button("Clear Document & Revert to Default"):
        st.session_state.session_id = None; st.session_state.active_doc_name = None
        st.session_state.chat_history.append({"bot": f"Reverted to default knowledge base.<br><span class='note-text'>{disclaimer_text}</span>"})
        st.rerun()

user_prompt = None
if st.session_state.voice_enabled:
    user_prompt = speech_to_text(language=st.session_state.input_accent, use_container_width=True, just_once=True, key='STT')
else:
    user_prompt = st.chat_input("Type your question here...")

if user_prompt:
    st.markdown(f"<div class='msg-user'>{user_prompt}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"user": user_prompt})

    with st.spinner("Thinking..."):
        answer, pdf_filename = handle_query_logic(user_prompt, st.session_state.get("session_id"))
        clean_text = re.sub(r'<.*?>', '', answer); raw_answer_text = clean_text.replace('`', '').replace('*', '')
        full_answer_html = f"{answer}<br><span class='note-text'>{disclaimer_text}</span>"
        
        st.markdown(f"<div class='msg-bot'>{full_answer_html}</div>", unsafe_allow_html=True)

        if st.session_state.voice_enabled:
            spoken_text = raw_answer_text + " Is there anything else I can help with?"
            audio_b64 = text_to_audio_b64(spoken_text, st.session_state.output_accent)
            if audio_b64:
                render_audio_player_b64(audio_b64)

        if pdf_filename:
            pdf_path = os.path.join(CHEATSHEET_PATH, pdf_filename)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📥 Download Cheatsheet", pdf_file.read(), pdf_filename, "application/pdf", key=f"dl_{pdf_filename}_{uuid.uuid4()}")

        st.session_state.chat_history.append({"bot": full_answer_html, "pdf_filename": pdf_filename})
