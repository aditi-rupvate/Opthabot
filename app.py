import os
import re
import uuid
import time
import base64
import json
import io
import csv
from datetime import datetime
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
OPHTH_KEYWORDS = [
    "eye", "ocular", "ophthalmology", "ophthalmic", "vision", "visual acuity", "refraction",
    "cornea", "conjunctiva", "sclera", "anterior chamber", "iris", "pupil", "lens",
    "vitreous", "retina", "macula", "fovea", "optic nerve", "optic disc",
    "cataract", "glaucoma", "amd", "age-related macular degeneration", "diabetic retinopathy",
    "dr", "csr", "central serous", "uveitis", "keratoconus", "dry eye", "meibomian",
    "blepharitis", "strabismus", "amblyopia", "endophthalmitis", "retinal detachment",
    "rhegmatogenous", "retinitis pigmentosa", "toxoplasmosis", "cmv retinitis",
    "slit lamp", "gonioscopy", "tonometry", "intraocular pressure", "iop",
    "oct", "optical coherence tomography", "perimetry", "visual field",
    "fundus", "ophthalmoscopy", "fluorescein angiography", "ultrasound b-scan",
    "lasik", "prk", "iol", "intraocular lens", "phaco", "vitrectomy", "trabeculectomy",
    "latanoprost", "timolol", "brimonidine", "dorzolamide", "pilocarpine",
    "prednisolone", "moxifloxacin", "cyclopentolate", "tropicamide",
    "red eye", "floaters", "flashes", "metamorphopsia", "diplopia", "photophobia",
    "eye trauma", "chemical injury", "contact lens", "orthokeratology"
]
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

# ---------------- DYNAMIC FOLLOW-UP (warmer, human tone) --------------------
def generate_teaching_followup(user_q: str, explanation: str) -> str:
    """
    ONE short, emotionally intelligent follow-up line:
      - Gentle check, offer to simplify, or tiny self-check (MCQ/short answer).
      - Supportive, conversational tone with contractions.
      - Max 20 words. No emojis/preamble. Don’t reveal answers.
    """
    tmpl = PromptTemplate.from_template(
        """You are an empathetic ophthalmology tutor with a warm, human style.
Write ONE short, emotionally intelligent follow-up line that best fits the learner and the content.

Choose exactly ONE:
- A gentle comprehension check, OR
- An offer to simplify with an analogy/example, OR
- A tiny self-check (MCQ/short answer, no answer revealed).

Tone & style:
- Supportive, conversational, encouraging. Use contractions.
- Avoid robotic phrasing like "Did you understand?" or "Do you want me to explain it easier?".
- Vary natural wording; sound like a caring clinician-teacher, not a script.
- Max 20 words. No preamble. No emojis.

Context
User question: {user_q}
Your explanation: {explanation}

Return only the single follow-up line:"""
    )
    chain = LLMChain(llm=llm, prompt=tmpl)
    out = chain.run(user_q=user_q, explanation=explanation).strip()
    line = re.sub(r"`+", "", out).split("\n")[0]
    line = re.sub(r"\s+", " ", line).strip()
    return line[:200]
# ---------------------------------------------------------------------------

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
              const unlock = () => {{
                a.play().catch(() => {{ }});
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

# --- RAG/Chat Logic ----------------------------------------------------------
def handle_query_logic(query: str, session_id: str = None):
    # Domain gate
    if _is_greeting(query):
        return "Hello! I’m your ophthalmology-only assistant. How can I help with eyes/vision today?", None
    if not _is_ophthalmology(query):
        return DOMAIN_REFUSAL, None

    # Choose KB
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

    # Chains
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

    teaching_mode = st.session_state.get("teaching_mode", False)
    if teaching_mode:
        system_instruction = (
            "You are an expert ophthalmology tutor. Rules: "
            "1) Answer strictly ophthalmology questions. "
            "2) Explain clearly in short steps, define jargon in-line, and use concise examples or analogies when helpful. "
            "3) Prefer bullet points and short paragraphs; keep focus and avoid fluff. "
            "4) Calibrate depth to a postgraduate student. "
            "5) At the end, do NOT repeat the answer—just include one dynamic follow-up line. "
            "6) Never leave the ophthalmology domain."
        )
    else:
        system_instruction = (
            "You are an expert ophthalmology assistant. Answer questions strictly related to ophthalmology or the provided documents. "
            "If outside this scope, politely decline and state that you can only answer ophthalmology."
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
        agent=agent, tools=tools, memory=memory,
        verbose=False, handle_parsing_errors=True, return_intermediate_steps=True
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

    # Dynamic follow-up on all ophthal answers
    if isinstance(final_answer, str) and final_answer.strip():
        try:
            follow = generate_teaching_followup(query, final_answer)
            if follow:
                final_answer = f"{final_answer}\n\n{follow}"
        except Exception:
            pass

    return final_answer, pdf_filename

# ============================= NEW: EXAM MODE ================================

def _parse_json_block(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def generate_mcqs(topic: str, num_q: int = 5):
    """Return list of dicts: {question, options:[...], correct_index, explanation}"""
    prompt = PromptTemplate.from_template(
        """You are an ophthalmology exam item writer.
Create {num} high-quality single-best-answer MCQs on: "{topic}"
Constraints:
- Postgraduate level, strictly ophthalmology.
- 4 options (A–D). Exactly one correct.
- Provide a 1–2 line explanation.
- Output ONLY JSON as:
{{
  "mcqs": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "..."
    }}
  ]
}}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run(topic=topic, num=num_q)
    data = _parse_json_block(raw) or {"mcqs": []}
    mcqs = data.get("mcqs", [])
    clean = []
    for q in mcqs[:num_q]:
        if isinstance(q, dict) and all(k in q for k in ("question","options","correct_index","explanation")):
            if isinstance(q["options"], list) and len(q["options"]) == 4:
                try:
                    ci = int(q["correct_index"])
                except Exception:
                    ci = 0
                ci = max(0, min(3, ci))
                clean.append({
                    "question": q["question"].strip(),
                    "options": [str(x).strip() for x in q["options"]],
                    "correct_index": ci,
                    "explanation": q["explanation"].strip()
                })
    return clean

def render_exam_dashboard(exam_state):
    total = len(exam_state["questions"])
    attempted = len(exam_state["selected"])
    score = sum(1 for i, sel in exam_state["selected"].items() if sel == exam_state["questions"][i]["correct_index"])
    st.session_state.exam["score"] = score
    c1, c2, c3 = st.columns(3)
    c1.metric("Total MCQs", total)
    c2.metric("Attempted", attempted)
    c3.metric("Score", score)

def render_option_badges(q_idx, opts, correct_idx, selected_idx):
    for j, opt in enumerate(opts):
        cls = "neutral"
        if selected_idx is not None:
            if j == correct_idx:
                cls = "correct"
            if selected_idx == j and j != correct_idx:
                cls = "wrong"
        st.markdown(
            f"<div class='option {cls}'><b>{chr(65+j)}.</b> {opt}</div>",
            unsafe_allow_html=True
        )

def render_exam_ui():
    st.markdown("<div class='topbar-custom'>Exam Mode · MCQ Practice</div>", unsafe_allow_html=True)

    topic = st.text_input("Topic for MCQs (ophthalmology only)", placeholder="e.g., Primary open-angle glaucoma")
    cols = st.columns([1,1,2])
    with cols[0]:
        num_q = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5, step=1)
    with cols[1]:
        if st.button("Generate MCQs", use_container_width=True, type="primary"):
            mcqs = generate_mcqs(topic or "general ophthalmology", int(num_q))
            st.session_state.exam = {
                "topic": topic or "general ophthalmology",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "questions": mcqs,
                "selected": {},   # q_idx -> opt_idx
                "score": 0
            }
            st.rerun()

    exam_state = st.session_state.get("exam", None)
    if not exam_state or not exam_state.get("questions"):
        st.info("Enter a topic and click **Generate MCQs** to start.")
        return

    # Dashboard
    render_exam_dashboard(exam_state)
    st.markdown("<br/>", unsafe_allow_html=True)

    # Render each MCQ card (buttons before selection, badges after)
    for i, q in enumerate(exam_state["questions"]):
        st.markdown(f"<div class='card'><div class='card-title'>Q{i+1}. {q['question']}</div>", unsafe_allow_html=True)
        selected = exam_state["selected"].get(i, None)

        if selected is None:
            # BEFORE selection: show clickable buttons only
            bcols = st.columns(2)
            for j, opt in enumerate(q["options"]):
                if bcols[j % 2].button(f"{chr(65+j)}. {opt}", key=f"mcq_{i}_{j}"):
                    st.session_state.exam["selected"][i] = j
                    st.rerun()
        else:
            # AFTER selection: show colored badges + feedback (no buttons)
            render_option_badges(i, q["options"], q["correct_index"], selected)
            if selected == q["correct_index"]:
                st.markdown("<div class='explain good'>Correct ✅</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='explain bad'>Not quite ❌</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='explain note'><b>Why:</b> {q['explanation']}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # end card
        st.markdown("<br/>", unsafe_allow_html=True)

    # Refresh dashboard after answers
    render_exam_dashboard(st.session_state.exam)

    # Downloads
    rows = []
    for i, q in enumerate(st.session_state.exam["questions"]):
        sel = st.session_state.exam["selected"].get(i, None)
        rows.append({
            "index": i+1,
            "question": q["question"],
            "A": q["options"][0],
            "B": q["options"][1],
            "C": q["options"][2],
            "D": q["options"][3],
            "selected": "" if sel is None else chr(65+sel),
            "correct": chr(65+q["correct_index"]),
            "is_correct": sel == q["correct_index"] if sel is not None else None,
            "explanation": q["explanation"]
        })

    # CSV
    if rows:
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "📥 Download MCQ Session (CSV)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"exam_mcqs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # JSON
    json_payload = {
        "topic": st.session_state.exam.get("topic"),
        "generated_at": st.session_state.exam.get("generated_at"),
        "score": st.session_state.exam.get("score"),
        "attempted": len(st.session_state.exam.get("selected", {})),
        "total": len(st.session_state.exam.get("questions", [])),
        "items": rows
    }
    st.download_button(
        "📥 Download MCQ Session (JSON)",
        data=json.dumps(json_payload, indent=2).encode("utf-8"),
        file_name=f"exam_mcqs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

# ============================ NEW: CASE MODE =================================

def generate_case(topic: str):
    """Return dict with a scenario and evaluation keys."""
    prompt = PromptTemplate.from_template(
        """You are an ophthalmology simulation author.
Create ONE realistic case vignette (brief) for postgraduate level on: "{topic}"
Include:
- title
- scenario (2–5 sentences)
- key_points: list of 5–8 bullet keywords (diagnosis+workup+management targets)
Output ONLY JSON as:
{{
  "title": "...",
  "scenario": "...",
  "key_points": ["...", "...", "..."]
}}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run(topic=topic or "general ophthalmology")
    data = _parse_json_block(raw) or {}
    title = data.get("title", "Ophthalmology Case")
    scenario = data.get("scenario", "A patient presents to clinic...")
    key_points = data.get("key_points", [])
    return {"title": title, "scenario": scenario, "key_points": key_points}

def evaluate_case_response(scenario: str, key_points, user_answer: str):
    """Generate feedback + score using key_points as rubric."""
    rubric = "; ".join(key_points[:8])
    prompt = PromptTemplate.from_template(
        """You are grading a short free-text response for an ophthalmology case.
Case: {scenario}
Rubric key points (target ideas): {rubric}
Learner response: {answer}

Return ONLY JSON as:
{{
  "feedback": {{
    "strengths": ["...", "..."],
    "missed": ["...", "..."],
    "suggestions": "one concise paragraph with practical advice"
  }},
  "score": {{
    "achieved": <int 0-100>,
    "explanation": "one line on how the score was decided"
  }}
}}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run(scenario=scenario, rubric=rubric, answer=user_answer)
    data = _parse_json_block(raw) or {}
    fb = data.get("feedback", {})
    sc = data.get("score", {"achieved": 0, "explanation": ""})
    return fb, sc

def render_case_ui():
    st.markdown("<div class='topbar-custom'>Case-Based Mode · Simulation</div>", unsafe_allow_html=True)

    topic = st.text_input("Case focus (ophthalmology only)", placeholder="e.g., Painless vision loss · CRAO vs. NAION")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Generate Case", type="primary", use_container_width=True):
            c = generate_case(topic or "general ophthalmology")
            st.session_state.case = {
                "topic": topic or "general ophthalmology",
                "case": c,
                "response": "",
                "graded": None,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
            st.rerun()

    case_state = st.session_state.get("case", None)
    if not case_state:
        st.info("Enter a focus and click **Generate Case** to start.")
        return

    c = case_state["case"]
    st.markdown(
        f"""
        <div class='case-card'>
            <div class='case-title'>{c['title']}</div>
            <div class='case-body'>{c['scenario']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='case-instr'>Write your impression and next steps (investigations/initial management).</div>", unsafe_allow_html=True)
    st.session_state.case["response"] = st.text_area("Your response", value=case_state.get("response",""), height=160, placeholder="Type your reasoning here…")

    if st.button("Submit Answer", type="primary", use_container_width=True):
        fb, sc = evaluate_case_response(c["scenario"], c.get("key_points", []), st.session_state.case["response"])
        st.session_state.case["graded"] = {"feedback": fb, "score": sc}
        st.rerun()

    graded = case_state.get("graded")
    if graded:
        strengths = graded["feedback"].get("strengths", [])
        missed = graded["feedback"].get("missed", [])
        suggestions = graded["feedback"].get("suggestions","")
        score_val = graded["score"].get("achieved", 0)
        score_exp = graded["score"].get("explanation","")

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Score", f"{score_val}/100")
        cc2.metric("Key hits", len(strengths))
        cc3.metric("Gaps", len(missed))

        st.markdown("<div class='card'><div class='card-title'>Feedback</div>", unsafe_allow_html=True)
        if strengths:
            st.markdown("<b>What you did well</b>", unsafe_allow_html=True)
            for s_ in strengths: st.markdown(f"- {s_}")
        if missed:
            st.markdown("<b>What to add next time</b>", unsafe_allow_html=True)
            for m_ in missed: st.markdown(f"- {m_}")
        if suggestions:
            st.markdown(f"<div class='explain note'>{suggestions}</div>", unsafe_allow_html=True)
        if score_exp:
            st.caption(f"Scoring note: {score_exp}")
        st.markdown("</div>", unsafe_allow_html=True)

        payload = {
            "topic": case_state.get("topic"),
            "generated_at": case_state.get("generated_at"),
            "title": c["title"],
            "scenario": c["scenario"],
            "learner_response": st.session_state.case["response"],
            "feedback": graded["feedback"],
            "score": graded["score"]
        }
        st.download_button(
            "📥 Download Case Interaction (JSON)",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name=f"case_interaction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# --- Streamlit UI ---
st.set_page_config(layout="centered")

LIGHT = {"bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd", "text": "#191b22", "input": "#e8edf2", "border": "#d4dde7", "expander": "#f4f7fb"}
DARK = {"bg": "#202126", "bar": "#232733", "bot": "#232733", "user": "#22577a", "text": "#f3f5f8", "input": "#242730", "border": "#26282f", "expander": "#24272e"}

# Initialize session state variables
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "session_id" not in st.session_state: st.session_state.session_id = None
if "active_doc_name" not in st.session_state: st.session_state.active_doc_name = None
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = False
if "input_accent" not in st.session_state: st.session_state.input_accent = 'en-US'
if "output_accent" not in st.session_state: st.session_state.output_accent = 'com'
if "teaching_mode" not in st.session_state: st.session_state.teaching_mode = False
if "mode" not in st.session_state: st.session_state.mode = "Chat"  # Chat | Teaching | Exam | Case

THEME = DARK if st.session_state.theme == "dark" else LIGHT

# --- Global Styles (cards, options, case visuals) ---
st.markdown(f"""
<style>
    .stApp {{ background: {THEME['bg']}; color: {THEME['text']}; }}
    .topbar-custom {{ background: {THEME['bar']}; border-radius: 16px; padding: 1.0em 1.1em; margin: 0.8em 0 1.0em; box-shadow: 0 2px 12px 0 rgba(44,46,66,0.06); font-size: 1.3rem; font-weight: 800; letter-spacing: .01em; }}
    .card {{ background: {THEME['bot']}; color: {THEME['text']}; border-radius: 14px; padding: 1em 1.1em; border: 1px solid {THEME['border']}; box-shadow: 0 1px 12px 0 rgba(44,46,66,0.05); }}
    .card-title {{ font-weight: 700; margin-bottom: .5em; }}
    .option {{ margin-top:.5em; padding:.6em .75em; border:1px solid {THEME['border']}; border-radius:12px; }}
    .option.neutral {{ background: {THEME['bg']}; }}
    .option.correct {{ background:#0e4d2e; color:#fff; border-color:#2ea043; }}
    .option.wrong {{ background:#6b2222; color:#fff; border-color:#f85149; }}
    .explain {{ margin-top:.6em; }}
    .explain.good {{ color:#2ea043; font-weight:600; }}
    .explain.bad {{ color:#f85149; font-weight:600; }}
    .explain.note {{ opacity:.9; }}
    .msg-user {{ background: {THEME['user']}; color: {THEME['text']}; border-radius: 16px 16px 4px 20px; margin-bottom: 0.3em; padding: 1em 1.35em; width: fit-content; max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-left: auto; margin-right: 0; text-align: right; box-shadow: 0 1px 12px 0 rgba(55,96,148,0.05); word-break: break-word; }}
    .msg-bot {{ background: {THEME['bot']}; color: {THEME['text']}; border-radius: 16px 16px 20px 4px; margin-bottom: 0.7em; padding: 1.08em 1.23em 1em 1.18em; width: fit-content; max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-right: auto; margin-left: 0; text-align: left; box-shadow: 0 1px 12px 0 rgba(44,46,66,0.05); }}
    [data-testid="stExpander"] {{ border-color: {THEME['border']}; background: {THEME['expander']}; }}
    .stButton>button, .stDownloadButton>button {{ border: 1px solid {THEME['border']}; }}
    .note-text {{ color: #787878; font-size: 0.9rem; }}
    .case-card {{ background: linear-gradient(135deg, #182236 0%, #24324f 100%); color:#f6f7fb; padding:1.0em 1.1em; border-radius: 14px; border:1px solid #2a3350; }}
    .case-title {{ font-weight:800; margin-bottom:.4em; }}
    .case-body {{ opacity:.95; }}
    .case-instr {{ margin:.6em 0 .3em 0; font-weight:600; }}
    @media only screen and (max-width: 768px) {{ .topbar-custom {{ font-size: 1.1rem; padding: .9em; text-align: center; }} .msg-user, .msg-bot {{ font-size: 0.95rem; max-width: 95%; }} }}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='topbar-custom'>Ophtha Bot : AI Chatbot for Postgrad Ophthalmology Students</div>", unsafe_allow_html=True)

# --- Sidebar: MODES (renamed from Settings) + existing controls -------------
with st.sidebar:
    st.header("Modes")
    st.session_state.mode = st.radio(
        "Choose mode",
        options=["Chat", "Teaching", "Exam", "Case"],
        index=["Chat", "Teaching", "Exam", "Case"].index(st.session_state.mode),
        help="Switch between regular chat, tutor-style, MCQ exam practice, or case simulations."
    )

    # Theme toggle
    is_dark_on = st.session_state.theme == "dark"
    toggled = st.toggle("Dark Mode", value=is_dark_on, key="theme_toggle", help="Switch themes")
    if toggled != is_dark_on:
        st.session_state.theme = "dark" if toggled else "light"
        st.rerun()

    st.divider()

    # Teaching Mode flag still controls depth/style in chat
    st.session_state.teaching_mode = (st.session_state.mode == "Teaching")

    st.header("Voice Settings")
    st.session_state.voice_enabled = st.toggle("Enable Voice Chat", value=st.session_state.voice_enabled, help="Enable voice input and spoken responses.")

    if st.session_state.voice_enabled and st.session_state.mode in ["Chat", "Teaching"]:
        input_accent_options = {
            'American (US)': 'en-US', 'British (UK)': 'en-GB', 'Indian': 'en-IN',
            'Australian': 'en-AU', 'Canadian': 'en-CA', 'South African': 'en-ZA'
        }
        try:
            current_accent_index = list(input_accent_options.values()).index(st.session_state.input_accent)
        except ValueError:
            current_accent_index = 0
        selected_input_label = st.selectbox("Your Accent (for input)", options=list(input_accent_options.keys()), index=current_accent_index)
        st.session_state.input_accent = input_accent_options[selected_input_label]
        
        output_accent_options = {'American (US)': 'com', 'British (UK)': 'co.uk', 'Indian': 'co.in'}
        st.session_state.output_accent = output_accent_options[st.selectbox(
            "Assistant's Accent (for output)",
            options=list(output_accent_options.keys()),
            index=list(output_accent_options.values()).index(st.session_state.output_accent)
        )]

# --- Conversation history display (chat modes only) --------------------------
if st.session_state.mode in ["Chat", "Teaching"]:
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

# --- Upload Document (kept as-is; available in all modes) --------------------
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

# --- Main areas per MODE -----------------------------------------------------
if st.session_state.mode == "Exam":
    render_exam_ui()

elif st.session_state.mode == "Case":
    render_case_ui()

else:
    # ============ Chat / Teaching modes use original chat flow ===============
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
                spoken_text = raw_answer_text + " Anything you'd like to explore next?"
                audio_b64 = text_to_audio_b64(spoken_text, st.session_state.output_accent)
                if audio_b64:
                    render_audio_player_b64(audio_b64)

            if pdf_filename:
                pdf_path = os.path.join(CHEATSHEET_PATH, pdf_filename)
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button("📥 Download Cheatsheet", pdf_file.read(), pdf_filename, "application/pdf", key=f"dl_{pdf_filename}_{uuid.uuid4()}")

            st.session_state.chat_history.append({"bot": full_answer_html, "pdf_filename": pdf_filename})
import os
import re
import uuid
import time
import base64
import json
import io
import csv
from datetime import datetime
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

# ---------------- DYNAMIC FOLLOW-UP (warmer, human tone) --------------------
def generate_teaching_followup(user_q: str, explanation: str) -> str:
    """
    ONE short, emotionally intelligent follow-up line:
      - Gentle check, offer to simplify, or tiny self-check (MCQ/short answer).
      - Supportive, conversational tone with contractions.
      - Max 20 words. No emojis/preamble. Don’t reveal answers.
    """
    tmpl = PromptTemplate.from_template(
        """You are an empathetic ophthalmology tutor with a warm, human style.
Write ONE short, emotionally intelligent follow-up line that best fits the learner and the content.

Choose exactly ONE:
- A gentle comprehension check, OR
- An offer to simplify with an analogy/example, OR
- A tiny self-check (MCQ/short answer, no answer revealed).

Tone & style:
- Supportive, conversational, encouraging. Use contractions.
- Avoid robotic phrasing like "Did you understand?" or "Do you want me to explain it easier?".
- Vary natural wording; sound like a caring clinician-teacher, not a script.
- Max 20 words. No emojis. No preamble.

Context
User question: {user_q}
Your explanation: {explanation}

Return only the single follow-up line:"""
    )
    chain = LLMChain(llm=llm, prompt=tmpl)
    out = chain.run(user_q=user_q, explanation=explanation).strip()
    line = re.sub(r"`+", "", out).split("\n")[0]
    line = re.sub(r"\s+", " ", line).strip()
    return line[:200]
# ---------------------------------------------------------------------------

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

# --- RAG/Chat Logic ----------------------------------------------------------
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

    # ---- Append warm dynamic follow-up to ALL ophthalmology answers ----------
    if isinstance(final_answer, str) and final_answer.strip():
        try:
            follow = generate_teaching_followup(query, final_answer)
            if follow:
                final_answer = f"{final_answer}\n\n{follow}"
        except Exception:
            pass
    # ------------------------------------------------------------------------

    return final_answer, pdf_filename

# ============================= NEW: EXAM MODE ================================

def _parse_json_block(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def generate_mcqs(topic: str, num_q: int = 5):
    """Return list of dicts: {question, options:[...], correct_index, explanation}"""
    prompt = PromptTemplate.from_template(
        """You are an ophthalmology exam item writer.
Create {num} high-quality single-best-answer MCQs on: "{topic}"
Constraints:
- Postgraduate level, strictly ophthalmology.
- 4 options (A–D). Exactly one correct.
- Provide a 1–2 line explanation.
- Output ONLY JSON as:
{{
  "mcqs": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "..."
    }}
  ]
}}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run(topic=topic, num=num_q)
    data = _parse_json_block(raw) or {"mcqs": []}
    mcqs = data.get("mcqs", [])
    # simple sanitation
    clean = []
    for q in mcqs[:num_q]:
        if isinstance(q, dict) and all(k in q for k in ("question","options","correct_index","explanation")):
            if isinstance(q["options"], list) and len(q["options"]) == 4:
                ci = int(q["correct_index"]) if str(q["correct_index"]).isdigit() else 0
                ci = max(0, min(3, ci))
                clean.append({
                    "question": q["question"].strip(),
                    "options": [str(x).strip() for x in q["options"]],
                    "correct_index": ci,
                    "explanation": q["explanation"].strip()
                })
    return clean

def render_exam_dashboard(exam_state):
    total = len(exam_state["questions"])
    attempted = len(exam_state["selected"])
    score = sum(1 for i, sel in exam_state["selected"].items() if sel == exam_state["questions"][i]["correct_index"])
    st.session_state.exam["score"] = score
    c1, c2, c3 = st.columns(3)
    c1.metric("Total MCQs", total)
    c2.metric("Attempted", attempted)
    c3.metric("Score", score)

def render_option_badges(q_idx, opts, correct_idx, selected_idx):
    for j, opt in enumerate(opts):
        cls = "neutral"
        if selected_idx is not None:
            if j == correct_idx:
                cls = "correct"
            if selected_idx == j and j != correct_idx:
                cls = "wrong"
        st.markdown(
            f"<div class='option {cls}'><b>{chr(65+j)}.</b> {opt}</div>",
            unsafe_allow_html=True
        )

def render_exam_ui():
    st.markdown("<div class='topbar-custom'>Exam Mode · MCQ Practice</div>", unsafe_allow_html=True)

    topic = st.text_input("Topic for MCQs (ophthalmology only)", placeholder="e.g., Primary open-angle glaucoma")
    cols = st.columns([1,1,2])
    with cols[0]:
        num_q = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5, step=1)
    with cols[1]:
        if st.button("Generate MCQs", use_container_width=True, type="primary"):
            mcqs = generate_mcqs(topic or "general ophthalmology", int(num_q))
            st.session_state.exam = {
                "topic": topic or "general ophthalmology",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "questions": mcqs,
                "selected": {},   # q_idx -> opt_idx
                "score": 0
            }
            st.rerun()

    exam_state = st.session_state.get("exam", None)
    if not exam_state or not exam_state.get("questions"):
        st.info("Enter a topic and click **Generate MCQs** to start.")
        return

    # Dashboard
    render_exam_dashboard(exam_state)
    st.markdown("<br/>", unsafe_allow_html=True)

    # Render each MCQ card
    for i, q in enumerate(exam_state["questions"]):
        st.markdown(f"<div class='card'><div class='card-title'>Q{i+1}. {q['question']}</div>", unsafe_allow_html=True)
        selected = exam_state["selected"].get(i, None)

        # selection buttons
        bcols = st.columns(2)
        for j, opt in enumerate(q["options"]):
            if bcols[j % 2].button(f"{chr(65+j)}. {opt}", key=f"mcq_{i}_{j}"):
                st.session_state.exam["selected"][i] = j
                st.rerun()

        # colored badges after selection
        render_option_badges(i, q["options"], q["correct_index"], selected)

        # show explanation once answered
        if selected is not None:
            if selected == q["correct_index"]:
                st.markdown("<div class='explain good'>Correct ✅</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='explain bad'>Not quite ❌</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='explain note'><b>Why:</b> {q['explanation']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # end card
        st.markdown("<br/>", unsafe_allow_html=True)

    # Refresh dashboard after answers
    render_exam_dashboard(st.session_state.exam)

    # Downloads
    rows = []
    for i, q in enumerate(st.session_state.exam["questions"]):
        sel = st.session_state.exam["selected"].get(i, None)
        rows.append({
            "index": i+1,
            "question": q["question"],
            "A": q["options"][0],
            "B": q["options"][1],
            "C": q["options"][2],
            "D": q["options"][3],
            "selected": "" if sel is None else chr(65+sel),
            "correct": chr(65+q["correct_index"]),
            "is_correct": sel == q["correct_index"] if sel is not None else None,
            "explanation": q["explanation"]
        })

    # CSV
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
    st.download_button(
        "📥 Download MCQ Session (CSV)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"exam_mcqs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # JSON
    json_payload = {
        "topic": st.session_state.exam.get("topic"),
        "generated_at": st.session_state.exam.get("generated_at"),
        "score": st.session_state.exam.get("score"),
        "attempted": len(st.session_state.exam.get("selected", {})),
        "total": len(st.session_state.exam.get("questions", [])),
        "items": rows
    }
    st.download_button(
        "📥 Download MCQ Session (JSON)",
        data=json.dumps(json_payload, indent=2).encode("utf-8"),
        file_name=f"exam_mcqs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

# ============================ NEW: CASE MODE =================================

def generate_case(topic: str):
    """Return dict with a scenario and evaluation keys."""
    prompt = PromptTemplate.from_template(
        """You are an ophthalmology simulation author.
Create ONE realistic case vignette (brief) for postgraduate level on: "{topic}"
Include:
- title
- scenario (2–5 sentences)
- key_points: list of 5–8 bullet keywords (diagnosis+workup+management targets)
Output ONLY JSON as:
{{
  "title": "...",
  "scenario": "...",
  "key_points": ["...", "...", "..."]
}}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run(topic=topic or "general ophthalmology")
    data = _parse_json_block(raw) or {}
    title = data.get("title", "Ophthalmology Case")
    scenario = data.get("scenario", "A patient presents to clinic...")
    key_points = data.get("key_points", [])
    return {"title": title, "scenario": scenario, "key_points": key_points}

def evaluate_case_response(scenario: str, key_points, user_answer: str):
    """Generate feedback + score using key_points as rubric."""
    rubric = "; ".join(key_points[:8])
    prompt = PromptTemplate.from_template(
        """You are grading a short free-text response for an ophthalmology case.
Case: {scenario}
Rubric key points (target ideas): {rubric}
Learner response: {answer}

Return ONLY JSON as:
{{
  "feedback": {{
    "strengths": ["...", "..."],
    "missed": ["...", "..."],
    "suggestions": "one concise paragraph with practical advice"
  }},
  "score": {{
    "achieved": <int 0-100>,
    "explanation": "one line on how the score was decided"
  }}
}}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run(scenario=scenario, rubric=rubric, answer=user_answer)
    data = _parse_json_block(raw) or {}
    fb = data.get("feedback", {})
    sc = data.get("score", {"achieved": 0, "explanation": ""})
    return fb, sc

def render_case_ui():
    st.markdown("<div class='topbar-custom'>Case-Based Mode · Simulation</div>", unsafe_allow_html=True)

    topic = st.text_input("Case focus (ophthalmology only)", placeholder="e.g., Painless vision loss · CRAO vs. NAION")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Generate Case", type="primary", use_container_width=True):
            c = generate_case(topic or "general ophthalmology")
            st.session_state.case = {
                "topic": topic or "general ophthalmology",
                "case": c,
                "response": "",
                "graded": None,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
            st.rerun()

    case_state = st.session_state.get("case", None)
    if not case_state:
        st.info("Enter a focus and click **Generate Case** to start.")
        return

    c = case_state["case"]
    st.markdown(
        f"""
        <div class='case-card'>
            <div class='case-title'>{c['title']}</div>
            <div class='case-body'>{c['scenario']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='case-instr'>Write your impression and next steps (investigations/initial management).</div>", unsafe_allow_html=True)
    st.session_state.case["response"] = st.text_area("Your response", value=case_state.get("response",""), height=160, placeholder="Type your reasoning here…")

    if st.button("Submit Answer", type="primary", use_container_width=True):
        fb, sc = evaluate_case_response(c["scenario"], c.get("key_points", []), st.session_state.case["response"])
        st.session_state.case["graded"] = {"feedback": fb, "score": sc}
        st.rerun()

    graded = case_state.get("graded")
    if graded:
        strengths = graded["feedback"].get("strengths", [])
        missed = graded["feedback"].get("missed", [])
        suggestions = graded["feedback"].get("suggestions","")
        score_val = graded["score"].get("achieved", 0)
        score_exp = graded["score"].get("explanation","")

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Score", f"{score_val}/100")
        cc2.metric("Key hits", len(strengths))
        cc3.metric("Gaps", len(missed))

        st.markdown("<div class='card'><div class='card-title'>Feedback</div>", unsafe_allow_html=True)
        if strengths:
            st.markdown("<b>What you did well</b>", unsafe_allow_html=True)
            for s_ in strengths: st.markdown(f"- {s_}")
        if missed:
            st.markdown("<b>What to add next time</b>", unsafe_allow_html=True)
            for m_ in missed: st.markdown(f"- {m_}")
        if suggestions:
            st.markdown(f"<div class='explain note'>{suggestions}</div>", unsafe_allow_html=True)
        if score_exp:
            st.caption(f"Scoring note: {score_exp}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Download interaction
        payload = {
            "topic": case_state.get("topic"),
            "generated_at": case_state.get("generated_at"),
            "title": c["title"],
            "scenario": c["scenario"],
            "learner_response": st.session_state.case["response"],
            "feedback": graded["feedback"],
            "score": graded["score"]
        }
        st.download_button(
            "📥 Download Case Interaction (JSON)",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name=f"case_interaction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# --- Streamlit UI ---
st.set_page_config(layout="centered")

LIGHT = {"bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd", "text": "#191b22", "input": "#e8edf2", "border": "#d4dde7", "expander": "#f4f7fb"}
DARK = {"bg": "#202126", "bar": "#232733", "bot": "#232733", "user": "#22577a", "text": "#f3f5f8", "input": "#242730", "border": "#26282f", "expander": "#24272e"}

# Initialize session state variables
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "session_id" not in st.session_state: st.session_state.session_id = None
if "active_doc_name" not in st.session_state: st.session_state.active_doc_name = None
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = False
if "input_accent" not in st.session_state: st.session_state.input_accent = 'en-US'
if "output_accent" not in st.session_state: st.session_state.output_accent = 'com'
if "teaching_mode" not in st.session_state: st.session_state.teaching_mode = False
if "mode" not in st.session_state: st.session_state.mode = "Chat"  # Chat | Teaching | Exam | Case

THEME = DARK if st.session_state.theme == "dark" else LIGHT

# --- Global Styles (cards, options, case visuals) ---
st.markdown(f"""
<style>
    .stApp {{ background: {THEME['bg']}; color: {THEME['text']}; }}
    .topbar-custom {{ background: {THEME['bar']}; border-radius: 16px; padding: 1.0em 1.1em; margin: 0.8em 0 1.0em; box-shadow: 0 2px 12px 0 rgba(44,46,66,0.06); font-size: 1.3rem; font-weight: 800; letter-spacing: .01em; }}
    .card {{ background: {THEME['bot']}; color: {THEME['text']}; border-radius: 14px; padding: 1em 1.1em; border: 1px solid {THEME['border']}; box-shadow: 0 1px 12px 0 rgba(44,46,66,0.05); }}
    .card-title {{ font-weight: 700; margin-bottom: .5em; }}
    .option {{ margin-top:.5em; padding:.6em .75em; border:1px solid {THEME['border']}; border-radius:12px; }}
    .option.neutral {{ background: {THEME['bg']}; }}
    .option.correct {{ background:#0e4d2e; color:#fff; border-color:#2ea043; }}
    .option.wrong {{ background:#6b2222; color:#fff; border-color:#f85149; }}
    .explain {{ margin-top:.6em; }}
    .explain.good {{ color:#2ea043; font-weight:600; }}
    .explain.bad {{ color:#f85149; font-weight:600; }}
    .explain.note {{ opacity:.9; }}
    .msg-user {{ background: {THEME['user']}; color: {THEME['text']}; border-radius: 16px 16px 4px 20px; margin-bottom: 0.3em; padding: 1em 1.35em; width: fit-content; max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-left: auto; margin-right: 0; text-align: right; box-shadow: 0 1px 12px 0 rgba(55,96,148,0.05); word-break: break-word; }}
    .msg-bot {{ background: {THEME['bot']}; color: {THEME['text']}; border-radius: 16px 16px 20px 4px; margin-bottom: 0.7em; padding: 1.08em 1.23em 1em 1.18em; width: fit-content; max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-right: auto; margin-left: 0; text-align: left; box-shadow: 0 1px 12px 0 rgba(44,46,66,0.05); word-break: break-word; }}
    [data-testid="stExpander"] {{ border-color: {THEME['border']}; background: {THEME['expander']}; }}
    .stButton>button, .stDownloadButton>button {{ border: 1px solid {THEME['border']}; }}
    .note-text {{ color: #787878; font-size: 0.9rem; }}
    .case-card {{ background: linear-gradient(135deg, #182236 0%, #24324f 100%); color:#f6f7fb; padding:1.0em 1.1em; border-radius: 14px; border:1px solid #2a3350; }}
    .case-title {{ font-weight:800; margin-bottom:.4em; }}
    .case-body {{ opacity:.95; }}
    .case-instr {{ margin:.6em 0 .3em 0; font-weight:600; }}
    @media only screen and (max-width: 768px) {{ .topbar-custom {{ font-size: 1.1rem; padding: .9em; text-align: center; }} .msg-user, .msg-bot {{ font-size: 0.95rem; max-width: 95%; }} }}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='topbar-custom'>Ophtha Bot : AI Chatbot for Postgrad Ophthalmology Students</div>", unsafe_allow_html=True)

# --- Sidebar: MODES (renamed from Settings) + existing controls -------------
with st.sidebar:
    st.header("Modes")  # renamed
    st.session_state.mode = st.radio(
        "Choose mode",
        options=["Chat", "Teaching", "Exam", "Case"],
        index=["Chat", "Teaching", "Exam", "Case"].index(st.session_state.mode),
        help="Switch between regular chat, tutor-style, MCQ exam practice, or case simulations."
    )

    # keep theme toggle
    is_dark_on = st.session_state.theme == "dark"
    toggled = st.toggle("Dark Mode", value=is_dark_on, key="theme_toggle", help="Switch themes")
    if toggled != is_dark_on:
        st.session_state.theme = "dark" if toggled else "light"
        st.rerun()

    st.divider()

    # Teaching Mode flag still controls depth/style in chat
    st.session_state.teaching_mode = (st.session_state.mode == "Teaching")

    st.header("Voice Settings")
    st.session_state.voice_enabled = st.toggle("Enable Voice Chat", value=st.session_state.voice_enabled, help="Enable voice input and spoken responses.")

    if st.session_state.voice_enabled and st.session_state.mode in ["Chat", "Teaching"]:
        input_accent_options = {
            'American (US)': 'en-US', 'British (UK)': 'en-GB', 'Indian': 'en-IN',
            'Australian': 'en-AU', 'Canadian': 'en-CA', 'South African': 'en-ZA'
        }
        try:
            current_accent_index = list(input_accent_options.values()).index(st.session_state.input_accent)
        except ValueError:
            current_accent_index = 0
        selected_input_label = st.selectbox("Your Accent (for input)", options=list(input_accent_options.keys()), index=current_accent_index)
        st.session_state.input_accent = input_accent_options[selected_input_label]
        
        output_accent_options = {'American (US)': 'com', 'British (UK)': 'co.uk', 'Indian': 'co.in'}
        st.session_state.output_accent = output_accent_options[st.selectbox(
            "Assistant's Accent (for output)",
            options=list(output_accent_options.keys()),
            index=list(output_accent_options.values()).index(st.session_state.output_accent)
        )]

# --- Conversation history display (chat modes only) --------------------------
if st.session_state.mode in ["Chat", "Teaching"]:
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

# --- Upload Document (kept as-is; available in all modes) --------------------
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

# --- Main areas per MODE -----------------------------------------------------
if st.session_state.mode == "Exam":
    render_exam_ui()

elif st.session_state.mode == "Case":
    render_case_ui()

else:
    # ============ Chat / Teaching modes use original chat flow ===============
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
                spoken_text = raw_answer_text + " Anything you'd like to explore next?"
                audio_b64 = text_to_audio_b64(spoken_text, st.session_state.output_accent)
                if audio_b64:
                    render_audio_player_b64(audio_b64)

            if pdf_filename:
                pdf_path = os.path.join(CHEATSHEET_PATH, pdf_filename)
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button("📥 Download Cheatsheet", pdf_file.read(), pdf_filename, "application/pdf", key=f"dl_{pdf_filename}_{uuid.uuid4()}")

            st.session_state.chat_history.append({"bot": full_answer_html, "pdf_filename": pdf_filename})
