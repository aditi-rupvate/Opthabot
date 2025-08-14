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

# ── CONFIG ─────────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_DEFAULT_API_KEY_HERE")
FAISS_INDEX_PATH = "oxford_handbook_kb"
TEMP_STORAGE_PATH = "temp_user_docs"
CHEATSHEET_PATH = "downloads"
os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
os.makedirs(CHEATSHEET_PATH, exist_ok=True)

disclaimer_text = "— Note: This output is for academic purposes only and must not be used for clinical diagnosis."

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# ── OPHTHALMOLOGY DOMAIN GATE ─────────────────────────────────────────────────
OPHTH_KEYWORDS = [
    "eye","ocular","ophthalmology","ophthalmic","vision","visual acuity","refraction",
    "cornea","conjunctiva","sclera","anterior chamber","iris","pupil","lens",
    "vitreous","retina","macula","fovea","optic nerve","optic disc",
    "cataract","glaucoma","amd","age-related macular degeneration","diabetic retinopathy",
    "dr","csr","central serous","uveitis","keratoconus","dry eye","meibomian",
    "blepharitis","strabismus","amblyopia","endophthalmitis","retinal detachment",
    "retinitis pigmentosa","slit lamp","gonioscopy","tonometry","intraocular pressure","iop",
    "oct","optical coherence tomography","perimetry","visual field","fundus",
    "ophthalmoscopy","fluorescein angiography","b-scan","vitrectomy","trabeculectomy",
    "latanoprost","timolol","brimonidine","dorzolamide","pilocarpine",
    "prednisolone","moxifloxacin","cyclopentolate","tropicamide","red eye","floaters","flashes",
]
GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey)\b",
    r"^\s*good (morning|afternoon|evening)\b",
    r"^\s*(how are you|how's it going|what's up)\b",
    r"^\s*(thanks|thank you)\b",
    r"^\s*(bye|goodbye|see you)\b",
]
def _is_greeting(t: str) -> bool: return any(re.search(p, t.lower().strip()) for p in GREETING_PATTERNS)
def _is_ophthalmology(t: str) -> bool: return any(k in t.lower() for k in OPHTH_KEYWORDS)
DOMAIN_REFUSAL = "I'm specialised in ophthalmology (eye care) and basic greetings only. Please ask an eye-related question."

# ── DYNAMIC FOLLOW-UP (teaching tone) ─────────────────────────────────────────
def generate_teaching_followup(user_q: str, explanation: str) -> str:
    tmpl = PromptTemplate.from_template(
        """You are an empathetic ophthalmology tutor.
Write ONE short, warm follow-up line that best fits the learner and the content.
Pick exactly ONE: a gentle comprehension check OR offer to simplify OR a tiny self-check (no answer).
Tone: friendly, conversational, no robotic phrasing, <= 20 words, no emojis.

User question: {user_q}
Your explanation: {explanation}

Return only the line:"""
    )
    out = LLMChain(llm=llm, prompt=tmpl).invoke({"user_q": user_q, "explanation": explanation})["text"].strip()
    out = re.sub(r"`+", "", out).split("\n")[0]
    return re.sub(r"\s+", " ", out)[:200]

# ── TTS ───────────────────────────────────────────────────────────────────────
def text_to_audio_b64(text: str, tld: str) -> str | None:
    try:
        tts = gTTS(text=text, lang='en', tld=tld, slow=False)
        audio_filename = os.path.join(CHEATSHEET_PATH, f"response_{uuid.uuid4()}.mp3")
        tts.save(audio_filename)
        with open(audio_filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        try: os.remove(audio_filename)
        except OSError: pass
        return b64
    except Exception as e:
        st.warning(f"Could not generate audio response: {e}")
        return None

def render_audio_player_b64(audio_b64: str):
    aid = f"audio_{uuid.uuid4().hex}"
    st.markdown(f"""
    <audio id="{aid}" autoplay playsinline preload="auto" style="width:0;height:0;visibility:hidden;"
           controlslist="nodownload noplaybackrate"
           src="data:audio/mpeg;base64,{audio_b64}">
      <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
    </audio>
    <script>(function(){{
      const a=document.getElementById("{aid}");
      if(!a)return;
      function go(){const p=a.play();if(p&&p.then)p.catch(()=>{const u=()=>{{a.play().catch(()=>{{}});document.removeEventListener('touchstart',u,true);document.removeEventListener('click',u,true);}};document.addEventListener('touchstart',u,true);document.addEventListener('click',u,true);});}
      if(document.readyState!=="loading")go();else document.addEventListener('DOMContentLoaded',go,{once:true});
    }})();</script>
    """, unsafe_allow_html=True)

# ── Unicode-safe PDF helpers ──────────────────────────────────────────────────
def _add_dejavu(pdf: FPDF):
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
        try: pdf.add_font("DejaVu", "I", "DejaVuSans-Oblique.ttf", uni=True); italic=True
        except Exception: italic=False
        return "DejaVu", True, italic
    except RuntimeError:
        return "Helvetica", False, True

def _safe(unicode_ok: bool, s: str) -> str:
    return s if unicode_ok else s.encode("latin-1", "ignore").decode("latin-1")

def create_exam_pdf(rows, meta) -> str:
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    FONT, U, IT = _add_dejavu(pdf)
    check, cross = ("✔ ", "✖ ") if U else ("[OK] ", "[X] ")
    pdf.set_font(FONT, "B", 16); pdf.cell(0, 10, _safe(U, "Ophthalmology MCQ Session"), ln=1)
    pdf.set_font(FONT, "", 11)
    hdr = f"Topic: {meta.get('topic','-')}  |  Score: {meta.get('score',0)}/{meta.get('total',0)}  |  Attempted: {meta.get('attempted',0)}  |  Generated: {meta.get('generated_at','')}"
    pdf.multi_cell(0, 6, _safe(U, hdr)); pdf.ln(2)
    for r in rows:
        pdf.set_font(FONT, "B", 12); pdf.multi_cell(0, 7, _safe(U, f"Q{r['index']}. {r['question']}"))
        pdf.set_font(FONT, "", 11)
        for k in ["A","B","C","D"]:
            tag=""
            if r["correct"]==k: tag=check
            if r["selected"]==k and r["selected"]!=r["correct"]: tag=cross
            pdf.multi_cell(0, 6, _safe(U, f"{tag}{k}. {r[k]}"))
        pdf.set_font(FONT, "I" if IT else "", 10); pdf.set_text_color(60,60,60)
        pdf.multi_cell(0, 5, _safe(U, f"Why: {r['explanation']}"))
        pdf.set_text_color(0,0,0); pdf.ln(2)
    filename=f"exam_mcqs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    path=os.path.join(CHEATSHEET_PATH, filename); pdf.output(path); return path

def create_case_pdf(payload) -> str:
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    FONT,U,_ = _add_dejavu(pdf)
    pdf.set_font(FONT, "B", 16); pdf.cell(0,10,_safe(U,"Ophthalmology Case Interaction"),ln=1)
    pdf.set_font(FONT, "", 11)
    pdf.multi_cell(0,6,_safe(U,f"Topic: {payload.get('topic','-')}"))
    pdf.multi_cell(0,6,_safe(U,f"Generated: {payload.get('generated_at','')}")); pdf.ln(2)
    pdf.set_font(FONT,"B",12); pdf.multi_cell(0,7,_safe(U,f"Title: {payload.get('title','')}"))
    pdf.set_font(FONT,"",11);  pdf.multi_cell(0,6,_safe(U,f"Scenario: {payload.get('scenario','')}")); pdf.ln(1)
    pdf.set_font(FONT,"B",12); pdf.multi_cell(0,7,_safe(U,"Your Response"))
    pdf.set_font(FONT,"",11);  pdf.multi_cell(0,6,_safe(U,payload.get("learner_response","")))
    fb=payload.get("feedback",{}); strengths=fb.get("strengths",[]); missed=fb.get("missed",[]); suggestions=fb.get("suggestions","")
    pdf.ln(2); pdf.set_font(FONT,"B",12); pdf.multi_cell(0,7,_safe(U,"Feedback")); pdf.set_font(FONT,"",11)
    if strengths: pdf.multi_cell(0,6,_safe(U,"What you did well: "+"; ".join(strengths)))
    if missed:    pdf.multi_cell(0,6,_safe(U,"What to add next time: "+"; ".join(missed)))
    if suggestions: pdf.multi_cell(0,6,_safe(U,"Suggestions: "+suggestions))
    sc=payload.get("score",{}); pdf.ln(1)
    pdf.multi_cell(0,6,_safe(U,f"Score: {sc.get('achieved',0)}/100 — {sc.get('explanation','')}"))
    filename=f"case_interaction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    path=os.path.join(CHEATSHEET_PATH, filename); pdf.output(path); return path

def create_flash_pdf(cards, meta) -> str:
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    FONT,U,_ = _add_dejavu(pdf)
    pdf.set_font(FONT,"B",16); pdf.cell(0,10,_safe(U,"Ophthalmology Flashcards"),ln=1)
    pdf.set_font(FONT,"",11)
    hdr=f"Topic: {meta.get('topic','-')} | Reviewed: {meta.get('reviewed',0)}/{meta.get('total',0)} | Correct: {meta.get('correct',0)} | Generated: {meta.get('generated_at','')}"
    pdf.multi_cell(0,6,_safe(U,hdr)); pdf.ln(2)
    for i,c in enumerate(cards,1):
        pdf.set_font(FONT,"B",12); pdf.multi_cell(0,7,_safe(U,f"Card {i}: {c['front']}"))
        pdf.set_font(FONT,"",11);  pdf.multi_cell(0,6,_safe(U,"Answer: "+c["back"]))
        m=c.get("mark")
        if m is True:  pdf.multi_cell(0,6,_safe(U,"Marked: ✔ Correct" if U else "Marked: OK"))
        if m is False: pdf.multi_cell(0,6,_safe(U,"Marked: ✖ Incorrect" if U else "Marked: X"))
        pdf.ln(2)
    filename=f"flashcards_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    path=os.path.join(CHEATSHEET_PATH, filename); pdf.output(path); return path

# ── Cheatsheet PDF (existing feature) ─────────────────────────────────────────
class PDF(FPDF):
    def __init__(self, topic,*a,**k): super().__init__(*a,**k); self.topic=topic
    def header(self):
        self.set_font("DejaVu","B",9); self.set_text_color(128,128,128)
        self.cell(0,10,f"Ophthalmology Cheatsheet: {self.topic.title()}",0,0,'L'); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font("DejaVu","",8); self.set_text_color(128,128,128)
        self.cell(0,10,f"Page {self.page_no()}/{{nb}}",0,0,'C')

def create_formatted_pdf(text_content: str, topic: str) -> str:
    pdf = PDF(topic)
    try:
        pdf.add_font("DejaVu","", "DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu","B","DejaVuSans-Bold.ttf", uni=True)
    except RuntimeError:
        st.error("Missing DejaVu fonts."); return ""
    pdf.alias_nb_pages(); pdf.add_page()
    pdf.set_margins(15,15,15); pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_font("DejaVu","B",20); pdf.set_text_color(40,40,40)
    pdf.multi_cell(0,10,f"Cheatsheet: {topic.title()}",0,'C'); pdf.ln(2)
    pdf.set_draw_color(200,200,200); pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x()+180, pdf.get_y()); pdf.ln(10)
    line_h=7; pdf.set_text_color(50,50,50)
    for line in text_content.split('\n'):
        line=line.strip()
        if not line: continue
        if line.startswith('## '):
            pdf.set_font("DejaVu","B",14); pdf.set_text_color(0,80,150)
            pdf.multi_cell(0,line_h,line.replace('## ',''))
            pdf.set_text_color(50,50,50); pdf.ln(2)
        elif line.startswith('- '):
            pdf.set_font("DejaVu","",11); pdf.set_x(20)
            pdf.multi_cell(0,line_h,f"• {line[2:]}"); pdf.ln(1)
        else:
            pdf.set_font("DejaVu","",11); pdf.multi_cell(0,line_h,line)
    pdf.ln(5); pdf.set_draw_color(200,200,200)
    x=pdf.get_x(); pdf.line(x, pdf.get_y(), x+180, pdf.get_y()); pdf.ln(4)
    pdf.set_font("DejaVu","",8); pdf.set_text_color(120,120,120)
    pdf.multi_cell(0,6,"Note: This content is for academic purposes only and must not be used for clinical diagnosis.")
    clean=re.sub(r'[\W_]+','_',topic).lower()
    filename=f"{clean}_cheatsheet.pdf"; path=os.path.join(CHEATSHEET_PATH, filename); pdf.output(path); return filename

# ── RAG / Chat logic ──────────────────────────────────────────────────────────
def handle_query_logic(query: str, session_id: str = None):
    if _is_greeting(query): return "Hello! I’m your ophthalmology-only assistant. How can I help with eyes/vision today?", None
    if not _is_ophthalmology(query): return DOMAIN_REFUSAL, None

    if session_id:
        temp_db_path = os.path.join(TEMP_STORAGE_PATH, session_id)
        if not os.path.exists(temp_db_path): return "Error: Your document session has expired.", None
        db = FAISS.load_local(temp_db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(FAISS_INDEX_PATH): return "Error: Default knowledge base not available.", None
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def qa(q: str) -> str:
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return chain.invoke(q)['result']

    def explain(topic: str) -> str:
        ctx = "\n\n".join([d.page_content for d in retriever.get_relevant_documents(topic)])
        pt = PromptTemplate.from_template("Provide a comprehensive explanation for {topic}.\n\nContext: {context}\nResponse:")
        return LLMChain(llm=llm, prompt=pt).invoke({"topic":topic,"context":ctx})["text"]

    def cheatsheet(topic: str) -> str:
        ctx = "\n\n".join([d.page_content for d in retriever.get_relevant_documents(topic)])
        pt = PromptTemplate.from_template("Create a detailed cheat sheet for {topic} using '##' for headings and '-' for bullets.\nContext: {context}\nCheat Sheet:")
        txt = LLMChain(llm=llm, prompt=pt).invoke({"topic":topic,"context":ctx})["text"]
        pdf_filename = create_formatted_pdf(txt, topic)
        return f"PDF_GENERATED::{pdf_filename}::{txt}"

    tools = [
        StructuredTool.from_function(func=qa, name="QuestionAnswerTool", description="Use for direct questions."),
        StructuredTool.from_function(func=explain, name="ConceptExplainerTool", description="Use for summaries/explanations."),
        StructuredTool.from_function(func=cheatsheet, name="CheatsheetGeneratorTool", description="Use ONLY for a downloadable 'cheat sheet'.")
    ]
    base_prompt = hub.pull("hwchase17/react-chat")

    teaching_mode = st.session_state.get("teaching_mode", False)
    if teaching_mode:
        sys = ("You are an expert ophthalmology tutor. Rules: "
               "1) Strictly ophthalmology. 2) Clear steps, define jargon inline, concise examples. "
               "3) Use short paragraphs or bullets. 4) Postgraduate level. "
               "5) End with one warm dynamic follow-up line only. 6) Decline other domains.")
    else:
        sys = ("You are an expert ophthalmology assistant. Only answer ophthalmology or document-related queries. "
               "Politely refuse anything outside ophthalmology.")

    prompt = base_prompt.partial(system_message=sys)
    agent = create_react_agent(llm, tools, prompt)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for msg in st.session_state.chat_history:
        if "user" in msg: memory.chat_memory.add_user_message(msg["user"])
        else: memory.chat_memory.add_ai_message(re.sub(r'<.*?>','',msg["bot"]))

    execu = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False,
                          handle_parsing_errors=True, return_intermediate_steps=True)
    resp = execu.invoke({"input": query})
    final_answer = resp.get('output', "I couldn't find an answer.")
    pdf_filename = None
    for _, obs in resp.get('intermediate_steps', []):
        if isinstance(obs, str) and obs.startswith("PDF_GENERATED::"):
            try: pdf_filename = obs.split("::")[1]
            except IndexError: pass

    try:
        follow = generate_teaching_followup(query, final_answer)
        if follow: final_answer = f"{final_answer}\n\n{follow}"
    except Exception:
        pass
    return final_answer, pdf_filename

# ── Utils ─────────────────────────────────────────────────────────────────────
def _parse_json_block(text: str):
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: return None
    return None

# ── Difficulty guides ─────────────────────────────────────────────────────────
DIFF_LEVELS = ["easy","medium","difficult","extra difficult","extremely difficult"]
DIFF_GUIDE_MCQ = {
    "easy":"recall/common facts; no tricks.",
    "medium":"apply to common scenarios; mild distractors.",
    "difficult":"multi-step reasoning; nuanced management.",
    "extra difficult":"atypical; subtle traps; multi-order reasoning.",
    "extremely difficult":"expert level; rare but examinable nuances."
}
DIFF_GUIDE_CASE = {
    "easy":"classic presentation; clear path; 4–5 points.",
    "medium":"slightly tricky differential; 5–6 points.",
    "difficult":"atypical; broader differential; 6–7 points.",
    "extra difficult":"multisystem clues; 7–8 points.",
    "extremely difficult":"edge-case; rare pitfalls; 8 points."
}

# ── EXAM: MCQs ────────────────────────────────────────────────────────────────
def generate_mcqs(topic: str, num_q: int = 5, difficulty: str = "medium"):
    diff = difficulty if difficulty in DIFF_LEVELS else "medium"
    guide = DIFF_GUIDE_MCQ[diff]
    template = (
        "You are an ophthalmology exam item writer.\n"
        "Create {num} single-best-answer MCQs on: \"{topic}\"\n"
        "Difficulty: {diff}. Guidance: {guide}\n"
        "- 4 options (A–D); exactly one correct.\n"
        "- 1–2 line explanation focused on decision point.\n"
        "Output ONLY JSON: {{\"mcqs\":[{{\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],\"correct_index\":0,\"explanation\":\"...\"}}]}}"
    )
    raw = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["topic","num","diff","guide"], template=template
    )).invoke({"topic":topic or "general ophthalmology","num":num_q,"diff":diff,"guide":guide})["text"]
    data = _parse_json_block(raw) or {"mcqs":[]}
    clean=[]
    for q in data.get("mcqs",[])[:num_q]:
        if isinstance(q,dict) and {"question","options","correct_index","explanation"}<=set(q.keys()) and len(q["options"])==4:
            try: ci=int(q["correct_index"])
            except Exception: ci=0
            ci=max(0,min(3,ci))
            clean.append({"question":q["question"].strip(),"options":[str(x).strip() for x in q["options"]],
                          "correct_index":ci,"explanation":q["explanation"].strip()})
    return clean

def render_exam_dashboard(state):
    total=len(state["questions"]); attempted=len(state["selected"])
    unattempted=max(0,total-attempted)
    score=sum(1 for i,s in state["selected"].items() if s==state["questions"][i]["correct_index"])
    st.session_state.exam["score"]=score
    c1,c2,c3,c4=st.columns(4); c1.metric("Total MCQs", total)
    c2.metric("Attempted", attempted); c3.metric("Unattempted", unattempted); c4.metric("Score", score)

def render_exam_ui():
    st.markdown(f"""
    <style>
      .exam-scope .card {{background:{THEME['bot']}; color:{THEME['text']}; border-radius:14px; padding:1em 1.1em; border:1px solid {THEME['border']};}}
      .exam-scope .card-title {{font-weight:700; margin-bottom:.6em;}}
      .exam-scope .option {{margin:.35em 0; padding:.7em .85em; border:1px solid {THEME['border']}; border-radius:12px; background:{THEME['bg']};}}
      .exam-scope .option.correct {{background:#0e4d2e; color:#fff; border-color:#2ea043;}}
      .exam-scope .option.wrong {{background:#6b2222; color:#fff; border-color:#f85149;}}
      .row-align .stButton>button {{margin-top:2.15rem !important;}}
      @media (max-width:900px) {{.row-align .stButton>button {{margin-top:.25rem !important;}}}}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='topbar-custom'>Exam Mode · MCQ Practice</div>", unsafe_allow_html=True)

    topic = st.text_input("Topic for MCQs (ophthalmology only)", placeholder="e.g., Primary open-angle glaucoma", key="mcq_topic")
    c_num,c_diff,c_btn = st.columns([1,1,1], gap="small")
    with c_num:
        num_q = st.number_input("Number of MCQs", 1, 20, 5, key="mcq_num")
    with c_diff:
        difficulty = st.selectbox("Difficulty", DIFF_LEVELS, index=1, key="mcq_difficulty")
    with c_btn:
        st.markdown("<div class='row-align'>", unsafe_allow_html=True)
        if st.button("Generate MCQs", type="primary", use_container_width=True, key="mcq_generate"):
            with st.spinner("Generating MCQs…"):
                mcqs = generate_mcqs(topic or "general ophthalmology", int(num_q), difficulty)
            st.session_state.exam = {"topic":topic or "general ophthalmology","generated_at":datetime.utcnow().isoformat()+"Z",
                                     "questions":mcqs,"selected":{},"score":0,"difficulty":difficulty}
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    state = st.session_state.get("exam")
    if not state or not state.get("questions"):
        st.info("Enter a topic, choose difficulty, and click **Generate MCQs** to start."); return

    st.caption(f"Difficulty: {state.get('difficulty','medium').title()}"); render_exam_dashboard(state); st.markdown("<br/>", unsafe_allow_html=True)

    for i,q in enumerate(state["questions"]):
        st.markdown(f"<div class='exam-scope card'><div class='card-title'>Q{i+1}. {q['question']}</div>", unsafe_allow_html=True)
        sel = state["selected"].get(i)
        if sel is None:
            cols = st.columns(2, gap="small")
            for j,opt in enumerate(q["options"]):
                with cols[j%2]:
                    if st.button(f"{chr(65+j)}. {opt}", key=f"mcq_{i}_{j}", use_container_width=True):
                        with st.spinner("Checking…"):
                            st.session_state.exam["selected"][i]=j
                        st.rerun()
        else:
            for j,opt in enumerate(q["options"]):
                cls = "correct" if j==q["correct_index"] else ("wrong" if sel==j else "")
                st.markdown(f"<div class='option {cls}'>{chr(65+j)}. {opt}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='explain note'><b>Why:</b> {q['explanation']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True); st.markdown("<br/>", unsafe_allow_html=True)

    render_exam_dashboard(st.session_state.exam)

    rows=[]
    for i,q in enumerate(state["questions"]):
        sel=state["selected"].get(i)
        rows.append({"index":i+1,"question":q["question"],"A":q["options"][0],"B":q["options"][1],"C":q["options"][2],"D":q["options"][3],
                     "selected":"" if sel is None else chr(65+sel),"correct":chr(65+q["correct_index"]),
                     "is_correct":(sel==q["correct_index"]) if sel is not None else None,"explanation":q["explanation"]})
    if rows:
        buf=io.StringIO(); w=csv.DictWriter(buf, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        st.download_button("📥 Download MCQ Session (CSV)", buf.getvalue().encode("utf-8"),
                           file_name=f"exam_mcqs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)
        meta={"topic":state["topic"],"generated_at":state["generated_at"],"score":state["score"],
              "attempted":len(state["selected"]), "total":len(state["questions"])}
        with st.spinner("Preparing PDF…"):
            pdf_path = create_exam_pdf(rows, meta)
        with open(pdf_path,"rb") as f:
            st.download_button("📥 Download MCQ Session (PDF)", f.read(),
                               file_name=os.path.basename(pdf_path), mime="application/pdf",
                               use_container_width=True)

# ── CASE MODE ─────────────────────────────────────────────────────────────────
def generate_case(topic: str, difficulty: str="medium"):
    diff=difficulty if difficulty in DIFF_LEVELS else "medium"; guide=DIFF_GUIDE_CASE[diff]
    tmpl=("You are an ophthalmology simulation author.\nCreate ONE case on: \"{topic}\"\n"
          "Difficulty: {diff}. Guidance: {guide}\nInclude: title, scenario (2–5 sentences), key_points list.\n"
          "Output ONLY JSON: {\"title\":\"...\",\"scenario\":\"...\",\"key_points\":[\"...\"]}")
    raw=LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["topic","diff","guide"], template=tmpl))\
        .invoke({"topic":topic or "general ophthalmology","diff":diff,"guide":guide})["text"]
    data=_parse_json_block(raw) or {}
    return {"title":data.get("title","Ophthalmology Case"),
            "scenario":data.get("scenario","A patient presents..."),
            "key_points":data.get("key_points",[]),
            "difficulty":diff}

def evaluate_case_response(scenario: str, key_points, ans: str):
    rubric="; ".join(key_points[:8])
    tmpl=("You grade a short free-text ophthalmology case answer.\nCase: {scenario}\nRubric: {rubric}\nAnswer: {answer}\n"
          "Return ONLY JSON: {\"feedback\":{\"strengths\":[\"...\"],\"missed\":[\"...\"],\"suggestions\":\"...\"},"
          "\"score\":{\"achieved\":0,\"explanation\":\"...\"}}")
    raw=LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["scenario","rubric","answer"], template=tmpl))\
        .invoke({"scenario":scenario,"rubric":rubric,"answer":ans})["text"]
    data=_parse_json_block(raw) or {}
    return data.get("feedback",{}), data.get("score",{"achieved":0,"explanation":""})

def render_case_ui():
    st.markdown("<div class='topbar-custom'>Case-Based Mode · Simulation</div>", unsafe_allow_html=True)
    topic = st.text_input("Case focus (ophthalmology only)", placeholder="e.g., Painless vision loss · CRAO vs. NAION")
    c1,c2=st.columns([1,1], gap="small")
    with c1:
        diff=st.selectbox("Difficulty", DIFF_LEVELS, index=1, key="case_diff")
    with c2:
        st.markdown("<div class='row-align'>", unsafe_allow_html=True)
        if st.button("Generate Case", type="primary", use_container_width=True):
            with st.spinner("Generating case…"):
                c=generate_case(topic or "general ophthalmology", diff)
            st.session_state.case={"topic":topic or "general ophthalmology","case":c,"response":"", "graded":None,
                                   "generated_at":datetime.utcnow().isoformat()+"Z"}
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    state=st.session_state.get("case")
    if not state:
        with st.spinner("Loading case mode…"): time.sleep(0.3)
        st.info("Enter a focus, choose difficulty, and click **Generate Case** to start."); return

    c=state["case"]; st.caption(f"Difficulty: {c.get('difficulty','medium').title()}")
    st.markdown(f"""
    <div class='card'><div class='card-title'>{c['title']}</div>
    <div>{c['scenario']}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='case-instr'>Write your impression and next steps.</div>", unsafe_allow_html=True)
    st.session_state.case["response"]=st.text_area("Your response", value=state.get("response",""),
                                                   height=160, placeholder="Type your reasoning here…")
    if st.button("Submit Answer", type="primary", use_container_width=True):
        with st.spinner("Scoring your response…"):
            fb,sc=evaluate_case_response(c["scenario"], c.get("key_points",[]), st.session_state.case["response"])
        st.session_state.case["graded"]={"feedback":fb,"score":sc}; st.rerun()

    graded=state.get("graded")
    if graded:
        st.markdown("<div class='card'><div class='card-title'>Feedback</div>", unsafe_allow_html=True)
        for s in graded["feedback"].get("strengths",[]): st.markdown(f"- {s}")
        for m in graded["feedback"].get("missed",[]): st.markdown(f"- {m}")
        sug=graded["feedback"].get("suggestions",""); 
        if sug: st.markdown(f"<div class='explain note'>{sug}</div>", unsafe_allow_html=True)
        st.caption(f"Score: {graded['score'].get('achieved',0)}/100 — {graded['score'].get('explanation','')}")
        st.markdown("</div>", unsafe_allow_html=True)

        payload={"topic":state["topic"],"generated_at":state["generated_at"],"title":c["title"],"scenario":c["scenario"],
                 "learner_response":st.session_state.case["response"],"feedback":graded["feedback"],"score":graded["score"]}
        buf=io.StringIO(); fields=["topic","generated_at","title","scenario","learner_response","strengths","missed","suggestions","score","score_note"]
        w=csv.DictWriter(buf, fieldnames=fields); w.writeheader()
        w.writerow({"topic":payload["topic"],"generated_at":payload["generated_at"],"title":payload["title"],"scenario":payload["scenario"],
                    "learner_response":payload["learner_response"],"strengths":"; ".join(payload["feedback"].get("strengths",[])),
                    "missed":"; ".join(payload["feedback"].get("missed",[])),"suggestions":payload["feedback"].get("suggestions",""),
                    "score":payload["score"].get("achieved",0),"score_note":payload["score"].get("explanation","")})
        st.download_button("📥 Download Case Interaction (CSV)", buf.getvalue().encode("utf-8"),
                           file_name=f"case_interaction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)
        with st.spinner("Preparing PDF…"):
            pdf_path=create_case_pdf(payload)
        with open(pdf_path,"rb") as f:
            st.download_button("📥 Download Case Interaction (PDF)", f.read(),
                               file_name=os.path.basename(pdf_path), mime="application/pdf",
                               use_container_width=True)

# ── FLASHCARDS ────────────────────────────────────────────────────────────────
def _escape_html(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def generate_flashcards(topic: str, num_cards: int = 10):
    tmpl=("Create {num} concise ophthalmology flashcards on \"{topic}\". "
          "Each: front=short prompt (<=18 words), back=crisp answer (1–3 bullets or short paragraph). "
          "Output ONLY JSON: {\"cards\":[{\"front\":\"...\",\"back\":\"...\"}]} ")
    raw=LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["topic","num"], template=tmpl))\
        .invoke({"topic":topic or "general ophthalmology","num":int(num_cards)})["text"]
    data=_parse_json_block(raw) or {"cards":[]}
    cards=[]
    for c in data.get("cards",[])[:int(num_cards)]:
        if isinstance(c,dict) and "front" in c and "back" in c:
            cards.append({"front":str(c["front"]).strip(),"back":str(c["back"]).strip(),"mark":None})
    return cards

def render_flash_dashboard(fs):
    total=len(fs["cards"]); reviewed=sum(1 for c in fs["cards"] if c.get("mark") is not None)
    correct=sum(1 for c in fs["cards"] if c.get("mark") is True); incorrect=sum(1 for c in fs["cards"] if c.get("mark") is False)
    remaining=max(0,total-reviewed); c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Total", total); c2.metric("Reviewed", reviewed); c3.metric("Correct", correct); c4.metric("Incorrect", incorrect); c5.metric("Remaining", remaining)

def render_flash_ui():
    st.markdown("<div class='topbar-custom'>Flashcards Mode · Rapid Recall</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <style>
      #flash-scope .flip-wrap {{ perspective:1200px; width:min(720px,95%); margin:0 auto .75rem; }}
      #flash-scope .flip-card {{ width:100%; height:320px; border-radius:18px; border:1px solid {THEME['border']};
        background:linear-gradient(135deg,#1a2233 0%,#2a3550 100%); box-shadow:0 10px 28px rgba(0,0,0,.25);
        position:relative; cursor:pointer; outline:none; }}
      #flash-scope .flip-card-inner {{ position:relative; width:100%; height:100%; transform-style:preserve-3d;
        transition:transform .55s cubic-bezier(.2,.7,.2,1); }}
      #flash-scope .flip-card.flipped .flip-card-inner {{ transform: rotateY(180deg); }}
      #flash-scope .side {{ position:absolute; inset:0; backface-visibility:hidden; border-radius:18px; display:flex;
        flex-direction:column; justify-content:center; align-items:center; padding:1.2rem; color:#f6f7fb; }}
      #flash-scope .back {{ transform:rotateY(180deg); background:linear-gradient(135deg,#26375b 0%,#1e2a45 100%); }}
      #flash-scope .front .hint, #flash-scope .back .hint {{ position:absolute; bottom:12px; opacity:.85; font-size:.95rem; }}
      #flash-scope .front .hint::before {{ content:"Tap to reveal"; }}
      #flash-scope .back .hint::before {{ content:"Swipe up for next"; }}
      #flash-scope .front .q, #flash-scope .back .a {{ max-width:92%; text-align:center; font-size:1.15rem; line-height:1.35; white-space:pre-wrap; word-break:break-word; }}
      #flash-controls {{ width:min(720px,95%); margin:.5rem auto 0; display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
      #flash-controls .stButton>button {{ height:56px; border-radius:12px; }}
      .row-align .stButton>button {{ margin-top:2.15rem !important; }} @media (max-width:900px){{ .row-align .stButton>button {{ margin-top:.25rem !important; }} }}
    </style>
    """, unsafe_allow_html=True)

    topic = st.text_input("Flashcards topic (ophthalmology only)", placeholder="e.g., Glaucoma medications")
    c_num,c_btn = st.columns([1,1], gap="small")
    with c_num: n = st.number_input("Cards", 3, 40, 10, key="flash_num")
    with c_btn:
        st.markdown("<div class='row-align'>", unsafe_allow_html=True)
        if st.button("Generate Deck", type="primary", use_container_width=True, key="flash_generate"):
            with st.spinner("Building your deck…"):
                deck = generate_flashcards(topic or "general ophthalmology", int(n))
            st.session_state.flash={"topic":topic or "general ophthalmology","generated_at":datetime.utcnow().isoformat()+"Z","cards":deck,"idx":0}
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    fs=st.session_state.get("flash")
    if not fs or not fs.get("cards"):
        st.info("Enter a topic and click **Generate Deck** to begin."); return

    idx=int(fs.get("idx",0)); total=len(fs["cards"]); idx=max(0,min(total-1,idx)); st.session_state.flash["idx"]=idx
    render_flash_dashboard(fs); st.markdown("<br/>", unsafe_allow_html=True)

    card=fs["cards"][idx]; front=_escape_html(card["front"]); back=_escape_html(card["back"])
    card_id=f"fc_{idx}_{uuid.uuid4().hex[:6]}"; next_host_id=f"next_host_{idx}"

    st.markdown(f"""
    <div id="flash-scope" tabindex="0">
      <div class="flip-wrap">
        <div id="{card_id}" class="flip-card" role="button" aria-label="Flashcard (tap to flip)" tabindex="0">
          <div class="flip-card-inner">
            <div class="side front"><div class="q">{front}</div><div class="hint"></div></div>
            <div class="side back"><div class="a">{back}</div><div class="hint"></div></div>
          </div>
        </div>
      </div>
    </div>
    <script>(function(){{
      const card=document.getElementById("{card_id}"), nextHost=document.getElementById("{next_host_id}"), scope=document.getElementById("flash-scope");
      if(!card) return;
      const reset=()=>card.classList.remove("flipped"); reset(); requestAnimationFrame(reset); setTimeout(reset,60); setTimeout(reset,300);
      let canToggle=false; setTimeout(()=>{{canToggle=true;}},120);
      const goNext=()=>{{ const b=nextHost&&nextHost.querySelector("button"); if(b) b.click(); }};
      const toggle=()=>{{ if(canToggle) card.classList.toggle("flipped"); }};
      card.addEventListener("click", toggle, {{passive:true}});
      card.addEventListener("keydown", e=>{{ const k=e.key||e.code;
        if(k===" "||k==="Space"||k==="Spacebar"){{e.preventDefault();toggle();}}
        else if((k==="Enter"||k==="ArrowRight")&&card.classList.contains("flipped")){{e.preventDefault();goNext();}}
        else if(k==="Escape"){{e.preventDefault();card.classList.remove("flipped");}} }});
      let sy=null,sx=null,stt=0,sw=false;
      if(scope){{
        scope.addEventListener("touchstart",e=>{{ if(!e.changedTouches?.length) return; const t=e.changedTouches[0]; sy=t.clientY;sx=t.clientX;stt=Date.now(); sw=False; }},{{passive:true}});
        scope.addEventListener("touchend",e=>{{ if(sy===null) return; const t=e.changedTouches[0]; const dy=sy-t.clientY; const dx=Math.abs(sx-t.clientX); const dt=Date.now()-stt;
          if(card.classList.contains("flipped") && dy>40 && dx<60 && dt<800){{ sw=true; e.preventDefault(); e.stopPropagation(); goNext(); }} sy=null; }},{{passive:false}});
        scope.addEventListener("click",e=>{{ if(sw){{ e.preventDefault(); e.stopPropagation(); sw=false; }} }}, true);
      }}
    }})()</script>
    """, unsafe_allow_html=True)

    st.markdown("<div id='flash-controls'>", unsafe_allow_html=True)
    l,m,r = st.columns(3)
    with l:
        if st.button("👍 I got it", key=f"fc_right_{idx}", use_container_width=True):
            st.session_state.flash["cards"][idx]["mark"]=True
            if idx<total-1: st.session_state.flash["idx"]=idx+1
            st.rerun()
    with m:
        if st.button("👎 Not yet", key=f"fc_wrong_{idx}", use_container_width=True):
            st.session_state.flash["cards"][idx]["mark"]=False
            if idx<total-1: st.session_state.flash["idx"]=idx+1
            st.rerun()
    with r:
        st.markdown(f"<div id='{next_host_id}'>", unsafe_allow_html=True)
        if st.button("Next card", key=f"fc_next_{idx}", use_container_width=True):
            if idx<total-1: st.session_state.flash["idx"]=idx+1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True); render_flash_dashboard(st.session_state.flash)

    rows=[{"index":i+1,"front":c["front"],"back":c["back"],"marked":("" if c.get("mark") is None else ("correct" if c["mark"] else "incorrect"))}
          for i,c in enumerate(st.session_state.flash["cards"])]
    if rows:
        buf=io.StringIO(); w=csv.DictWriter(buf, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        st.download_button("📥 Download Flashcards (CSV)", buf.getvalue().encode("utf-8"),
                           file_name=f"flashcards_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)
        meta={"topic":st.session_state.flash["topic"],"generated_at":st.session_state.flash["generated_at"],
              "total":len(st.session_state.flash["cards"]),
              "reviewed":sum(1 for c in st.session_state.flash["cards"] if c.get("mark") is not None),
              "correct":sum(1 for c in st.session_state.flash["cards"] if c.get("mark") is True)}
        with st.spinner("Preparing PDF…"):
            pdf_path=create_flash_pdf(st.session_state.flash["cards"], meta)
        with open(pdf_path,"rb") as f:
            st.download_button("📥 Download Flashcards (PDF)", f.read(), file_name=os.path.basename(pdf_path),
                               mime="application/pdf", use_container_width=True)

# ── THEME & STATE ─────────────────────────────────────────────────────────────
LIGHT={"bg":"#f8fafb","bar":"#fff","bot":"#e9eef6","user":"#22577a","text":"#191b22","input":"#e8edf2","border":"#d4dde7","expander":"#f4f7fb"}
DARK ={"bg":"#202126","bar":"#232733","bot":"#232733","user":"#22577a","text":"#f3f5f8","input":"#242730","border":"#26282f","expander":"#24272e"}

if "theme" not in st.session_state: st.session_state.theme="dark"
if "chat_history" not in st.session_state: st.session_state.chat_history=[]
if "session_id" not in st.session_state: st.session_state.session_id=None
if "active_doc_name" not in st.session_state: st.session_state.active_doc_name=None
if "voice_enabled" not in st.session_state: st.session_state.voice_enabled=False
if "input_accent" not in st.session_state: st.session_state.input_accent='en-US'
if "output_accent" not in st.session_state: st.session_state.output_accent='com'
if "teaching_mode" not in st.session_state: st.session_state.teaching_mode=False
if "mode" not in st.session_state: st.session_state.mode=None  # baseline chat when modes OFF

THEME = DARK if st.session_state.theme=="dark" else LIGHT

st.markdown(f"""
<style>
  .stApp {{ background:{THEME['bg']}; color:{THEME['text']}; }}
  .topbar-custom {{ background:{THEME['bar']}; border-radius:16px; padding:1.0em 1.1em; margin:.8em 0 1.0em;
                    box-shadow:0 2px 12px rgba(44,46,66,.06); font-size:1.3rem; font-weight:800; letter-spacing:.01em; }}
  .msg-user {{ background:{THEME['user']}; color:{THEME['text']}; border-radius:16px 16px 4px 20px; margin-bottom:.3em;
               padding:1em 1.35em; width:fit-content; max-width:85%; font-size:1.13rem; border:1.5px solid {THEME['border']};
               margin-left:auto; text-align:right; box-shadow:0 1px 12px rgba(55,96,148,.05); }}
  .msg-bot {{ background:{THEME['bot']}; color:{THEME['text']}; border-radius:16px 16px 20px 4px; margin-bottom:.7em;
              padding:1.08em 1.23em 1em 1.18em; width:fit-content; max-width:85%; font-size:1.13rem; border:1.5px solid {THEME['border']}; }}
  [data-testid="stExpander"] {{ border-color:{THEME['border']}; background:{THEME['expander']}; }}
  .stButton>button {{ width:100%; padding:.70em 1em; margin:.25rem 0 !important; border-radius:12px; }}
  .note-text {{ color:#787878; font-size:.9rem; }}
  .case-instr {{ margin:.6em 0 .3em; font-weight:600; }}
  @media (max-width:768px){{ .topbar-custom{{ font-size:1.1rem; padding:.9em; text-align:center; }}
    .msg-user,.msg-bot{{ font-size:.95rem; max-width:95%; }} }}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='topbar-custom'>Ophtha Bot : AI Chatbot for Postgrad Ophthalmology Students</div>", unsafe_allow_html=True)

# ── SIDEBAR: Upload → Modes → Voice Chat → Settings ──────────────────────────
with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")
    if uploaded_file and st.button("Process Document", key="process_pdf", use_container_width=True):
        with st.spinner("Processing document..."):
            session_id=str(uuid.uuid4()); temp_dir=os.path.join(TEMP_STORAGE_PATH, session_id); os.makedirs(temp_dir, exist_ok=True)
            path=os.path.join(temp_dir, uploaded_file.name)
            with open(path,"wb") as b: b.write(uploaded_file.getbuffer())
            doc=fitz.open(path)
            texts=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)\
                    .split_text("".join(page.get_text() for page in doc))
            doc.close()
            FAISS.from_texts(texts, embeddings).save_local(temp_dir)
            st.session_state.session_id=session_id; st.session_state.active_doc_name=uploaded_file.name
            st.success(f"Ready for questions about **{uploaded_file.name}**."); time.sleep(0.4); st.rerun()
    if st.session_state.active_doc_name:
        st.caption(f"Active Document: **{st.session_state['active_doc_name']}**")
        if st.button("Clear Document & Revert to Default", key="clear_doc_sidebar", use_container_width=True):
            st.session_state.session_id=None; st.session_state.active_doc_name=None; st.rerun()

    st.divider()
    st.header("Modes")
    allowed_modes=["Teaching","Exam","Case","Flashcards"]
    enable_modes = st.toggle("Enable study modes", value=st.session_state.mode in allowed_modes)
    if enable_modes:
        st.session_state.mode = st.radio("Choose mode", options=allowed_modes,
                                         index=allowed_modes.index(st.session_state.mode) if st.session_state.mode in allowed_modes else 0,
                                         key="mode_radio")
    else:
        st.session_state.mode = None
    st.session_state.teaching_mode = (st.session_state.mode=="Teaching")

    st.divider()
    st.header("Voice Chat")
    st.session_state.voice_enabled = st.toggle("Enable Voice Chat", value=st.session_state.voice_enabled)
    if st.session_state.voice_enabled and (st.session_state.mode in ["Teaching"] or st.session_state.mode is None):
        input_opts={'American (US)':'en-US','British (UK)':'en-GB','Indian':'en-IN','Australian':'en-AU','Canadian':'en-CA','South African':'en-ZA'}
        try: idx=list(input_opts.values()).index(st.session_state.input_accent)
        except ValueError: idx=0
        st.session_state.input_accent = input_opts[st.selectbox("Your Accent (for input)", list(input_opts.keys()), index=idx)]
        out_opts={'American (US)':'com','British (UK)':'co.uk','Indian':'co.in'}
        st.session_state.output_accent = out_opts[st.selectbox("Assistant's Accent (for output)", list(out_opts.keys()),
                                                               index=list(out_opts.values()).index(st.session_state.output_accent))]

    st.divider()
    st.header("Settings")
    dark_on = st.session_state.theme=="dark"
    t = st.toggle("Dark Mode", value=dark_on, key="theme_toggle")
    if t != dark_on: st.session_state.theme="dark" if t else "light"; st.rerun()

# ── CHAT (baseline or Teaching) history display ───────────────────────────────
if (st.session_state.mode in ["Teaching"]) or (st.session_state.mode is None):
    for entry in st.session_state.chat_history:
        if "user" in entry: st.markdown(f"<div class='msg-user'>{entry['user']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg-bot'>{entry['bot']}</div>", unsafe_allow_html=True)
            if entry.get("pdf_filename"):
                path=os.path.join(CHEATSHEET_PATH, entry["pdf_filename"])
                if os.path.exists(path):
                    with open(path,"rb") as f: st.download_button("📥 Download Cheatsheet", f.read(), entry["pdf_filename"], "application/pdf", key=f"dl_{entry['pdf_filename']}_{uuid.uuid4()}")

# ── MODE ROUTER ───────────────────────────────────────────────────────────────
if st.session_state.mode=="Exam":
    render_exam_ui()
elif st.session_state.mode=="Case":
    render_case_ui()
elif st.session_state.mode=="Flashcards":
    render_flash_ui()
else:
    user_prompt = speech_to_text(language=st.session_state.input_accent, use_container_width=True, just_once=True, key='STT') \
                  if st.session_state.voice_enabled else st.chat_input("Type your question here...")
    if user_prompt:
        st.markdown(f"<div class='msg-user'>{user_prompt}</div>", unsafe_allow_html=True)
        st.session_state.chat_history.append({"user":user_prompt})
        with st.spinner("Thinking..."):
            answer, pdf_filename = handle_query_logic(user_prompt, st.session_state.get("session_id"))
            clean = re.sub(r'<.*?>','',answer).replace('`','').replace('*','')
            bot_html = f"{answer}<br><span class='note-text'>{disclaimer_text}</span>"
            st.markdown(f"<div class='msg-bot'>{bot_html}</div>", unsafe_allow_html=True)
            if st.session_state.voice_enabled:
                audio = text_to_audio_b64(clean+" Anything you'd like to explore next?", st.session_state.output_accent)
                if audio: render_audio_player_b64(audio)
            if pdf_filename:
                p=os.path.join(CHEATSHEET_PATH, pdf_filename)
                if os.path.exists(p):
                    with open(p,"rb") as f: st.download_button("📥 Download Cheatsheet", f.read(), pdf_filename, "application/pdf", key=f"dl_{pdf_filename}_{uuid.uuid4()}")
            st.session_state.chat_history.append({"bot":bot_html, "pdf_filename":pdf_filename})
