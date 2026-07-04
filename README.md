Ophtha Bot: AI-Powered Ophthalmology Tutor for Postgraduates

Ophtha Bot is a Retrieval-Augmented Generation (RAG) chatbot built to support postgraduate ophthalmology learners. It combines a domain-restricted knowledge base with Google's Gemini LLM to provide accurate, source-grounded explanations, exam practice, and revision tools — without hallucinating outside its ophthalmology scope.

Live Demo

App link: https://my-chatbot-project-uh7vkxnrnuneiswhm3n8wu.streamlit.app/

Made in UK as part of my MSc Dissertation in Aug 2025

If trying to run now :

⚠️ Known issue: The app currently loads, but query handling is broken due to a dependency version mismatch. The FAISS knowledge base index was pickled under an older pydantic/langchain_community version. Streamlit Cloud has since rebuilt the environment with newer package versions, and pydantic's internal object state format changed between versions — so the old pickled index can no longer be deserialized correctly. This produces a KeyError when a query is submitted. A rebuild of the knowledge base under the current dependency versions is in progress (see Rebuilding the Knowledge Base below).

Issue will be fixed asap.



GitLab (original dissertation submission): https://git.cs.bham.ac.uk/projects-2024-25/axr1167

Features


Chat mode — natural back-and-forth Q&A grounded in the knowledge base
Teaching mode — structured, didactic explanations of ophthalmology concepts
Exam mode — MCQ generation by topic/difficulty, with instant scoring and rationale
Case Simulation mode — scenario-based clinical vignettes with structured feedback
Flashcards mode — tap-to-reveal cards with progress tracking
Voice I/O — speech-to-text input and text-to-speech output
Export — CSV/PDF export for exam results, case feedback, and flashcard decks
Custom upload — users can upload their own PDFs for a session-specific knowledge base
Domain gate — rejects out-of-scope (non-ophthalmology) queries to reduce hallucination risk


Prerequisites


Python 3.11+ (see version pinning note below)
A Google API key with access to Gemini and Generative AI Embeddings



Architecture

User Input (text/voice)
        │
        ▼
   Domain Gate (ophthalmology / greeting check)
        │
        ▼
   FAISS Retriever (top-k=5 similarity search)
        │
        ▼
   ReAct Agent (selects 1 of 3 tools)
        ├── RetrievalQA        → direct Q&A
        ├── Concept Explainer  → structured teaching
        └── Cheat Sheet Gen    → Markdown → PDF
        │
        ▼
   Response (text + optional TTS audio)
