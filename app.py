import os
import json
import tempfile
import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field
from docx import Document
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv

# ==============================================================================
# SECTION 1: CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================
load_dotenv()

st.set_page_config(
    page_title="NyayaSathi | General Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

resolved_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not resolved_api_key:
    st.error("🔑 API Key Missing! Please set `GEMINI_API_KEY` in your `.env` file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=resolved_api_key)
model_pro = genai.GenerativeModel("gemini-3.8-flash")

# Supported Legal Domains for Everyday Indian Citizens
LEGAL_DOMAINS = [
    "Tenant & Landlord Disputes",
    "Unpaid Salary & Workplace Issues",
    "Consumer Refunds & E-Commerce Disputes",
    "Defective Goods & Breach of Warranty",
    "Traffic Fines & Motor Vehicle Issues",
    "Cyber Fraud & Online Scams",
    "Cheque Bounce & Debt Recovery (Sec 138 NI Act)",
    "Neighbour Disputes & Public Nuisance",
    "General Contract & Online Subscription Cancellation"
]

# Comprehensive Multi-Act RAG Knowledge Base (Anti-Hallucination Grounding)
EXPANDED_LEGAL_KNOWLEDGE_BASE = [
    {
        "id": "mta_2021_11",
        "domain": "Tenant & Landlord Disputes",
        "act": "Model Tenancy Act, 2021",
        "section": "Section 11",
        "text": "The security deposit for residential premises shall not exceed two months' rent. The deposit must be refunded by the landlord on taking over vacant possession after due deduction for actual, verified damages."
    },
    {
        "id": "tpa_1882_108",
        "domain": "Tenant & Landlord Disputes",
        "act": "Transfer of Property Act, 1882",
        "section": "Section 108",
        "text": "Lessor is bound to disclose material defects in property. Lessee bound to keep property in good condition, reasonable wear and tear excepted."
    },
    {
        "id": "pwa_1936_5",
        "domain": "Unpaid Salary & Workplace Issues",
        "act": "Payment of Wages Act, 1936",
        "section": "Section 5 & 7",
        "text": "Wages must be paid within 7 to 10 days of the wage period. Deductions are strictly limited to authorized items such as absence or damage caused by neglect, up to prescribed statutory caps."
    },
    {
        "id": "ida_1947_25f",
        "domain": "Unpaid Salary & Workplace Issues",
        "act": "Industrial Disputes Act, 1947",
        "section": "Section 25F",
        "text": "No workman employed in any industry who has been in continuous service for not less than one year shall be retrenched until given one month notice in writing or paid wages in lieu of notice."
    },
    {
        "id": "cpa_2019_2_11",
        "domain": "Consumer Refunds & E-Commerce Disputes",
        "act": "Consumer Protection Act, 2019",
        "section": "Section 2(11) & Section 2(47)",
        "text": "'Deficiency' means any fault, imperfection, shortcoming or inadequacy in quality, nature and manner of performance. 'Unfair trade practice' includes refusing to take back defective goods or refund money paid within specified timelines."
    },
    {
        "id": "cpa_ecomm_2020",
        "domain": "Consumer Refunds & E-Commerce Disputes",
        "act": "Consumer Protection (E-Commerce) Rules, 2020",
        "section": "Rule 6",
        "text": "E-commerce entities cannot impose cancellation charges unless similar charges are borne by the platform, and must process refunds within a reasonable timeframe as per CCPA guidelines."
    },
    {
        "id": "nia_1881_138",
        "domain": "Cheque Bounce & Debt Recovery (Sec 138 NI Act)",
        "act": "Negotiable Instruments Act, 1881",
        "section": "Section 138",
        "text": "Where any cheque drawn for discharge of any debt or liability is returned unpaid by bank due to insufficiency of funds, it is a criminal offence punishable with imprisonment up to 2 years or fine up to double the cheque amount. Demand notice must be served within 30 days of dishonour memo."
    },
    {
        "id": "it_act_66d",
        "domain": "Cyber Fraud & Online Scams",
        "act": "Information Technology Act, 2000",
        "section": "Section 66D",
        "text": "Punishment for cheating by personation by using computer resource: Imprisonment up to 3 years and fine up to 1 lakh rupees. Victims of financial cyber fraud must report immediately to 1930 Cyber Helpline or cybercrime.gov.in within the 'Golden Hour'."
    },
    {
        "id": "bns_2023_270",
        "domain": "Neighbour Disputes & Public Nuisance",
        "act": "Bharatiya Nyaya Sanhita (BNS), 2023 / IPC",
        "section": "Section 270 BNS (Public Nuisance)",
        "text": "A person is guilty of a public nuisance who does any act or is guilty of an illegal omission which causes any common injury, danger or annoyance to the public or people in the vicinity."
    },
    {
        "id": "mva_2019_191",
        "domain": "Traffic Fines & Motor Vehicle Issues",
        "act": "Motor Vehicles (Amendment) Act, 2019",
        "section": "Section 191 & Virtual Courts Rule",
        "text": "Wrongly issued e-challans can be contested online via Virtual Courts portal (vcourts.gov.in) or challenged before the concerned Senior Traffic Police Officer before pay deadline."
    }
]


# ==============================================================================
# SECTION 2: RAG ENGINE (RETRIEVAL LOGIC)
# ==============================================================================
def retrieve_laws_by_domain(query: str, selected_domain: str = None) -> str:
    query_lower = query.lower()
    matched_chunks = []

    if selected_domain and selected_domain != "Auto-Detect Domain":
        for chunk in EXPANDED_LEGAL_KNOWLEDGE_BASE:
            if chunk["domain"] == selected_domain:
                matched_chunks.append(f"[{chunk['act']} - {chunk['section']}]: {chunk['text']}")

    keywords_map = {
        "deposit": ["Tenant & Landlord Disputes"], "rent": ["Tenant & Landlord Disputes"],
        "landlord": ["Tenant & Landlord Disputes"],
        "salary": ["Unpaid Salary & Workplace Issues"], "pay": ["Unpaid Salary & Workplace Issues"],
        "notice period": ["Unpaid Salary & Workplace Issues"],
        "refund": ["Consumer Refunds & E-Commerce Disputes"], "e-commerce": ["Consumer Refunds & E-Commerce Disputes"],
        "defective": ["Defective Goods & Breach of Warranty"], "warranty": ["Defective Goods & Breach of Warranty"],
        "cheque": ["Cheque Bounce & Debt Recovery (Sec 138 NI Act)"],
        "bounce": ["Cheque Bounce & Debt Recovery (Sec 138 NI Act)"],
        "cyber": ["Cyber Fraud & Online Scams"], "fraud": ["Cyber Fraud & Online Scams"],
        "otp": ["Cyber Fraud & Online Scams"],
        "challan": ["Traffic Fines & Motor Vehicle Issues"], "fine": ["Traffic Fines & Motor Vehicle Issues"],
        "noise": ["Neighbour Disputes & Public Nuisance"], "nuisance": ["Neighbour Disputes & Public Nuisance"]
    }

    for kw, domains in keywords_map.items():
        if kw in query_lower:
            for chunk in EXPANDED_LEGAL_KNOWLEDGE_BASE:
                if chunk["domain"] in domains:
                    formatted_chunk = f"[{chunk['act']} - {chunk['section']}]: {chunk['text']}"
                    if formatted_chunk not in matched_chunks:
                        matched_chunks.append(formatted_chunk)

    if not matched_chunks:
        for chunk in EXPANDED_LEGAL_KNOWLEDGE_BASE[:5]:
            matched_chunks.append(f"[{chunk['act']} - {chunk['section']}]: {chunk['text']}")

    return "\n\n".join(matched_chunks)


# ==============================================================================
# SECTION 3: PYDANTIC SCHEMAS
# ==============================================================================
class ChronologyEvent(BaseModel):
    date_or_time: str = Field(description="Date, time, or period when event occurred")
    event_description: str = Field(description="Factual summary of event")
    available_proof: str = Field(description="Evidence (e.g., invoices, chats, screenshots, receipts)")


class UniversalCaseMatrix(BaseModel):
    complainant_name: str = Field(description="Name/identity of aggrieved user")
    opposing_party_name: str = Field(description="Opposing party, entity, platform, or individual")
    identified_domain: str = Field(description="Identified legal category from user context")
    summary: str = Field(description="Concise 2-sentence summary of core dispute")
    financial_claim_inr: float  # Required float without default value to avoid schema errors
    chronology: list[ChronologyEvent]
    key_missing_evidence: list[str] = Field(description="List of key proof items user should gather")


# ==============================================================================
# SECTION 4: STREAMLIT UI & APPLICATION MODULES
# ==============================================================================
st.title("⚖️ NyayaSathi: General Citizen Legal Assistant")
st.caption("AI-powered legal guidance for everyday Indian disputes — Voice-enabled & RAG-grounded.")

# Session State Initialization
if "universal_matrix" not in st.session_state:
    st.session_state.universal_matrix = None
if "nav_tab" not in st.session_state:
    st.session_state.nav_tab = "1. Universal Case Builder (Voice/Text)"

with st.sidebar:
    st.header("📌 Legal Domain Selector")
    selected_domain = st.selectbox(
        "Choose Dispute Category:",
        ["Auto-Detect Domain"] + LEGAL_DOMAINS
    )

    st.markdown("---")
    st.header("📎 Insert Evidence / Notice")
    uploaded_sidebar_file = st.file_uploader("Upload agreement, notice, or bill:", type=["jpg", "png", "jpeg", "pdf"])

    if uploaded_sidebar_file:
        st.success("File attached for analysis!")

    st.markdown("---")
    st.header("⚙️ Navigation")

    # Radio navigation bound to session state
    app_mode = st.radio(
        "Go to Module:",
        [
            "1. Universal Case Builder (Voice/Text)",
            "2. Legal Options & Notice Generator",
            "3. Document Audit & Contract Scanner"
        ],
        index=0 if st.session_state.nav_tab == "1. Universal Case Builder (Voice/Text)" else (
            1 if st.session_state.nav_tab == "2. Legal Options & Notice Generator" else 2)
    )

    st.markdown("---")
    st.warning(
        "⚠️ **Legal Disclaimer**: NyayaSathi provides educational legal information under Indian law. It is not court representation or formal legal advice.")

# Sync radio with session state updates
st.session_state.nav_tab = app_mode

# ------------------------------------------------------------------------------
# MODULE 1: UNIVERSAL CASE BUILDER (VOICE + TEXT)
# ------------------------------------------------------------------------------
if app_mode == "1. Universal Case Builder (Voice/Text)":
    st.header("🎙️ Step 1: Explain Your Problem (Text or Voice)")
    st.write(
        "NyayaSathi handles everyday issues: online order non-refunds, tenancy disputes, salary delays, cheque bounce, traffic fines, cyber fraud, and contract issues.")

    col1, col2 = st.columns([1, 2])
    transcribed_text = ""

    with col1:
        st.markdown("### 🔴 Voice Input")
        audio_info = mic_recorder(start_prompt="Record Audio", stop_prompt="Stop Recording", key='voice_recorder')
        if audio_info:
            st.audio(audio_info['bytes'])
            with st.spinner("Transcribing audio using Gemini..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_info['bytes'])
                        tmp_path = tmp.name
                    audio_ref = genai.upload_file(tmp_path, mime_type="audio/wav")
                    resp = model_pro.generate_content([
                        "Accurately transcribe this audio snippet of an Indian citizen describing their legal problem. Translate Hindi/Hinglish or any other indian language to English if needed:",
                        audio_ref
                    ])
                    transcribed_text = resp.text
                    st.success("Transcribed!")
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Voice error: {str(e)}")

    with col2:
        st.markdown("### ⌨️ Input / Edit Case Description")
        user_story = st.text_area(
            "Describe what happened:",
            value=transcribed_text,
            height=200,
            placeholder="e.g. Flipkart did not refund ₹2,499 for a canceled earphone order..."
        )

    if st.button("🚀 Process & Generate Case Matrix", type="primary"):
        if not user_story.strip():
            st.warning("Please record audio or type your story.")
        else:
            with st.spinner("Retrieving legal contexts & building structured case matrix..."):
                rag_laws = retrieve_laws_by_domain(user_story, selected_domain)

                prompt = f"""
                You are NyayaSathi's Legal Case Builder. Organize this citizen's dispute.

                RAG STATUTORY CONTEXT:
                {rag_laws}

                USER DISPUTE ACCOUNT:
                "{user_story}"

                Construct a full UniversalCaseMatrix output matching the schema.
                """
                try:
                    res = model_pro.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=UniversalCaseMatrix,
                            temperature=0.1
                        )
                    )
                    st.session_state.universal_matrix = json.loads(res.text)
                    st.success("Case Matrix created successfully!")
                except Exception as e:
                    st.error(f"Extraction error: {str(e)}")

    if st.session_state.universal_matrix:
        matrix = st.session_state.universal_matrix
        st.markdown("---")
        st.subheader("📊 Structured Case Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Complainant", matrix.get("complainant_name", "N/A"))
        c2.metric("Opposing Party", matrix.get("opposing_party_name", "N/A"))
        c3.metric("Category", matrix.get("identified_domain", "N/A"))
        c4.metric("Claim Amount", f"₹ {matrix.get('financial_claim_inr', 0):,.2f}")

        st.info(f"**Case Summary:** {matrix.get('summary', '')}")

        st.markdown("### 📅 Fact Chronology")
        if matrix.get("chronology"):
            st.table(matrix.get("chronology"))

        st.markdown("### 🔍 Evidence Checklist")
        for item in matrix.get("key_missing_evidence", []):
            st.write(f"- 📋 {item}")

        st.markdown("---")
        # 1. Proceed to Legal Advice Button at the end of Module 1
        if st.button("👉 Proceed to Legal Advice", type="primary", use_container_width=True):
            st.session_state.nav_tab = "2. Legal Options & Notice Generator"
            st.rerun()

# ------------------------------------------------------------------------------
# MODULE 2: RAG ADVICE & NOTICE GENERATOR
# ------------------------------------------------------------------------------
elif app_mode == "2. Legal Options & Notice Generator":
    st.header("📄 Step 2: Plain Language Advice & Notice Generator")

    tab1, tab2 = st.tabs(["💡 Legal Advice", "📜 Draft Formal Legal Notice / Complaint"])

    with tab1:
        st.subheader("Legal Position & Escalation Roadmap")
        if st.button("Generate Guidance & Legal Options", type="primary"):
            matrix_str = json.dumps(st.session_state.universal_matrix or {})
            rag_context = retrieve_laws_by_domain(matrix_str, selected_domain)

            prompt = f"""
            You are NyayaSathi. Provide comprehensive legal advice under Indian Law.

            STRICT RAG CONTEXT (Do not invent section numbers outside verified context):
            {rag_context}

            CASE DATA:
            {matrix_str}

            Deliver:
            1. Plain Language Legal Position (Where the citizen stands).
            2. Available Legal Options & Escalation Pathway (Customer Support -> Consumer Commission / Police Cyber Cell / Civil Court / Legal Notice).
            3. Estimated timeline, costs, and key risks.
            4. Explicit trigger conditions when a licensed Advocate is strictly necessary.
            """
            with st.spinner("Analyzing Indian statutory provisions..."):
                resp_advice = model_pro.generate_content(prompt)
                st.session_state.cached_advice = resp_advice.text
                st.markdown(resp_advice.text)
        elif "cached_advice" in st.session_state:
            st.markdown(st.session_state.cached_advice)

        st.markdown("---")
        # 1. Button at the end of Legal Advice to jump to Legal Notice Template
        if st.button("📝 Generate Legal Notice Template", type="primary", use_container_width=True):
            # Switch view to tab 2 or handle via state if needed (Streamlit tabs can't be set programmatically directly, so we instruct the user or render notice creator)
            st.success(
                "Scroll to the **'Draft Formal Legal Notice / Complaint'** tab above to customize and download your notice!")

    with tab2:
        st.subheader("Draft Pre-Litigation Notice or Complaint")
        col1, col2 = st.columns(2)
        sender = col1.text_input("Sender / Complainant Name", st.session_state.universal_matrix.get("complainant_name",
                                                                                                    "") if st.session_state.universal_matrix else "")
        recipient = col2.text_input("Recipient / Opposing Party Name",
                                    st.session_state.universal_matrix.get("opposing_party_name",
                                                                          "") if st.session_state.universal_matrix else "")
        doc_type = st.selectbox("Document Type:", ["Formal Legal Demand Notice", "Consumer Forum Complaint Draft",
                                                   "Cyber Cell Grievance Letter"])

        if st.button("📝 Draft Legal Document"):
            matrix_str = json.dumps(st.session_state.universal_matrix or {})
            rag_context = retrieve_laws_by_domain(matrix_str, selected_domain)

            notice_prompt = f"""
            Draft a high-quality {doc_type} under Indian law.
            Sender: {sender}
            Recipient: {recipient}
            RAG Grounding Laws: {rag_context}
            Case Details: {matrix_str}

            Follow standard legal drafting structure under Indian procedure.
            """
            with st.spinner("Drafting document..."):
                resp = model_pro.generate_content(notice_prompt)
                doc_text = resp.text
                st.text_area("Document Preview:", value=doc_text, height=350)

                doc = Document()
                for p in doc_text.split("\n\n"):
                    if p.strip(): doc.add_paragraph(p.strip())
                buf = BytesIO()
                doc.save(buf)
                buf.seek(0)

                st.download_button("📥 Download Document (.docx)", data=buf, file_name="Legal_Document.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ------------------------------------------------------------------------------
# MODULE 3: DOCUMENT AUDIT & CONTRACT SCANNER
# ------------------------------------------------------------------------------
elif app_mode == "3. Document Audit & Contract Scanner":
    st.header("🔎 Document & Notice Audit Scanner")
    st.write(
        "Inspect agreements, notices, bills, or e-challans uploaded via the sidebar to extract key statutory clauses and potential risks.")

    # Check if user uploaded a file in the sidebar
    if uploaded_sidebar_file is not None:
        st.image(uploaded_sidebar_file, caption="Attached Evidence / Document", use_column_width=True)

        if st.button("🔬 Audit Uploaded Document with Vision AI", type="primary"):
            with st.spinner("Analyzing document structure and legal terms..."):
                try:
                    img_bytes = uploaded_sidebar_file.getvalue()
                    mime_type = uploaded_sidebar_file.type if uploaded_sidebar_file.type else "image/jpeg"

                    vision_prompt = """
                    Analyze this Indian legal document/receipt/notice:
                    1. Extract key operational details (Dates, Statutory Sections cited, Monetary Penalties, Cure Periods).
                    2. Highlight any hidden or oppressive clauses (e.g., unilateral cancellation, excessive forfeiture).
                    3. Provide next recommended action steps under Indian law.
                    """
                    response = model_pro.generate_content([
                        vision_prompt,
                        {"mime_type": mime_type, "data": img_bytes}
                    ])
                    st.markdown("### Document Inspection Report")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Vision analysis error: {str(e)}")
    else:
        st.info(
            "💡 **Tip:** Use the **'Insert File'** uploader in the sidebar to attach an image or PDF of your agreement, notice, or bill for audit.")