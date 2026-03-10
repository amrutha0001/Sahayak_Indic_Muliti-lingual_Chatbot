# Sahayak_Indic_Muliti-lingual_Chatbot
Multilingual RAG chatbot for Indian government welfare schemes — ask in Hindi, Telugu, Tamil and 8 other languages, get accurate answers powered by a local LLM + ChromaDB vector search.


# 🇮🇳 Sahayak — India's Government Scheme Assistant

> **Multilingual RAG chatbot for Indian government welfare schemes.**  
> Ask about PM-KISAN, PMJDY, MUDRA, Ayushman Bharat and 15+ other schemes — in Hindi, Telugu, Tamil, Kannada, Bengali, and 7 more Indian languages. Get complete, accurate answers powered by a local LLM and ChromaDB vector search.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📸 Preview

The interface features a dark navy sidebar listing all loaded schemes, a clean cream chat area, and a saffron/tricolour palette inspired by the Indian flag.

---

## ✨ Features

- **11 Indian languages** — English, Hindi, Telugu, Tamil, Kannada, Bengali, Marathi, Gujarati, Punjabi, Odia, Malayalam
- **3-tier retrieval cascade** — ChromaDB vector search → official scheme website → DuckDuckGo web search, automatically escalating when local results are insufficient
- **Multi-format ingestion** — `.txt`, `.pdf`, `.docx`/`.doc`, `.md`, `.html`, `.rtf`
- **Accurate translation pipeline** — query translated to English for retrieval, full answer generated in English, then translated back to the user's language
- **Section-aware chunking** — documents parsed into semantic sections (Overview, Benefits, Eligibility, Documents Required, etc.) for precise retrieval
- **Source attribution** — every answer shows which document and section it came from, with similarity scores
- **Tier badges** — answers are labelled 🟢 From document / 🟡 From official website / 🔵 From web search

---

## 🏗️ Architecture

```
User Query (any language)
        │
        ▼
  Translation → English          (deep-translator)
        │
        ▼
  ChromaDB Retrieval              (sentence-transformers/all-MiniLM-L6-v2)
        │
   score ≥ 0.55?  ──No──▶  Tier 2: Scrape official URL from scheme file
        │                          │
       Yes                    Still thin? ──▶ Tier 3: DuckDuckGo search
        │                          │
        └──────────────────────────┘
                      │
                      ▼
              LLM Generation              (Meta-Llama-3-8B-Instruct via HF Router)
                      │
                      ▼
           Translate Answer → User Language
                      │
                      ▼
                   Response
```

---

## 📁 Project Structure

```
sahayak/
├── app.py              # FastAPI server — /chat, /schemes, /health endpoints
├── rag.py              # RAG pipeline — retrieval, 3-tier cascade, LLM, translation
├── ingest.py           # Document ingestion — multi-format parsing, ChromaDB embedding
├── requirements.txt    # Python dependencies
├── schema.txt          # Template for adding new scheme .txt files
├── env.example         # Environment variable template
├── schemes/            # Scheme documents (.txt, .pdf, .docx)
│   ├── pm_jan_dhan_yojana.txt
│   ├── pm_kisan_samman_nidhi.txt
│   ├── Janani_Suraksha_Yojana.pdf
│   ├── pm_vanbandhu_kalyan_yojana.docx
│   └── ...
└── static/
    └── index.html      # Single-page frontend UI
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- A [Hugging Face](https://huggingface.co/settings/tokens) account with API token
- Access to `meta-llama/Meta-Llama-3-8B-Instruct` (request access on HF if needed)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sahayak.git
cd sahayak
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `torch==2.3.0` is ~2 GB. First install will take a few minutes.

### 4. Configure environment

```bash
# Copy the example file
cp env.example .env      # macOS/Linux
copy env.example .env    # Windows
```

Edit `.env` and add your Hugging Face token:

```env
HF_TOKEN=hf_your_actual_token_here
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

### 5. Build the vector database

```bash
python ingest.py
```

This reads all files in `schemes/`, parses them into sections, embeds them with `all-MiniLM-L6-v2`, and saves to `chroma_db/`.

### 6. Start the server

```bash
uvicorn app:app --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 💬 Usage

1. Select your language from the dropdown in the top-right corner
2. Type your question in any supported language, or click a suggestion card
3. Sahayak retrieves relevant scheme information and answers in your language
4. Source documents and similarity scores are shown below each answer

**Example questions:**
- `Who is eligible for PMJDY?`
- `PMJDY కి ఎవరు అర్హులు?`
- `PM-KISAN के तहत कितनी राशि मिलती है?`
- `What documents are needed for a MUDRA loan?`

---

## 📄 Adding New Schemes

### From a .txt file
Follow the schema in `schema.txt`. The 7-section format works best:

```
Scheme Name
Ministry: ...
Department: ...
Launch Date: ...
Type: ...
--------------------------------------------------
1. Overview:
...
2. Benefits:
...
3. Eligibility:
...
4. Application Process:
...
5. Documents Required:
...
6. Frequently Asked Questions:
...
7. Website URL:
https://...
```

### From a PDF or Word document
Just drop the `.pdf` or `.docx` file into the `schemes/` folder. The parser handles:
- Numbered headings (`1. Benefits:`)
- Word `Heading 1/2/3` styles
- ALL CAPS headings
- Common single-word headings (`Benefits`, `Eligibility`, etc.)
- PDF font glyph cleanup (`(cid:127)` → `•`, font-encoded `₹`)

### Re-ingest after adding files

```bash
# Delete old database and rebuild
rm -rf chroma_db/          # macOS/Linux
rmdir /s /q chroma_db      # Windows

python ingest.py
```

---

## 🌐 Supported Languages

| Language | Code | Script |
|----------|------|--------|
| English  | en   | Latin  |
| Hindi    | hi   | Devanagari |
| Telugu   | te   | Telugu |
| Tamil    | ta   | Tamil  |
| Kannada  | kn   | Kannada |
| Bengali  | bn   | Bengali |
| Marathi  | mr   | Devanagari |
| Gujarati | gu   | Gujarati |
| Punjabi  | pa   | Gurmukhi |
| Odia     | or   | Odia   |
| Malayalam| ml   | Malayalam |

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API server |
| `chromadb` | Vector database for semantic search |
| `sentence-transformers` | Document and query embeddings |
| `torch` + `transformers` | LLM inference via Hugging Face |
| `deep-translator` | Query and answer translation |
| `pdfplumber` + `pdfminer.six` + `pypdfium2` | PDF text extraction (3 fallbacks) |
| `python-docx` | Word document extraction |
| `beautifulsoup4` | HTML/URL content parsing |

---

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token | required |
| `LLM_MODEL` | HF model ID | `meta-llama/Meta-Llama-3-8B-Instruct` |

Retrieval thresholds (in `rag.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `SCORE_CONFIDENT` | 0.55 | Use ChromaDB answer directly |
| `SCORE_TRY_WEB` | 0.30 | Try official URL first |
| `SCORE_HOPELESS` | 0.18 | Return "not found" |

---

## 🗂️ Currently Loaded Schemes

- Atal Pension Yojana (APY)
- Ayushman Bharat – PM Jan Arogya Yojana (PM-JAY)
- Beti Bachao Beti Padhao (BBBP)
- DAY – National Rural Livelihood Mission (DAY-NRLM)
- Jal Jeevan Mission (JJM)
- Janani Suraksha Yojana (JSY)
- National Digital Health Mission (NDHM/ABDM)
- PM Street Vendor's AtmaNirbhar Nidhi (PM SVANidhi)
- Pradhan Mantri Awas Yojana (PMAY)
- Pradhan Mantri Jan Dhan Yojana (PMJDY)
- Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)
- Pradhan Mantri Mudra Yojana (PMMY)
- Pradhan Mantri Vanbandhu Kalyan Yojana (PMVKY)
- Skill India Mission
- Stand-Up India
- Startup India
- Swachh Bharat Mission – Grameen (Phase I & II)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-scheme`)
3. Add your scheme file to `schemes/` following the schema
4. Run `python ingest.py` to verify it ingests correctly
5. Open a pull request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co) for the LLM inference API
- [ChromaDB](https://www.trychroma.com) for the vector database
- [Government of India](https://india.gov.in) for public scheme information
- Built as part of the SIC Projects initiative
