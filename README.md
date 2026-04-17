# Sahayak — India's Government Scheme Assistant

**Multilingual RAG chatbot for Indian government welfare schemes.**
Ask about PM-KISAN, PMJDY, MUDRA, Ayushman Bharat and 17 other schemes in Hindi, Telugu, Tamil, Kannada, Bengali, and 6 more Indian languages. Speak your question aloud and hear the answer read back to you. Powered by a local LLM and ChromaDB vector search.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Preview

The interface uses a dark navy sidebar listing all loaded schemes, a clean cream chat area, and a saffron and tricolour palette inspired by the Indian flag. A microphone button sits in the input bar for voice queries, and a voice output toggle in the header reads answers aloud.

---

## Features

- **11 Indian languages** — English, Hindi, Telugu, Tamil, Kannada, Bengali, Marathi, Gujarati, Punjabi, Odia, Malayalam
- **Voice input** — click the microphone button and speak your question in any supported language; the transcript is auto-sent when you stop speaking
- **Voice output** — toggle voice output in the header to have every bot response read aloud; individual messages have a Speak button for on-demand replay
- **Smart TTS fallback** — if the device has no voice pack installed for the selected language, the system automatically speaks the English version of the answer instead of producing garbled output
- **3-tier retrieval cascade** — ChromaDB vector search, then official scheme website, then DuckDuckGo web search, escalating automatically when local results are insufficient
- **Multi-format ingestion** — `.txt`, `.pdf`, `.docx`, `.doc`, `.md`, `.html`, `.rtf`
- **Accurate translation pipeline** — query translated to English for retrieval, answer generated in English, then translated to the user's language; the English version is always preserved as a fallback
- **Section-aware chunking** — documents parsed into semantic sections (Overview, Benefits, Eligibility, Documents Required, and so on) for precise retrieval
- **Source attribution** — every answer shows the source document and section with similarity scores
- **Tier badges** — answers are labelled From document, From official website, or From web search
- **Dark mode** — full dark theme toggle with preference saved across sessions
- **Acronym expansion** — 25 common scheme acronyms (PMJDY, PM-KISAN, MUDRA, etc.) are expanded before retrieval to improve matching

---

## Architecture

```
User Query (any language)  /  Voice Input (Web Speech API)
        |
        v
  Translation to English          (deep-translator, explicit lang code)
        |
        v
  Acronym Expansion               (25 scheme acronyms normalised)
        |
        v
  ChromaDB Retrieval              (all-MiniLM-L6-v2 embeddings, cosine similarity)
        |
   score >= 0.55?  --No-->  Tier 2: Scrape official URL from Section 7 of scheme file
        |                          |
       Yes                   Still thin (score < 0.30)? --> Tier 3: DuckDuckGo search
        |                          |
        +---------------------------+
                      |
                      v
              LLM Generation    (Meta-Llama-3-8B-Instruct via HF Router, temp=0.2)
                      |
                      v
           Post-processing      (deduplication, context label stripping)
                      |
                      v
           Translate to User Language   (batched, 2000 char chunks)
                      |
                      v
                   Response     (answer + english_answer + sources + tier)
                      |
                      v
           Voice Output (SpeechSynthesis API, Indic voice or English fallback)
```

---

## Project Structure

```
sahayak/
├── app.py              # FastAPI server -- /chat, /schemes, /health, /ingest/url
├── rag.py              # RAG pipeline -- 3-tier cascade, LLM, translation, acronym expansion
├── ingest.py           # Document ingestion -- multi-format parsing, ChromaDB embedding
├── translator.py       # Standalone translation utility
├── requirements.txt    # Python dependencies (pip install -r requirements.txt)
├── schema.txt          # 7-section template for adding new scheme .txt files
├── env.example         # Environment variable template
├── schemes/            # Scheme documents (.txt, .pdf, .docx)
│   ├── pm_jan_dhan_yojana.txt
│   ├── pm_kisan_samman_nidhi.txt
│   ├── Janani Suraksha Yojana.pdf
│   ├── pm_vanbandhu_kalyan_yojana.docx
│   └── ...             # 17 schemes total
└── static/
    └── index.html      # Single-page frontend (chat UI, voice input/output, dark mode)
```

---

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- A Hugging Face account with a read-access API token
- Model access approved for `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face
- A modern browser (Chrome or Edge recommended for full voice input support)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sahayak.git
cd sahayak
```

### 2. Create and activate a virtual environment

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

Note: `torch==2.3.0` is approximately 2 GB. The first install will take several minutes depending on connection speed.

### 4. Configure environment variables

```bash
# macOS / Linux
cp env.example .env

# Windows
copy env.example .env
```

Edit `.env` and fill in your values:

```env
HF_TOKEN=hf_your_actual_token_here
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

### 5. Build the vector database

```bash
python ingest.py
```

This reads all files in `schemes/`, parses them into sections, generates embeddings with `all-MiniLM-L6-v2`, and writes the vector index to `chroma_db/`.

### 6. Start the server

```bash
uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

---

## Usage

1. Select your language from the dropdown in the header
2. Type your question, click a suggestion card, or press the microphone button and speak
3. Sahayak retrieves relevant scheme information and answers in your chosen language
4. Source documents and similarity scores appear below each answer
5. Toggle the voice output button in the header to have answers read aloud automatically
6. Use the Speak button on any individual message to replay that answer at any time

### Example questions

```
Who is eligible for PMJDY?
PMJDY ki eligibility kya hai?
PMJDY కి ఎవరు అర్హులు?
PM-KISAN के तहत कितनी राशि मिलती है?
What documents are needed for a MUDRA loan?
SVANidhi scheme for street vendors details
```

---

## Voice Features

### Voice Input

Voice input uses the browser's Web Speech API (available in Chrome and Edge). When you click the microphone button:

- The button pulses orange and a "Listening" badge appears above the input bar
- Speak your question; the live transcript appears in the text box as you speak
- Recognition stops automatically when you pause; the question is sent immediately
- Click the button again at any time to stop recording early
- The recognition language is automatically set to match the selected UI language

**Browser support:** Chrome and Edge support voice input. Firefox does not implement the Web Speech API.

### Voice Output

Voice output uses the browser's SpeechSynthesis API (built into all modern browsers). By default it is turned off so the app works silently in public or shared spaces.

To enable, click the Voice Off button in the header. It will turn green and show Voice On. Your preference is saved and restored on the next visit.

**Indic voice packs:** For native-language speech, the device must have the relevant voice pack installed. Without one, the system automatically speaks the English version of the answer rather than producing garbled output. To install voice packs:

| Platform | Steps |
|----------|-------|
| Windows 11 | Settings > Time and Language > Speech > Add voices |
| Android | Text-to-speech output in Accessibility settings (Google TTS supports all languages natively) |
| macOS | System Settings > Accessibility > Spoken Content > System Voice |
| Chrome on Android | Works natively with Google TTS |

---

## Adding New Schemes

### From a plain text file

Follow the 7-section format in `schema.txt`:

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

Place the file in `schemes/` and re-run `python ingest.py`.

### From a PDF or Word document

Drop the `.pdf` or `.docx` file into `schemes/`. The parser handles numbered headings, Word Heading styles, ALL CAPS headings, and PDF font glyph cleanup.

### From a live URL

Use the API endpoint directly:

```bash
curl -X POST http://localhost:8000/ingest/url \
     -H "Content-Type: application/json" \
     -d '{"url": "https://pmjdy.gov.in/about"}'
```

### Rebuild the index after adding files

```bash
# Delete the existing database first
rm -rf chroma_db/          # macOS / Linux
rmdir /s /q chroma_db      # Windows

python ingest.py
```

---

## Supported Languages

| Language | Code | Script |
|----------|------|--------|
| English | en | Latin |
| Hindi | hi | Devanagari |
| Telugu | te | Telugu |
| Tamil | ta | Tamil |
| Kannada | kn | Kannada |
| Bengali | bn | Bengali |
| Marathi | mr | Devanagari |
| Gujarati | gu | Gujarati |
| Punjabi | pa | Gurmukhi |
| Odia | or | Odia |
| Malayalam | ml | Malayalam |

---

## Currently Loaded Schemes

| Scheme | Acronym |
|--------|---------|
| Atal Pension Yojana | APY |
| Ayushman Bharat PM Jan Arogya Yojana | PM-JAY |
| Beti Bachao Beti Padhao | BBBP |
| DAY National Rural Livelihood Mission | DAY-NRLM |
| Jal Jeevan Mission | JJM |
| Janani Suraksha Yojana | JSY |
| National Digital Health Mission | NDHM |
| PM Street Vendor's AtmaNirbhar Nidhi | PM SVANidhi |
| Pradhan Mantri Awas Yojana | PMAY |
| Pradhan Mantri Jan Dhan Yojana | PMJDY |
| Pradhan Mantri Kisan Samman Nidhi | PM-KISAN |
| Pradhan Mantri Mudra Yojana | PMMY |
| Pradhan Mantri Vanbandhu Kalyan Yojana | PMVKY |
| Skill India Mission | SIM |
| Stand-Up India | SUI |
| Startup India | SI |
| Swachh Bharat Mission Grameen Phase I | SBM-G I |
| Swachh Bharat Mission Grameen Phase II | SBM-G II |

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API server |
| `uvicorn` | ASGI server |
| `chromadb` | Vector database for semantic search |
| `sentence-transformers` | Document and query embeddings |
| `torch` + `transformers` | LLM inference via Hugging Face |
| `deep-translator` | Query and answer translation (Google Translate backend) |
| `pdfplumber` | PDF text extraction (primary) |
| `pdfminer.six` + `pypdfium2` | PDF extraction fallbacks |
| `python-docx` | Word document extraction |
| `beautifulsoup4` | HTML and URL content parsing |
| `requests` | HTTP client for Tier 2 and Tier 3 retrieval |

Voice input and voice output use the browser's built-in Web Speech API and SpeechSynthesis API respectively. No additional packages are required.

---

## Configuration

### Environment variables

| Variable | Description | Required |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token | Yes |
| `LLM_MODEL` | Hugging Face model ID | Yes (default: `meta-llama/Meta-Llama-3-8B-Instruct`) |

### Retrieval thresholds (rag.py)

| Constant | Value | Meaning |
|----------|-------|---------|
| `SCORE_CONFIDENT` | 0.55 | ChromaDB answer is sufficient; skip Tier 2 and 3 |
| `SCORE_TRY_WEB` | 0.30 | Below this, attempt Tier 3 web search after Tier 2 |
| `SCORE_HOPELESS` | 0.18 | Below this, return a not-found message immediately |

### LLM settings (rag.py)

| Parameter | Value |
|-----------|-------|
| `max_tokens` | 900 |
| `temperature` | 0.2 |
| `repetition_penalty` | 1.15 |
| `TOP_K` | 4 retrieved chunks |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the frontend (index.html) |
| GET | `/health` | Returns server status and total chunk count |
| GET | `/schemes` | Lists all scheme names loaded in ChromaDB |
| POST | `/chat` | Main chat endpoint; accepts `{question, language}` |
| POST | `/ingest/url` | Crawls a URL and adds its content to ChromaDB |

### Chat request and response

```json
POST /chat
{
  "question": "Who is eligible for PMJDY?",
  "language": "Telugu"
}

Response:
{
  "answer": "...(Telugu text)...",
  "english_answer": "...(English text, used as TTS fallback)...",
  "sources": [
    { "scheme": "Pradhan Mantri Jan Dhan Yojana", "section": "Eligibility", "score": 0.82 }
  ],
  "tier": "txt"
}
```

---

## Troubleshooting

**Voice input is not working.**
Voice input requires Chrome or Edge. Firefox does not support the Web Speech API. Also ensure microphone permission is granted in the browser.

**Voice output only reads numbers.**
This means the device has no voice pack installed for the selected language. The system will automatically fall back to English speech. Install the relevant voice pack from your device's language settings, or use the Speak button after switching to English.

**The server returns a 500 error on the first question.**
Ensure `python ingest.py` was run at least once and the `chroma_db/` directory exists. Also confirm the `.env` file has a valid `HF_TOKEN`.

**Translation is slow or times out.**
The `deep-translator` library calls the Google Translate API. On a slow connection or if the service is rate-limiting, translation may time out. The system will return the untranslated English answer in that case.

**A scheme is not found even though its file is in `schemes/`.**
Delete `chroma_db/` and re-run `python ingest.py` to rebuild the index from scratch.
