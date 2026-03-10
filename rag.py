"""
rag.py  —  3-Tier RAG Cascade
══════════════════════════════════════════════════════════════════════════════
Tier 1 │ ChromaDB (indexed .txt / .pdf / .docx files)
       │   ↓  if answer thin / score low / LLM says "no info"
Tier 2 │ Official URL from Section 7 of the scheme's .txt file
       │   ↓  if still thin
Tier 3 │ Live DuckDuckGo web search  (last resort)

Translation pipeline (all tiers):
  user query → English → retrieve/fetch → English answer → local language
══════════════════════════════════════════════════════════════════════════════
"""

import os, re, textwrap
import chromadb
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

CHROMA_DIR    = "./chroma_db"
SCHEMES_DIR   = "./schemes"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K         = 4
LLM_MODEL     = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# Cascade thresholds
SCORE_CONFIDENT = 0.55   # Tier 1 alone is sufficient above this
SCORE_TRY_WEB   = 0.30   # below this, also try Tier 3
SCORE_HOPELESS  = 0.18   # below this, don't even attempt

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file!")

sentence_ef   = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection    = chroma_client.get_collection(name="schemes", embedding_function=sentence_ef)
print(f"✅ HF ready — {LLM_MODEL}")
print(f"📚 ChromaDB — {collection.count()} chunks")

# ── Acronym expansion ─────────────────────────────────────────────────────────
SCHEME_ACRONYMS = {
    r"\bPMJDY\b":       "Pradhan Mantri Jan Dhan Yojana",
    r"\bPMJAY\b":       "Pradhan Mantri Jan Arogya Yojana Ayushman Bharat",
    r"\bPM-JAY\b":      "Pradhan Mantri Jan Arogya Yojana Ayushman Bharat",
    r"\bPMAY\b":        "Pradhan Mantri Awas Yojana",
    r"\bPMKISAN\b":     "Pradhan Mantri Kisan Samman Nidhi",
    r"\bPM-KISAN\b":    "Pradhan Mantri Kisan Samman Nidhi",
    r"\bPMMY\b":        "Pradhan Mantri Mudra Yojana",
    r"\bMUDRA\b":       "Pradhan Mantri Mudra Yojana",
    r"\bAPY\b":         "Atal Pension Yojana",
    r"\bBBBP\b":        "Beti Bachao Beti Padhao",
    r"\bJJM\b":         "Jal Jeevan Mission",
    r"\bNDHM\b":        "National Digital Health Mission",
    r"\bABDM\b":        "Ayushman Bharat Digital Mission",
    r"\bDAY-NRLM\b":    "Deendayal Antyodaya Yojana National Rural Livelihood Mission",
    r"\bNRLM\b":        "National Rural Livelihood Mission",
    r"\bSBM\b":         "Swachh Bharat Mission",
    r"\bSVANidhi\b":    "PM Street Vendor AtmaNirbhar Nidhi SVANidhi",
    r"\bPMSVANidhi\b":  "PM Street Vendor AtmaNirbhar Nidhi SVANidhi",
    r"\bPMSY\b":        "Pradhan Mantri Suraksha Bima Yojana",
    r"\bPMJJBY\b":      "Pradhan Mantri Jeevan Jyoti Bima Yojana",
    r"\bPMGSY\b":       "Pradhan Mantri Gram Sadak Yojana",
    r"\bPMGKY\b":       "Pradhan Mantri Garib Kalyan Yojana",
    r"\bSTARTUP\s+INDIA\b":  "Startup India Scheme",
    r"\bSTAND.?UP\s+INDIA\b":"Stand-Up India Scheme",
    r"\bSKILL\s+INDIA\b":    "Skill India Mission",
}

def expand_acronyms(text):
    for pat, exp in SCHEME_ACRONYMS.items():
        text = re.sub(pat, exp, text, flags=re.IGNORECASE)
    return text

# ── Language helpers ──────────────────────────────────────────────────────────
LANG_CODES = {
    "hindi":"hi","marathi":"mr","gujarati":"gu","bengali":"bn",
    "tamil":"ta","telugu":"te","kannada":"kn","punjabi":"pa",
    "odia":"or","malayalam":"ml","english":"en",
}

def get_lang_code(language):
    return LANG_CODES.get(language.lower(), "en")

def translate_text(text, source, target):
    if source == target or not text.strip():
        return text
    src = "auto" if source in ("auto","detect") else source
    sentences = re.split(r'(?<=[।.!?\n])\s+', text)
    batches, cur = [], ""
    for s in sentences:
        if len(cur)+len(s)+1 <= 2000:
            cur += (" " if cur else "") + s
        else:
            if cur: batches.append(cur)
            if len(s) > 2000:
                batches.extend(textwrap.wrap(s, 2000))
            else:
                cur = s
            if len(s) <= 2000:
                continue
            cur = ""
    if cur: batches.append(cur)

    def _t(chunk, sc):
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=sc, target=target).translate(chunk) or chunk

    out = []
    for b in batches:
        try: out.append(_t(b, src))
        except Exception as e1:
            try: out.append(_t(b, "auto")); print(f"⚠️ Retried auto: {e1}")
            except Exception as e2: print(f"⚠️ Translation failed: {e2}"); out.append(b)
    return " ".join(out)

def to_english(query, language):
    """
    Translate query to English for retrieval.
    Uses the explicit source language code (not "auto") so that
    mixed-script queries like "PMJDY కి ఎవరు అర్హులు?" are detected
    correctly as Telugu, not mislabelled as Hindi by auto-detect.
    Falls back to auto if the explicit code fails.
    """
    expanded = expand_acronyms(query)
    src_code = get_lang_code(language)
    if src_code == "en":
        return expanded
    # Use explicit lang code — prevents Hindi/Telugu confusion
    result = translate_text(expanded, src_code, "en")
    # If translation returned the original unchanged (likely a failure),
    # try auto-detect as fallback
    if result.strip() == expanded.strip():
        result = translate_text(expanded, "auto", "en")
    result = expand_acronyms(result)
    print(f"🔤 [{language}/{src_code}→EN]: {result}")
    return result

def to_local(text, language):
    code = get_lang_code(language)
    if code == "en": return text
    return translate_text(text, "en", code)

# ── Tier 1: ChromaDB retrieval ────────────────────────────────────────────────
SECTION_KEYWORDS = {
    "eligib":    ["eligibility","who_can","criteria"],
    "benefit":   ["benefits","features","overview"],
    "apply":     ["application_process","how_to_apply","steps"],
    "document":  ["documents_required","documents","kyc"],
    "faq":       ["frequently_asked_questions","faqs"],
    "overdraft": ["benefits","overdraft"],
    "insurance": ["benefits","rupay"],
    "interest":  ["benefits","interest"],
    "pension":   ["benefits","overview"],
    "loan":      ["benefits","overview","eligibility"],
    "amount":    ["benefits","overview"],
    "limit":     ["benefits","eligibility"],
}

def retrieve(query, top_k=TOP_K):
    results = collection.query(
        query_texts=[query], n_results=top_k+4,
        include=["documents","metadatas","distances"]
    )
    chunks, seen = [], set()
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        cid = f"{meta.get('source_file','')}::{meta.get('section','')}"
        if cid in seen: continue
        seen.add(cid)
        chunks.append({
            "text": doc,
            "scheme_name": meta.get("scheme_name","Unknown"),
            "section": meta.get("section","").replace("_"," ").title(),
            "source_file": meta.get("source_file",""),
            "score": round(1-dist, 3),
        })
    # Section boost
    ql = query.lower()
    for kw, secs in SECTION_KEYWORDS.items():
        if kw in ql:
            top_scheme = chunks[0]["scheme_name"] if chunks else None
            if top_scheme:
                try:
                    b = collection.query(
                        query_texts=[query], n_results=2,
                        where={"$and":[{"scheme_name":{"$eq":top_scheme}},{"section":{"$in":secs}}]},
                        include=["documents","metadatas","distances"]
                    )
                    for doc, meta, dist in zip(b["documents"][0], b["metadatas"][0], b["distances"][0]):
                        cid = f"{meta.get('source_file','')}::{meta.get('section','')}"
                        if cid not in seen:
                            seen.add(cid)
                            chunks.insert(0,{
                                "text": doc, "scheme_name": meta.get("scheme_name","Unknown"),
                                "section": meta.get("section","").replace("_"," ").title(),
                                "source_file": meta.get("source_file",""),
                                "score": round(1-dist+0.05, 3),
                            })
                except Exception: pass
            break
    return chunks[:top_k]

# ── Tier 2: Official URL from Section 7 ──────────────────────────────────────
_url_cache = {}   # scheme_name → (url, text)

_SEC7_RE = [
    re.compile(r'7\.\s+Web(?:site|iste)[:\s\w]*\n\s*(https?://[^\s\r\n]+)', re.IGNORECASE),
    re.compile(r'Web(?:site)?\s+URL[:\s]+(https?://[^\s\r\n]+)', re.IGNORECASE),
]
_ANY_URL = re.compile(r'https?://[^\s\r\n]+')

def _find_scheme_url(scheme_name):
    """
    Scan schemes/ for the .txt file matching scheme_name.
    Try Section 7 regex first, then fall back to the last URL in the file.
    """
    if not os.path.isdir(SCHEMES_DIR): return None
    sn = scheme_name.lower()
    for fn in sorted(os.listdir(SCHEMES_DIR)):
        if not fn.lower().endswith(".txt"): continue
        try:
            content = open(os.path.join(SCHEMES_DIR, fn), encoding="utf-8", errors="replace").read()
        except Exception: continue
        # Match by first line (scheme title) or filename
        first = next((l.strip().lower() for l in content.splitlines() if l.strip()), "")
        fn_clean = fn.lower().replace("_"," ").replace(".txt","")
        if sn not in first and first not in sn and sn not in fn_clean and fn_clean not in sn:
            continue
        # Try section-7 patterns
        for pat in _SEC7_RE:
            m = pat.search(content)
            if m: return m.group(1).strip()
        # Fallback: last URL in file (almost always section 7)
        urls = _ANY_URL.findall(content)
        if urls: return urls[-1].strip()
    return None

def _scrape(url):
    try:
        from bs4 import BeautifulSoup
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (IndicGovAdvisor/2.0)"}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","form","noscript"]): tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main",re.I)) or soup.body
        raw  = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        lines = [l.strip() for l in raw.splitlines() if l.strip() and len(l.strip())>5]
        return "\n".join(lines[:400])
    except Exception as e:
        print(f"⚠️  Scrape failed ({url}): {e}"); return ""

def tier2_fetch(scheme_name):
    """Returns (url, text). Cached per session."""
    if scheme_name in _url_cache: return _url_cache[scheme_name]
    url = _find_scheme_url(scheme_name)
    if not url:
        print(f"ℹ️  No URL found for: {scheme_name}")
        _url_cache[scheme_name] = ("",""); return ("","")
    print(f"🌐 Tier 2: {url}")
    text = _scrape(url)
    _url_cache[scheme_name] = (url, text)
    return (url, text)

# ── Tier 3: DuckDuckGo web search ────────────────────────────────────────────
def tier3_search(query, n=5):
    results = []
    q = f"{query} India government scheme"
    # DDG Instant Answer API
    try:
        r = requests.get(f"https://api.duckduckgo.com/?q={quote_plus(q)}&format=json&no_html=1&skip_disambig=1",
                         headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        d = r.json()
        if d.get("AbstractText"):
            results.append({"title":d.get("Heading",q),"snippet":d["AbstractText"],"url":d.get("AbstractURL","")})
        for t in d.get("RelatedTopics",[])[:n]:
            if isinstance(t,dict) and t.get("Text"):
                results.append({"title":t.get("Name",""),"snippet":t["Text"],"url":t.get("FirstURL","")})
    except Exception as e: print(f"⚠️  DDG API: {e}")
    # DDG HTML fallback
    if not results:
        try:
            from bs4 import BeautifulSoup
            r = requests.get(f"https://html.duckduckgo.com/html/?q={quote_plus(q)}",
                             headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
            soup = BeautifulSoup(r.text,"html.parser")
            for res in soup.select(".result__body")[:n]:
                snip = res.select_one(".result__snippet")
                if snip:
                    results.append({
                        "title": (res.select_one(".result__title") or "").get_text(strip=True) if res.select_one(".result__title") else "",
                        "snippet": snip.get_text(strip=True),
                        "url": (res.select_one(".result__url") or "").get_text(strip=True) if res.select_one(".result__url") else "",
                    })
        except Exception as e: print(f"⚠️  DDG HTML: {e}")
    print(f"🔍 Tier 3: {len(results)} results")
    return results[:n]

# ── LLM helpers ───────────────────────────────────────────────────────────────
_SYS = """You are a precise Indian Government Scheme Assistant. Answer citizens\'s questions about Indian government welfare schemes.

STRICT RULES — violating any of these is an error:
1. Use ONLY facts that appear verbatim in the Context. Never invent, infer, or add any fact not explicitly stated.
2. NEVER echo the context labels like "[Doc 1: ...]" or "[Source 1: ...]" anywhere in your answer. Those are internal references only.
3. NEVER use placeholder text like "[name]", "[url]", "[scheme name]" — use the actual name/URL or omit it.
4. If a fact is not in the context, do NOT mention it at all. Do not say "not specified" either — just skip it.
5. Structure your answer EXACTLY like this:
   Line 1: [Full scheme name] — [one sentence describing the scheme]
   Then: bullet points (•) for every relevant fact, copied verbatim from context
   Last line: Source: [actual scheme name]  (omit URL line if you don't have a real URL)
6. Be EXHAUSTIVE — list every relevant bullet point from the context.
7. No repetition. Write in English only — translation is handled separately after you respond."""

def _llm(messages):
    r = requests.post(HF_ROUTER_URL,
        headers={"Authorization":f"Bearer {hf_token}","Content-Type":"application/json"},
        json={"model":LLM_MODEL,"messages":messages,"max_tokens":900,"temperature":0.2,
              "repetition_penalty":1.15,"stop":["<|eot_id|>","---","Note:","Disclaimer:"]},
        timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def _fmt(chunks):
    return "\n\n".join(f"[Doc {i}: {c['scheme_name']} — {c['section']}]\n{c['text']}" for i,c in enumerate(chunks,1))

def _dedup(text):
    seen, out = [], []
    for s in re.split(r'(?<=[।.!?\n])\s*', text):
        c = s.strip()
        if not c: continue
        if c not in seen: seen.append(c); out.append(s)
        if seen.count(c) >= 2: break
    return " ".join(out).strip()

# Patterns that should NEVER appear in the final answer
_LEAK_PATTERNS = [
    re.compile(r'\[Doc \d+:[^\]]+\]', re.IGNORECASE),          # [Doc 1: scheme — section]
    re.compile(r'\[Source \d+:[^\]]+\]', re.IGNORECASE),        # [Source 1: ...]
    re.compile(r'\[Official Website:[^\]]+\]', re.IGNORECASE),   # [Official Website: url]
    re.compile(r'\[Web \d+:[^\]]+\]', re.IGNORECASE),           # [Web 1: title | url]
    re.compile(r'Source:\s*\[.*?\].*$', re.MULTILINE),           # Source: [name] | [url]
    re.compile(r'\[I don\'t have the exact URL\]', re.IGNORECASE),
    re.compile(r'\[name\]|\[url\]|\[scheme name\]', re.IGNORECASE),
]

def _clean(text):
    """Strip any internal context labels that leaked into the LLM output."""
    for pat in _LEAK_PATTERNS:
        text = pat.sub("", text)
    # Collapse multiple blank lines left after removal
    text = re.sub(r'\n{3,}', "\n\n", text)
    # Fix "Source:  \n" left after removing [name] placeholders
    text = re.sub(r'Source:\s*[\|\s]*$', "", text, flags=re.MULTILINE)
    return text.strip()

def _process(text):
    """Deduplicate then clean — apply to every LLM output."""
    return _clean(_dedup(text))

def _no_info(text):
    return any(p in text.lower() for p in [
        "i don't have information","not in my documents","context does not",
        "cannot find","no information","not mentioned","not provided",
        "not available","don't have specific","i do not have",
    ])

# ── Main answer function ──────────────────────────────────────────────────────
def answer(query, language="English"):
    """
    3-tier cascade:
      Tier 1: ChromaDB (.txt/.pdf/.docx)
        sufficient if: score >= 0.55 AND answer not thin AND len > 120
      Tier 2: official URL from Section 7 of matching .txt
        (always tried when Tier 1 insufficient)
      Tier 3: DuckDuckGo web search
        (only when score < 0.30 or answer still thin after Tier 2)
    """
    # Translate query → English
    eq = to_english(query, language)

    # ─── Tier 1 ───────────────────────────────────────────────────────────────
    chunks    = retrieve(eq)
    top_score = chunks[0]["score"] if chunks else 0.0
    print(f"📊 score={top_score:.3f}  chunks={len(chunks)}")

    if top_score < SCORE_HOPELESS:
        msg = "I couldn't find relevant information. Please visit https://india.gov.in for official details."
        return {"answer": to_local(msg, language), "sources":[], "chunks":[], "tier":"none"}

    tier, eng = "txt", ""
    try:
        eng = _process(_llm([{"role":"system","content":_SYS},
                           {"role":"user","content":f"Context:\n{_fmt(chunks)}\n\nQuestion: {eq}"}]))
        print(f"✅ Tier 1: {len(eng)} chars")
    except Exception as e: print(f"❌ Tier 1 LLM: {e}")

    tier1_ok = top_score >= SCORE_CONFIDENT and eng and not _no_info(eng) and len(eng) > 120

    # ─── Tier 2: official scheme URL ──────────────────────────────────────────
    if not tier1_ok:
        scheme   = chunks[0]["scheme_name"] if chunks else ""
        url, txt = tier2_fetch(scheme)
        if txt:
            tier = "url"
            sys2 = _SYS + "\nYou have the indexed doc AND official website. Use website to fill document gaps."
            try:
                doc_ctx = _fmt(chunks) if chunks else "No indexed chunks."
                web_ctx = f"[Official Website: {url}]\n{txt[:3500]}"
                candidate = _process(_llm([
                    {"role":"system","content":sys2},
                    {"role":"user","content":f"Indexed Doc:\n{doc_ctx}\n\nOfficial Website:\n{web_ctx}\n\nQuestion: {eq}"}
                ]))
                if candidate and (not eng or _no_info(eng) or len(candidate) > len(eng)):
                    eng = candidate
                    print(f"✅ Tier 2: {len(eng)} chars")
            except Exception as e: print(f"❌ Tier 2 LLM: {e}")

    tier2_ok = eng and not _no_info(eng) and len(eng) > 120 and top_score >= SCORE_TRY_WEB

    # ─── Tier 3: web search ───────────────────────────────────────────────────
    if not tier2_ok:
        tier  = "web"
        webs  = tier3_search(eq)
        if webs:
            sys3 = _SYS + "\nYou have indexed docs AND live web results. Use web to fill gaps. Cite URLs for web facts."
            web_ctx = "\n\n".join(f"[Web {i}: {r['title']} | {r['url']}]\n{r['snippet']}" for i,r in enumerate(webs,1))
            doc_ctx = _fmt(chunks) if chunks else "No indexed chunks."
            try:
                candidate = _process(_llm([
                    {"role":"system","content":sys3},
                    {"role":"user","content":f"Docs:\n{doc_ctx}\n\nWeb:\n{web_ctx}\n\nQuestion: {eq}"}
                ]))
                if candidate and (not eng or _no_info(eng) or len(candidate) > len(eng)):
                    eng = candidate
                    print(f"✅ Tier 3: {len(eng)} chars")
            except Exception as e: print(f"❌ Tier 3 LLM: {e}")

    if not eng or _no_info(eng):
        eng  = "I couldn't find complete information. Please visit https://india.gov.in for official details."
        tier = "none"

    return {
        "answer":         to_local(eng, language),
        "english_answer": eng,
        "tier":           tier,
        "sources": [{"scheme":c["scheme_name"],"section":c["section"],"score":c["score"]} for c in chunks],
        "chunks": chunks,
    }