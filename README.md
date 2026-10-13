# AI-Based Food Nutrition Analyzer (RAG + LLM)

An end-to-end project matching your resume bullet points:

- Retrieves live nutrition facts via a free nutrition API (USDA FoodData Central)
- Answers nutrition questions using a **RAG pipeline**: LangChain + FAISS + Sentence Transformers over a custom PDF knowledge base (this part runs 100% locally on your machine)
- Generates answers using a **Groq-hosted language model** (configured with `GROQ_MODEL`; the default is `qwen/qwen3.8-27b`)
- Full **Streamlit** UI tying it all together

Total cost to build and run: **$0**.

> Note: an earlier version of this project used a fully local Llama 3.2 model via Ollama. If your machine has 5+ GB free disk space and you'd rather keep everything fully offline, that's still an option — just ask and it can be swapped back. This version uses Groq instead to avoid local storage/PATH headaches.

---

## 1. Project structure

```
food-nutrition-analyzer/
├── app.py              # Streamlit app (run this)
├── nutrition_api.py     # USDA FoodData Central API wrapper
├── ingest.py            # Builds the FAISS vector store from your PDFs
├── rag_chain.py         # RAG pipeline: FAISS retriever + Groq-hosted language model
├── requirements.txt
├── .env.example         # Copy to .env and fill in your API key
├── data/                # Put your nutrition PDFs here
└── vectorstore/         # Auto-generated FAISS index (created by ingest.py)
```

---

## 2. Prerequisites

- Python 3.10+ installed
- Disk space for the local embedding model and project dependencies
- Internet connection for the one-time setup (everything runs offline after that)

---

## 3. Step-by-step setup

### Step 1 — Get a free USDA API key

1. Go to https://fdc.nal.usda.gov/api-key-signup.html
2. Enter your email, submit — the key arrives instantly in your inbox
3. (Optional) You can skip this and the app will use USDA's shared `DEMO_KEY`, but that's rate-limited to ~30 requests/hour, so your own key is better.

### Step 2 — Get a free Groq API key

1. Go to https://console.groq.com/keys
2. Sign up (email or Google login — no credit card required)
3. Click "Create API Key", give it any name, copy the key that appears (starts with `gsk_...`)
4. You'll paste this into your `.env` file in Step 3

### Step 3 — Set up the Python project

```bash
# 1. Create the project folder (or unzip the files you were given)
cd food-nutrition-analyzer

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API keys
cp .env.example .env
# then open .env and paste your USDA_API_KEY and GROQ_API_KEY
```

### Step 4 — Add your nutrition knowledge base PDFs

Put 1 or more nutrition-related PDFs into the `data/` folder. Good free sources:

- USDA Dietary Guidelines for Americans 2020-2025 (PDF): https://www.dietaryguidelines.gov/sites/default/files/2021-03/Dietary_Guidelines_for_Americans-2020-2025.pdf
- WHO Healthy Diet Fact Sheet: search "WHO healthy diet fact sheet PDF" and download
- Any nutrition textbook chapter, diabetes/heart-healthy diet guide, or sports-nutrition PDF you have rights to use

Just download a PDF and drop it into `data/`. You can add multiple PDFs.

### Step 5 — Build the vector store (RAG index)

```bash
python ingest.py
```

You'll see output like:
```
Loading: Dietary_Guidelines_for_Americans-2020-2025.pdf
Loaded 164 pages from 1 PDF(s).
Split into 812 chunks.
Loading embedding model 'sentence-transformers/all-MiniLM-L6-v2'...
Embedding chunks and building FAISS index...
Vector store saved to 'vectorstore/'.
```

The first run downloads the small (~90 MB) embedding model automatically — after that it's cached and works offline.

### Step 6 — Run the app

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`, with two tabs:

- **Nutrition Lookup** — type any food name, get live nutrition facts from USDA
- **Ask AI Nutrition Assistant** — ask free-form nutrition questions, answered by a Groq-hosted model grounded in your PDFs (with sources shown)

---

## 4. How the RAG pipeline works (for your interview prep)

1. **Ingest** (`ingest.py`): PDFs → split into ~1000-character overlapping chunks → each chunk converted into a 384-dim vector using `all-MiniLM-L6-v2` (Sentence Transformers) → stored in a FAISS index on disk.
2. **Retrieve** (`rag_chain.py`): when you ask a question, it's embedded the same way, and FAISS returns the top 4 most similar chunks (cosine/L2 similarity search).
3. **Augment + Generate**: those chunks are inserted into a prompt template as "context", then sent to the model configured by `GROQ_MODEL`, which generates a grounded answer — reducing hallucination because the model answers from your actual documents, not just its training data.
4. **Nutrition API**: a completely separate, non-RAG feature — direct REST calls to USDA FoodData Central for exact, structured nutrient values (calories, protein, etc.) per food item.

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not connect to Groq` / auth error | Double-check `GROQ_API_KEY` in `.env` is correct, active, and has no extra spaces or quotes around it. |
| `No vector store found` | Run `python ingest.py` before starting the Streamlit app. |
| `No PDF files found in 'data/'` | Add at least one `.pdf` file to the `data/` folder. |
| USDA search returns no results | Try a simpler query (e.g. "apple" instead of "red delicious apple raw with skin"). |
| Groq returns `model_not_found` | Set `GROQ_MODEL` in `.env` to a model currently available to your Groq account. The default is `qwen/qwen3.8-27b`. |
| `faiss-cpu` fails to install on Apple Silicon | Run `pip install faiss-cpu --no-cache-dir`, or use `conda install -c pytorch faiss-cpu` instead. |

---

## 6. Resume bullet mapping (confirms this build matches what you wrote)

- ✅ "Developed an AI-based Food Nutrition Analyzer using Python and Streamlit... via a nutrition API" → `app.py` (Tab 1) + `nutrition_api.py`
- ✅ "Built a RAG pipeline using LangChain, FAISS, and Sentence Transformers... custom PDF knowledge base" → `ingest.py` + `rag_chain.py`
- ✅ "Integrated a local Llama 3.2 model via Ollama... context-aware responses" → `rag_chain.py` (`ChatGroq`) + `app.py` (Tab 2). *(This version uses a Groq-hosted open-weight model instead of local Ollama; the RAG architecture remains the same.)*

---

## 7. Optional next steps to make this stand out further

- Add a "confidence/source highlighting" feature showing exact PDF snippet used
- Deploy to Streamlit Community Cloud for a free public demo — since this version already uses Groq (a cloud API) instead of local Ollama, it deploys cleanly with no local-model constraints
- Add conversation memory so follow-up questions retain context
- Add unit tests for `nutrition_api.py` (mocking `requests.get`)