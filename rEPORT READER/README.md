# Report Reader

A Streamlit app for uploading PDF or TXT reports and asking questions about them.
It uses OpenRouter embeddings to find relevant report sections and an OpenRouter
chat model to answer from those sections.

## Setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Add your OpenRouter API key to `.env`:

   ```dotenv
   OPENROUTER_API_KEY=your-openrouter-api-key
   OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
   OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
   ```

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

The app accepts PDF and UTF-8 TXT files up to 20 MB. Scanned/image-only PDFs
need OCR before their text can be searched. Your report text and questions are
sent to OpenRouter to create embeddings and generate answers.
