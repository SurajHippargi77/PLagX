# PLagX Troubleshooting Guide

## Common Errors and Solutions

### Error: `ModuleNotFoundError: No module named 'fastapi'`

**Solution:** Install dependencies:
```powershell
pip install -r requirements.txt
```

If you get permission errors, try:
```powershell
pip install -r requirements.txt --user
```

Or use a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### Error: `Permission denied` when installing packages

**Solution:** 
1. Run PowerShell/CMD as Administrator, OR
2. Use `--user` flag: `pip install -r requirements.txt --user`, OR
3. Use a virtual environment (recommended)

---

### Error: `No module named 'uvicorn'` when running

**Solution:** Install uvicorn:
```powershell
pip install uvicorn[standard]
```

Or install all requirements:
```powershell
pip install -r requirements.txt
```

---

### Error: `ImportError` or `ModuleNotFoundError` for app modules

**Solution:** Make sure you're running from the project root (PLagX folder):
```powershell
cd C:\Users\hippa\PLagX
python -m uvicorn app.main:app --reload
```

---

### Error: Python 3.14 compatibility issues

**Note:** Python 3.14 is very new. If you encounter compatibility issues:

1. Try Python 3.11 or 3.12 (recommended)
2. Or update packages to latest versions:
   ```powershell
   pip install --upgrade fastapi uvicorn sentence-transformers scikit-learn PyPDF2 pydantic pydantic-settings
   ```

---

### Error: `AttributeError: 'FastAPI' object has no attribute 'on_event'`

**Solution:** Update FastAPI:
```powershell
pip install --upgrade fastapi
```

Or use FastAPI 0.109.2+ which supports `@app.on_event`.

---

### Error: Port 8000 already in use

**Solution:** Use a different port:
```powershell
python -m uvicorn app.main:app --reload --port 8001
```

---

### Error: Model download fails (sentence-transformers)

**Solution:** 
1. Check internet connection
2. The model downloads automatically on first use
3. If it fails, manually download:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
   ```

---

## Diagnostic Steps

1. **Run the test script:**
   ```powershell
   python test_imports.py
   ```
   This will show exactly which modules are missing.

2. **Check Python version:**
   ```powershell
   python --version
   ```
   Should be Python 3.9+ (3.11-3.12 recommended).

3. **Check installed packages:**
   ```powershell
   pip list | findstr "fastapi uvicorn sentence torch sklearn"
   ```

4. **Test imports manually:**
   ```powershell
   python -c "from app.main import app; print('OK')"
   ```

---

## Still Having Issues?

1. Share the **full error message** (copy/paste)
2. Share your **Python version**: `python --version`
3. Share output of: `pip list | findstr "fastapi uvicorn"`
4. Share output of: `python test_imports.py`
