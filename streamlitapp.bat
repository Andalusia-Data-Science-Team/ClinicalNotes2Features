@echo off
REM === Activate virtual environment ===
call venv\Scripts\activate


REM === Change directory to project folder ===
cd C:\DataScience\Testing\ClinicalNotes2Features


REM === Run Streamlit on custom port ===
streamlit run app.py --server.port 2002