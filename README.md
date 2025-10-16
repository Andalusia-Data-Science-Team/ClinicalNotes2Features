# 🏥 ClinicalNotes2Features

Transform unstructured clinical notes into structured features using AI-powered natural language processing.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Overview

ClinicalNotes2Features is a powerful web application that automatically extracts structured clinical information from unstructured clinical notes using advanced language models. Built for healthcare professionals, researchers, and data analysts who need to convert free-text clinical documentation into structured, analyzable data.

### 🎯 Key Features

- **🤖 AI-Powered Extraction**: Uses Fireworks AI with Llama models for accurate clinical information extraction
- **📊 Structured Output**: Extracts 10 key clinical fields from unstructured notes
- **📁 Batch Processing**: Process multiple clinical notes efficiently with configurable batch sizes
- **📈 Real-time Statistics**: View extraction completion rates and field-by-field analysis
- **💾 Multiple Export Formats**: Download results as Excel or CSV files
- **🌐 Web Interface**: User-friendly Streamlit interface with drag-and-drop file upload
- **⚙️ Environment Configuration**: Secure configuration through environment variables
- **🔄 Error Handling**: Robust retry logic and fallback mechanisms

### 🏥 Extracted Clinical Fields

1. **Chief Complaint** - Primary reason for patient visit
2. **History of Present Illness** - Detailed symptom progression
3. **Past Medical History** - Previous conditions and surgeries
4. **Current Medications** - Active medications with dosages
5. **Allergies** - Known allergies and reactions
6. **Physical Exam** - Clinical examination findings
7. **Review of Systems** - Systematic symptom inventory
8. **Labs/Imaging Results** - Diagnostic test results
9. **Assessment/Impression** - Working diagnoses
10. **Plan** - Treatment recommendations and follow-up

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Fireworks AI API key [Get one here](https://fireworks.ai/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ClinicalNotes2Features.git
   cd ClinicalNotes2Features

2. **Create and activate virtual environment**
   ```bash
    python -m venv venv

    # On Windows
    venv\Scripts\activate

    # On macOS/Linux
    source venv/bin/activate

3. **Install dependencies**
   ```bash
    pip install -r requirements.txt

4. **Set up environment variables**
    Create a .env file in the root directory:
       ```env
    # Fireworks AI Configuration
    FIREWORKS_API_KEY=your_api_key_here

    # Model Configuration
    MODEL=accounts/fireworks/models/llama4-maverick-instruct-basic
    TEMPERATURE=0.0
    BATCH_SIZE=5

5. **Run the application**
   ```bash
    streamlit run app.py

6. **Open your browser**
    Navigate to http://localhost:8501

## 📁 Project Structure
   ```text
   ClinicalNotes2Features/
        ├── app.py                 # Main Streamlit application
        ├── src/
        │   ├── __init__.py
        │   ├── extractor.py       # ClinicalNotesExtractor class
        │   ├── prompts.py         # AI prompts and templates
        │   └── utils.py           # Utility functions
        ├── .env                   # Environment variables (create this)
        ├── requirements.txt       # Python dependencies
        └── README.md             # This file









