import streamlit as st
import pandas as pd
from src.extractor import ClinicalNotesExtractor
from src.utils import load_excel_notes, save_to_excel, validate_structured_data
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ClinicalNotes2Features",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🏥 ClinicalNotes2Features</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform unstructured clinical notes into structured features using AI</p>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# API Key input
api_key = st.sidebar.text_input(
    "🔑 Fireworks AI API Key",
    type="password",
    value=os.getenv("FIREWORKS_API_KEY", ""),
    help="Enter your Fireworks AI API key"
)

# Model selection
available_models = {
    "Llama 4 Scout Instruct (Basic)": "accounts/fireworks/models/llama4-maverick-instruct-basic",
    "Llama 3.3 70B Instruct": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "Llama 3.1 405B Instruct": "accounts/fireworks/models/llama-v3p1-405b-instruct",
}

model_display_name = st.sidebar.selectbox(
    "🤖 Select Model",
    list(available_models.keys()),
    index=0,
    help="Choose the Llama model for extraction"
)

model = available_models[model_display_name]

# Temperature
temperature = st.sidebar.slider(
    "🌡️ Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Lower values = more consistent, Higher values = more creative"
)

# Batch size
batch_size = st.sidebar.slider(
    "📦 Batch Size",
    min_value=1,
    max_value=20,
    value=5,
    help="Number of notes to process at once (smaller batches are more reliable)"
)

# Notes column name
notes_column = st.sidebar.text_input(
    "📋 Notes Column Name",
    value="Notes",
    help="Name of the column containing clinical notes in your Excel file"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Smaller batch sizes (3-5) work better for complex notes")

# Main content
st.header("📤 Upload Your Data")

uploaded_file = st.file_uploader(
    "Upload Excel file with clinical notes",
    type=['xlsx', 'xls'],
    help="Excel file should contain a column with clinical notes"
)

# Sample data display
with st.expander("📋 View Sample Input Format"):
    sample_data = {
        "Patient_ID": ["P001", "P002", "P003"],
        "Notes": [
            "Meropenem 2 grams q8h IV infusion over 3 hours. Send VBG. NGT feeding as tolerated. Monitor BP.",
            "Sodium Bicarbonate 200 mL IV stat. No position of the patient till doctor order.",
            "1 PRBC to be given over HD session. Send VBG. Monitor labs."
        ]
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

# Process button
if uploaded_file is not None:
    
    st.success("✅ File uploaded successfully!")
    
    # Display uploaded data preview
    try:
        notes, original_df = load_excel_notes(uploaded_file, notes_column)
        
        st.subheader("📊 Data Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Notes Found", len(notes))
        with col2:
            st.metric("Total Columns", len(original_df.columns))
        
        st.dataframe(original_df.head(10), use_container_width=True)
        
        # Process button
        if st.button("🚀 Extract Structured Features", type="primary", use_container_width=True):
            
            if not api_key:
                st.error("❌ Please enter your Fireworks AI API key in the sidebar")
            else:
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize extractor
                    status_text.text("🔧 Initializing Fireworks AI extractor...")
                    extractor = ClinicalNotesExtractor(
                        api_key=api_key, 
                        model=model,
                        temperature=temperature
                    )
                    progress_bar.progress(10)
                    
                    # Extract features
                    status_text.text(f"🔍 Extracting features from {len(notes)} notes using {model_display_name}...")
                    structured_data = extractor.extract_batch(notes, batch_size=batch_size)
                    progress_bar.progress(70)
                    
                    # Validate data
                    status_text.text("✔️ Validating extracted data...")
                    if not validate_structured_data(structured_data):
                        st.warning("⚠️ Some extracted data may not follow expected structure. Review the results carefully.")
                    progress_bar.progress(85)
                    
                    # Display results
                    status_text.text("📊 Preparing results...")
                    st.subheader("✨ Extracted Structured Features")
                    
                    # Convert to DataFrame
                    result_df = pd.DataFrame(structured_data)
                    
                    # Display with tabs
                    tab1, tab2, tab3 = st.tabs(["📋 Structured Data", "📊 Statistics", "🔍 Sample Results"])
                    
                    with tab1:
                        st.dataframe(result_df, use_container_width=True, height=400)
                        st.caption(f"Showing all {len(result_df)} extracted records")
                    
                    with tab2:
                        # Statistics
                        st.markdown("### 📈 Extraction Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Notes Processed", 
                                len(notes),
                                help="Total number of clinical notes processed"
                            )
                        
                        with col2:
                            medications = sum(1 for item in structured_data if item.get("Medication_Name") and item.get("Medication_Name").strip())
                            st.metric(
                                "Medications Found", 
                                medications,
                                delta=f"{(medications/len(notes)*100):.1f}%" if len(notes) > 0 else "0%",
                                help="Number of notes with medication information"
                            )
                        
                        with col3:
                            procedures = sum(1 for item in structured_data if item.get("Procedure_Type") and item.get("Procedure_Type").strip())
                            st.metric(
                                "Procedures Found", 
                                procedures,
                                delta=f"{(procedures/len(notes)*100):.1f}%" if len(notes) > 0 else "0%",
                                help="Number of notes with procedure information"
                            )
                        
                        with col4:
                            labs = sum(1 for item in structured_data if item.get("Lab_Test_Name") and item.get("Lab_Test_Name").strip())
                            st.metric(
                                "Lab Tests Found", 
                                labs,
                                delta=f"{(labs/len(notes)*100):.1f}%" if len(notes) > 0 else "0%",
                                help="Number of notes with lab test information"
                            )
                        
                        # Additional statistics
                        st.markdown("---")
                        col5, col6, col7, col8 = st.columns(4)
                        
                        with col5:
                            feeding = sum(1 for item in structured_data if item.get("Feeding_Type") and item.get("Feeding_Type").strip())
                            st.metric("Feeding Instructions", feeding)
                        
                        with col6:
                            vitals = sum(1 for item in structured_data if item.get("Vital_Sign") and item.get("Vital_Sign").strip())
                            st.metric("Vital Signs", vitals)
                        
                        with col7:
                            instructions = sum(1 for item in structured_data if item.get("Instruction") and item.get("Instruction").strip())
                            st.metric("Special Instructions", instructions)
                        
                        with col8:
                            sessions = sum(1 for item in structured_data if item.get("Session_Context") and item.get("Session_Context").strip())
                            st.metric("Session Contexts", sessions)
                        
                        # Field completion chart
                        st.markdown("### 📊 Field Completion Rate")
                        field_completion = {}
                        for field in result_df.columns:
                            non_empty = sum(1 for val in result_df[field] if val and str(val).strip())
                            field_completion[field] = (non_empty / len(result_df) * 100) if len(result_df) > 0 else 0
                        
                        completion_df = pd.DataFrame({
                            'Field': list(field_completion.keys()),
                            'Completion %': list(field_completion.values())
                        })
                        st.bar_chart(completion_df.set_index('Field'))
                    
                    with tab3:
                        st.markdown("### 🔍 Sample Extracted Records")
                        num_samples = min(5, len(result_df))
                        for i in range(num_samples):
                            with st.expander(f"Record {i+1}: {notes[i][:100]}..."):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.markdown("**Original Note:**")
                                    st.info(notes[i])
                                with col_b:
                                    st.markdown("**Extracted Features:**")
                                    st.json(structured_data[i])
                    
                    progress_bar.progress(95)
                    
                    # Download button
                    status_text.text("💾 Preparing download...")
                    excel_data = save_to_excel(structured_data, original_df)
                    progress_bar.progress(100)
                    status_text.text("✅ Extraction complete!")
                    
                    st.markdown("---")
                    st.subheader("📥 Download Results")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        st.download_button(
                            label="📥 Download Structured Features (Excel)",
                            data=excel_data,
                            file_name=f"structured_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        csv_data = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_data,
                            file_name=f"structured_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Success message
                    st.success("🎉 Feature extraction completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during extraction: {str(e)}")
                    with st.expander("🔍 View Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                    progress_bar.empty()
                    status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        with st.expander("🔍 View Error Details"):
            import traceback
            st.code(traceback.format_exc())

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload an Excel file to get started")
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Prepare Your Data**
        - Create an Excel file
        - Add clinical notes in a column
        - Name the column (e.g., "Notes")
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Configure Settings**
        - Enter your Fireworks AI API key
        - Select your preferred model
        - Adjust batch size if needed
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Extract Features**
        - Upload your Excel file
        - Click "Extract Structured Features"
        - Download the results
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🏥 ClinicalNotes2Features</strong> | Powered by Fireworks AI & Llama Models</p>
    <p style='font-size: 0.9rem;'>Transform unstructured clinical notes into actionable structured data</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        Built for healthcare data analysis and machine learning applications
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🔗 Quick Links
- [Fireworks AI Docs](https://docs.fireworks.ai/)
- [Get API Key](https://fireworks.ai/)

### ℹ️ About
Version: 1.0.0  
Built with Streamlit & Fireworks AI
""")