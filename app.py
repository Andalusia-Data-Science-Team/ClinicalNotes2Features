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
st.sidebar.info("💡 **Tip:** Smaller batch sizes (3-5) work better for complex clinical notes")

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
            "CC: Chest pain. HPI: 65yo male presents with crushing substernal chest pain x 2 hours, radiating to left arm. PMH: HTN, DM. Medications: Metformin 500mg BID, Lisinopril 10mg daily. PE: BP 160/90, HR 110. Assessment: STEMI. Plan: Aspirin 325mg stat, cath lab activation.",
            "Chief complaint: Shortness of breath for 3 days. Past history of COPD. Currently on albuterol inhaler PRN. Physical exam shows decreased breath sounds bilaterally. Plan: Increase bronchodilator therapy, pulmonology follow-up.",
            "Patient complains of abdominal pain. ROS: GI - nausea, vomiting; GU - dysuria. Allergies: Sulfa drugs - Stevens-Johnson syndrome. Assessment: UTI vs gastroenteritis. Plan: UA and culture, ciprofloxacin 500mg BID x 7 days."
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
                        # Statistics - Updated for clinical note structure
                        st.markdown("### 📈 Extraction Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Notes Processed", 
                                len(notes),
                                help="Total number of clinical notes processed"
                            )
                        
                        with col2:
                            chief_complaints = sum(1 for item in structured_data if item.get("Chief_Complaint") and item.get("Chief_Complaint").strip())
                            st.metric(
                                "Chief Complaints", 
                                chief_complaints,
                                delta=f"{(chief_complaints/len(notes)*100):.1f}%" if len(notes) > 0 else "0%",
                                help="Number of notes with chief complaint information"
                            )
                        
                        with col3:
                            hpi = sum(1 for item in structured_data if item.get("History_Present_Illness") and item.get("History_Present_Illness").strip())
                            st.metric(
                                "History Present Illness", 
                                hpi,
                                delta=f"{(hpi/len(notes)*100):.1f}%" if len(notes) > 0 else "0%",
                                help="Number of notes with HPI information"
                            )
                        
                        with col4:
                            pmh = sum(1 for item in structured_data if item.get("Past_Medical_History") and item.get("Past_Medical_History").strip())
                            st.metric(
                                "Past Medical History", 
                                pmh,
                                delta=f"{(pmh/len(notes)*100):.1f}%" if len(notes) > 0 else "0%",
                                help="Number of notes with past medical history"
                            )
                        
                        # Additional statistics - Row 2
                        st.markdown("---")
                        col5, col6, col7, col8 = st.columns(4)
                        
                        with col5:
                            medications = sum(1 for item in structured_data if item.get("Current_Medications") and item.get("Current_Medications").strip())
                            st.metric("Current Medications", medications)
                        
                        with col6:
                            allergies = sum(1 for item in structured_data if item.get("Allergies") and item.get("Allergies").strip())
                            st.metric("Allergies", allergies)
                        
                        with col7:
                            physical_exam = sum(1 for item in structured_data if item.get("Physical_Exam") and item.get("Physical_Exam").strip())
                            st.metric("Physical Exams", physical_exam)
                        
                        with col8:
                            ros = sum(1 for item in structured_data if item.get("Review_of_Systems") and item.get("Review_of_Systems").strip())
                            st.metric("Review of Systems", ros)
                        
                        # Additional statistics - Row 3
                        st.markdown("---")
                        col9, col10, col11 = st.columns(3)
                        
                        with col9:
                            labs = sum(1 for item in structured_data if item.get("Labs_Imaging_Results") and item.get("Labs_Imaging_Results").strip())
                            st.metric("Labs/Imaging", labs)
                        
                        with col10:
                            assessment = sum(1 for item in structured_data if item.get("Assessment_Impression") and item.get("Assessment_Impression").strip())
                            st.metric("Assessment/Impression", assessment)
                        
                        with col11:
                            plan = sum(1 for item in structured_data if item.get("Plan") and item.get("Plan").strip())
                            st.metric("Treatment Plans", plan)
                        
                        # Field completion chart
                        st.markdown("### 📊 Field Completion Rate")
                        field_completion = {}
                        field_labels = {
                            "Chief_Complaint": "Chief Complaint",
                            "History_Present_Illness": "History Present Illness", 
                            "Past_Medical_History": "Past Medical History",
                            "Current_Medications": "Current Medications",
                            "Allergies": "Allergies",
                            "Physical_Exam": "Physical Exam",
                            "Review_of_Systems": "Review of Systems",
                            "Labs_Imaging_Results": "Labs/Imaging Results",
                            "Assessment_Impression": "Assessment/Impression",
                            "Plan": "Treatment Plan"
                        }
                        
                        for field in result_df.columns:
                            non_empty = sum(1 for val in result_df[field] if val and str(val).strip())
                            completion_rate = (non_empty / len(result_df) * 100) if len(result_df) > 0 else 0
                            field_name = field_labels.get(field, field)
                            field_completion[field_name] = completion_rate
                        
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
                                    
                                    # Display extracted features in a more readable format
                                    extracted = structured_data[i]
                                    for field, value in extracted.items():
                                        if value and str(value).strip():
                                            field_name = field_labels.get(field, field)
                                            st.markdown(f"**{field_name}:** {value}")
                    
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
                            file_name=f"clinical_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        csv_data = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_data,
                            file_name=f"clinical_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Success message
                    st.success("🎉 Clinical feature extraction completed successfully!")
                    
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
        - Create an Excel file with clinical notes
        - Include standard clinical sections:
          - Chief Complaint
          - History of Present Illness
          - Past Medical History
          - Current Medications
          - Physical Exam findings
          - Assessment & Plan
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Configure Settings**
        - Enter your Fireworks AI API key
        - Select your preferred model
        - Adjust batch size if needed
        - Specify the notes column name
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Extract Features**
        - Upload your Excel file
        - Click "Extract Structured Features"
        - Review the extracted clinical components
        - Download structured results
        """)
    
    # Feature explanation
    st.markdown("---")
    st.markdown("### 🏥 Extracted Clinical Features")
    
    feature_cols = st.columns(2)
    
    with feature_cols[0]:
        st.markdown("""
        **📋 Primary Components:**
        - **Chief Complaint:** Main reason for visit
        - **History Present Illness:** Detailed symptom description
        - **Past Medical History:** Previous conditions/surgeries
        - **Current Medications:** Active medications
        - **Allergies:** Known allergies and reactions
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **🔍 Clinical Assessment:**
        - **Physical Exam:** Examination findings
        - **Review of Systems:** Symptom inventory
        - **Labs/Imaging:** Diagnostic test results
        - **Assessment:** Working diagnoses
        - **Plan:** Treatment recommendations
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🏥 ClinicalNotes2Features</strong> | Powered by Fireworks AI & Llama Models</p>
    <p style='font-size: 0.9rem;'>Transform unstructured clinical notes into structured clinical components</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        Built for healthcare data analysis, EMR integration, and clinical research
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
Version: 2.0.0  
Clinical Note Structure Extraction  
Built with Streamlit & Fireworks AI
""")