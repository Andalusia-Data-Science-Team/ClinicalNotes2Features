import pandas as pd
from typing import List, Dict, Tuple
import io


def load_excel_notes(file, notes_column: str = "Notes") -> Tuple[List[str], pd.DataFrame]:
    """
    Load clinical notes from Excel file
    
    Args:
        file: Uploaded file object
        notes_column: Name of the column containing notes
        
    Returns:
        Tuple of (list of clinical notes, original dataframe)
    """
    try:
        # Read Excel file
        df = pd.read_excel(file)
        
        # Check if notes column exists
        if notes_column not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Column '{notes_column}' not found in Excel file. "
                f"Available columns: {available_columns}"
            )
        
        # Remove NaN values and convert to list
        notes = df[notes_column].dropna().astype(str).tolist()
        
        # Check if we have any notes
        if len(notes) == 0:
            raise ValueError(f"No valid notes found in column '{notes_column}'")
        
        return notes, df
        
    except Exception as e:
        raise Exception(f"Error loading Excel file: {str(e)}")


def save_to_excel(structured_data: List[Dict], original_df: pd.DataFrame = None) -> bytes:
    """
    Save structured data to Excel file
    
    Args:
        structured_data: List of structured feature dictionaries
        original_df: Original dataframe to merge with (optional)
        
    Returns:
        Bytes object of Excel file, In this context, it usually means:The function doesn’t directly save the Excel file to disk.Instead, it generates the Excel file in memory (as bytes), which you can:
        Write to a file later, or Send as a response in a Streamlit or web app (for download)."""
    try:
        # Convert to DataFrame
        result_df = pd.DataFrame(structured_data)
        
        # If original dataframe provided, merge
        if original_df is not None and len(original_df) == len(result_df):
            # Merge original data with extracted features
            result_df = pd.concat([original_df.reset_index(drop=True), result_df], axis=1)
        
        # Save to bytes
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Structured_Features')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Structured_Features']
            for idx, col in enumerate(result_df.columns):
                max_length = max(
                    result_df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        raise Exception(f"Error saving to Excel: {str(e)}")


def validate_structured_data(data: List[Dict]) -> bool:
    """
    Validate that extracted data follows the expected structure
    
    Args:
        data: List of structured dictionaries
        
    Returns:
        Boolean indicating if data is valid
    """
    required_fields = [
        "Medication_Name", "Dosage", "Route", "Frequency",
        "Procedure_Type", "Lab_Test_Name", "Feeding_Type",
        "Feeding_Status", "Vital_Sign", "Timing",
        "Instruction", "Session_Context"
    ]
    
    # Check if data exists
    if not data or len(data) == 0:
        return False
    
    # Check each item
    for item in data:
        if not isinstance(item, dict):
            return False
        
        # Check if all required fields are present
        if not all(field in item for field in required_fields):
            return False
    
    return True


def clean_extracted_data(data: List[Dict]) -> List[Dict]:
    """
    Clean and normalize extracted data
    
    Args:
        data: List of structured dictionaries
        
    Returns:
        Cleaned list of structured dictionaries
    """
    cleaned_data = []
    
    for item in data:
        cleaned_item = {}
        for key, value in item.items():
            # Convert None to empty string
            if value is None:
                cleaned_item[key] = ""
            # Strip whitespace from strings
            elif isinstance(value, str):
                cleaned_item[key] = value.strip()
            else:
                cleaned_item[key] = str(value).strip()
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data


def get_data_summary(data: List[Dict]) -> Dict:
    """
    Generate summary statistics for extracted data
    
    Args:
        data: List of structured dictionaries
        
    Returns:
        Dictionary with summary statistics
    """
    if not data:
        return {
            "total_records": 0,
            "fields_populated": {},
            "completion_rate": 0.0
        }
    
    total_records = len(data)
    fields_populated = {}
    
    # Count non-empty values for each field
    for field in data[0].keys():
        count = sum(1 for item in data if item.get(field) and str(item.get(field)).strip())
        fields_populated[field] = {
            "count": count,
            "percentage": (count / total_records * 100) if total_records > 0 else 0
        }
    
    # Calculate overall completion rate
    total_fields = len(data[0].keys())
    total_populated = sum(item["count"] for item in fields_populated.values())
    completion_rate = (total_populated / (total_records * total_fields) * 100) if total_records > 0 else 0
    
    return {
        "total_records": total_records,
        "fields_populated": fields_populated,
        "completion_rate": round(completion_rate, 2)
    }


def export_to_csv(structured_data: List[Dict]) -> bytes:
    """
    Export structured data to CSV format
    
    Args:
        structured_data: List of structured feature dictionaries
        
    Returns:
        Bytes object of CSV file
    """
    try:
        df = pd.DataFrame(structured_data)
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        raise Exception(f"Error exporting to CSV: {str(e)}")


def export_to_json(structured_data: List[Dict]) -> str:
    """
    Export structured data to JSON format
    
    Args:
        structured_data: List of structured feature dictionaries
        
    Returns:
        JSON string
    """
    import json
    try:
        return json.dumps(structured_data, indent=2)
    except Exception as e:
        raise Exception(f"Error exporting to JSON: {str(e)}")


def filter_empty_records(data: List[Dict]) -> List[Dict]:
    """
    Filter out records where all fields are empty
    
    Args:
        data: List of structured dictionaries
        
    Returns:
        Filtered list with non-empty records
    """
    filtered_data = []
    
    for item in data:
        # Check if at least one field has a value
        has_value = any(
            value and str(value).strip() 
            for value in item.values()
        )
        
        if has_value:
            filtered_data.append(item)
    
    return filtered_data


def merge_dataframes(original_df: pd.DataFrame, extracted_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge original dataframe with extracted features
    
    Args:
        original_df: Original dataframe with clinical notes
        extracted_df: Dataframe with extracted features
        
    Returns:
        Merged dataframe
    """
    try:
        # Ensure both dataframes have the same number of rows
        if len(original_df) != len(extracted_df):
            raise ValueError(
                f"Dataframe length mismatch: original has {len(original_df)} rows, "
                f"extracted has {len(extracted_df)} rows"
            )
        
        # Reset indices
        original_df = original_df.reset_index(drop=True)
        extracted_df = extracted_df.reset_index(drop=True)
        
        # Concatenate horizontally
        merged_df = pd.concat([original_df, extracted_df], axis=1)
        
        return merged_df
        
    except Exception as e:
        raise Exception(f"Error merging dataframes: {str(e)}")


def validate_excel_file(file) -> Dict:
    """
    Validate uploaded Excel file
    
    Args:
        file: Uploaded file object
        
    Returns:
        Dictionary with validation results
    """
    try:
        df = pd.read_excel(file)
        
        return {
            "valid": True,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "has_data": len(df) > 0
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }