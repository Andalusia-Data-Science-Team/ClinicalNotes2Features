import json
import re
import time 
from typing import List, Dict
from fireworks.client import Fireworks
from .prompts import SYSTEM_PROMPT, get_user_prompt


class ClinicalNotesExtractor:
    """
    Extract structured features from clinical notes using Fireworks AI
    """
    
    def __init__(self, api_key: str, model: str = "accounts/fireworks/models/llama4-maverick-instruct-basic", temperature: float = 0.0, max_retries: int = 3,
):
        """
        Initialize the extractor
        
        Args:
            api_key: Fireworks AI API key
            model: Model to use (default: llama4-maverick-instruct-basic)
            temperature: Temperature for generation
        """
        # Validate API key
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize client with error handling
        try:
            self.client = Fireworks(api_key=api_key)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Fireworks client: {str(e)}")       
         
    def extract_features(self, notes: List[str]) -> List[Dict]:
        """
        Extract structured features from a list of clinical notes
        
        Args:
            notes: List of clinical note strings
            
        Returns:
            List of dictionaries containing structured features
        """
        try:
            # Create the user prompt
            user_prompt = get_user_prompt(notes)
            
            # Call Fireworks AI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            
            # Extract the response
            content = response.choices[0].message.content
            
            # Parse JSON response
            structured_data = self._parse_response(content)
            
            # Ensure we have the right number of results
            if len(structured_data) != len(notes):
                print(f"Warning: Expected {len(notes)} results, got {len(structured_data)}")
                # Pad with empty structures if needed
                while len(structured_data) < len(notes):
                    structured_data.append(self._get_empty_structure())
            
            return structured_data
            
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_empty_structures(len(notes))
    
    def _parse_response(self, content: str) -> List[Dict]:
        """
        Parse the LLM response and extract JSON
        
        Args:
            content: Raw response from LLM
            
        Returns:
            List of structured dictionaries
        """
        try:
            # Clean the content
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Try to parse directly
            data = json.loads(content)
            
            # Handle different response formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check for common wrapper keys
                if 'results' in data:
                    return data['results']
                elif 'features' in data:
                    return data['features']
                elif 'data' in data:
                    return data['data']
                elif 'notes' in data:
                    return data['notes']
                elif 'extracted_features' in data:
                    return data['extracted_features']
                else:
                    # Check if it's a single structured object
                    if self._is_valid_structure(data):
                        return [data]
                    else:
                        # Try to find any list value
                        for key, value in data.items():
                            if isinstance(value, list):
                                return value
                        # Single object, wrap in list
                        return [data]
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Content: {content[:200]}...")  # Print first 200 chars for debugging
            
            # Try to extract JSON array from markdown or other formats
            json_match = re.search(r'$$.*$$', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception as parse_error:
                    print(f"Failed to parse extracted array: {parse_error}")
            
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    obj = json.loads(json_match.group())
                    if isinstance(obj, dict):
                        # Try to extract list from the object
                        for key, value in obj.items():
                            if isinstance(value, list):
                                return value
                        # Check if it's a valid structure
                        if self._is_valid_structure(obj):
                            return [obj]
                    return [obj] if isinstance(obj, dict) else obj
                except Exception as parse_error:
                    print(f"Failed to parse extracted object: {parse_error}")
        
        except Exception as e:
            print(f"Unexpected error in parse_response: {e}")
            import traceback
            traceback.print_exc()
        
        return []
    
    def _is_valid_structure(self, obj: Dict) -> bool:
        """
        Check if object has the expected structure fields
        
        Args:
            obj: Dictionary to validate
            
        Returns:
            Boolean indicating if it's a valid structure
        """
        expected_fields = {
            "Medication_Name", "Dosage", "Route", "Frequency",
            "Procedure_Type", "Lab_Test_Name", "Feeding_Type",
            "Feeding_Status", "Vital_Sign", "Timing",
            "Instruction", "Session_Context"
        }
        obj_fields = set(obj.keys())
        # Check if at least half of expected fields are present
        return len(expected_fields.intersection(obj_fields)) >= len(expected_fields) // 2
    
    def _get_empty_structure(self) -> Dict:
        """
        Generate a single empty structure dictionary
        
        Returns:
            Empty structured dictionary
        """
        return {
            "Medication_Name": "",
            "Dosage": "",
            "Route": "",
            "Frequency": "",
            "Procedure_Type": "",
            "Lab_Test_Name": "",
            "Feeding_Type": "",
            "Feeding_Status": "",
            "Vital_Sign": "",
            "Timing": "",
            "Instruction": "",
            "Session_Context": ""
        }
    
    def _get_empty_structures(self, count: int) -> List[Dict]:
        """
        Generate empty structure dictionaries
        
        Args:
            count: Number of empty structures needed
            
        Returns:
            List of empty structured dictionaries
        """
        return [self._get_empty_structure() for _ in range(count)]
    
def extract_batch(
    self, 
    notes: List[str], 
    batch_size: int = 5,
    progress_callback=None,
    rate_limit_delay: float = 0.5
) -> List[Dict]:
    """
    Extract features in batches with progress tracking
    
    Args:
        notes: List of all clinical notes
        batch_size: Number of notes per batch
        progress_callback: Optional callback function(current, total, message)
        rate_limit_delay: Delay between batches in seconds
        
    Returns:
        List of all structured features
    """
    all_results = []
    total_batches = (len(notes) + batch_size - 1) // batch_size
    failed_batches = []
    
    for i in range(0, len(notes), batch_size):
        batch = notes[i:i + batch_size]
        batch_number = i // batch_size + 1
        
        # Progress update
        message = f"Processing batch {batch_number}/{total_batches} ({len(batch)} notes)"
        print(message)
        
        if progress_callback:
            progress_callback(batch_number, total_batches, message)
        
        try:
            # Extract features for this batch
            results = self.extract_features(batch)
            
            # Validate results
            if not results or len(results) == 0:
                print(f"Batch {batch_number} returned no results")
                results = self._get_empty_structures(len(batch))
                failed_batches.append(batch_number)
            
            elif len(results) != len(batch):
                print(f"Batch {batch_number} count mismatch: expected {len(batch)}, got {len(results)}")
                # Pad or truncate
                if len(results) < len(batch):
                    results.extend(self._get_empty_structures(len(batch) - len(results)))
                else:
                    results = results[:len(batch)]
                
                failed_batches.append(batch_number)
            
            all_results.extend(results)
            
            # Rate limiting delay (except for last batch)
            if i + batch_size < len(notes):
                time.sleep(rate_limit_delay)
        
        except Exception as e:
            print(f"Error processing batch {batch_number}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add empty structures for failed batch
            all_results.extend(self._get_empty_structures(len(batch)))
            failed_batches.append(batch_number)
    
    # Summary
    print(f"\n Batch processing complete:")
    print(f"   Total batches: {total_batches}")
    print(f"   Successful: {total_batches - len(failed_batches)}")
    print(f"   Failed: {len(failed_batches)}")
    if failed_batches:
        print(f"   Failed batch numbers: {failed_batches}")
    
    return all_results