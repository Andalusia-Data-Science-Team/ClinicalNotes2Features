import json
import re
import time 
from typing import List, Dict, Optional
from fireworks.client import Fireworks
from .prompts import SYSTEM_PROMPT, get_user_prompt


class ClinicalNotesExtractor:
    """
    Extract structured features from clinical notes using Fireworks AI
    """
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "accounts/fireworks/models/llama4-maverick-instruct-basic", 
        temperature: float = 0.0, 
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize the extractor
        
        Args:
            api_key: Fireworks AI API key
            model: Model to use (default: llama4-maverick-instruct-basic)
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        # Validate API key
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize client with error handling
        try:
            self.client = Fireworks(api_key=api_key)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Fireworks client: {str(e)}")       
         
    def extract_features(self, notes: List[str], retry_count: int = 0) -> List[Dict]:
        """
        Extract structured features from a list of clinical notes with retry logic
        
        Args:
            notes: List of clinical note strings
            retry_count: Current retry attempt (internal use)
            
        Returns:
            List of dictionaries containing structured features
        """
        try:
            # Validate input
            if not notes or len(notes) == 0:
                raise ValueError("Notes list cannot be empty")
            
            # Create the user prompt
            user_prompt = get_user_prompt(notes)
            
            # Calculate dynamic max_tokens
            estimated_tokens = len(user_prompt) // 4
            max_tokens = min(4096, max(1000, estimated_tokens * 2))
            
            # Call Fireworks AI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Validate response
            if not response or not response.choices:
                raise ValueError("Empty response from API")
            
            # Extract the response content
            content = response.choices[0].message.content
            
            if not content or not content.strip():
                raise ValueError("Empty content in API response")
            
            # Parse JSON response
            structured_data = self._parse_response(content)
            
            # Validate parsed data
            if not structured_data:
                raise ValueError("Failed to parse response into structured data")
            
            # Handle length mismatch
            if len(structured_data) != len(notes):
                print(f"⚠️ Warning: Expected {len(notes)} results, got {len(structured_data)}")
                
                # If too few, pad with empty structures
                if len(structured_data) < len(notes):
                    print(f"   Padding {len(notes) - len(structured_data)} missing records")
                    while len(structured_data) < len(notes):
                        structured_data.append(self._get_empty_structure())
                
                # If too many, truncate
                elif len(structured_data) > len(notes):
                    print(f"   Truncating {len(structured_data) - len(notes)} extra records")
                    structured_data = structured_data[:len(notes)]
            
            return structured_data
            
        except Exception as e:
            error_msg = f"Error during extraction (attempt {retry_count + 1}/{self.max_retries}): {str(e)}"
            print(error_msg)
            
            # Retry logic with exponential backoff
            if retry_count < self.max_retries - 1:
                wait_time = 2 ** retry_count  # 1s, 2s, 4s
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                return self.extract_features(notes, retry_count + 1)
            
            # All retries exhausted
            print(f"❌ All {self.max_retries} retry attempts failed")
            import traceback
            traceback.print_exc()
            
            # Return empty structures as fallback
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
            content = self._clean_markdown(content)
            
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
            print(f"Content preview: {content[:200]}...")
            
            # Try regex extraction as fallback
            return self._extract_json_with_regex(content)
        
        except Exception as e:
            print(f"Unexpected error in parse_response: {e}")
            import traceback
            traceback.print_exc()
        
        return []
    
    def _clean_markdown(self, content: str) -> str:
        """Remove markdown code blocks from content"""
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()
    
    def _extract_json_with_regex(self, content: str) -> List[Dict]:
        """Fallback: Extract JSON using regex"""
        # Try to extract JSON array
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
        
        return []
    
    def _is_valid_structure(self, obj: Dict) -> bool:
        """
        Check if object has the expected structure fields
        
        Args:
            obj: Dictionary to validate
            
        Returns:
            Boolean indicating if it's a valid structure
        """
        # Updated to match new clinical note structure
        expected_fields = {
            "Chief_Complaint", "History_Present_Illness", "Past_Medical_History",
            "Current_Medications", "Allergies", "Physical_Exam",
            "Review_of_Systems", "Labs_Imaging_Results", 
            "Assessment_Impression", "Plan"
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
        # Updated to match new clinical note structure
        return {
            "Chief_Complaint": "",
            "History_Present_Illness": "",
            "Past_Medical_History": "",
            "Current_Medications": "",
            "Allergies": "",
            "Physical_Exam": "",
            "Review_of_Systems": "",
            "Labs_Imaging_Results": "",
            "Assessment_Impression": "",
            "Plan": ""
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
                    print(f"⚠️ Batch {batch_number} returned no results")
                    results = self._get_empty_structures(len(batch))
                    failed_batches.append(batch_number)
                
                elif len(results) != len(batch):
                    print(f"⚠️ Batch {batch_number} count mismatch: expected {len(batch)}, got {len(results)}")
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
                print(f"❌ Error processing batch {batch_number}: {e}")
                import traceback
                traceback.print_exc()
                
                # Add empty structures for failed batch
                all_results.extend(self._get_empty_structures(len(batch)))
                failed_batches.append(batch_number)
        
        # Summary
        print(f"\n✅ Batch processing complete:")
        print(f"   Total batches: {total_batches}")
        print(f"   Successful: {total_batches - len(failed_batches)}")
        print(f"   Failed: {len(failed_batches)}")
        if failed_batches:
            print(f"   Failed batch numbers: {failed_batches}")
        
        return all_results