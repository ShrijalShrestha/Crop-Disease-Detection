import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("API_KEY")

def fetch_definition_data(disease_name):
    query = f"Provide a brief summary of the cure and precautionary measures for {disease_name}.Focus on actionable steps and essential information that can be quickly understood by farmers"  # Refine the query for clarity
    genai.configure(api_key=key)  # Ensure API key is correctly set
    
    try:
        # Request content from the Gemini model
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            contents=[{"role": "user", "parts": [{"text": query}]}]
        )

        # Access the correct part of the response
        # This assumes 'candidates' is an attribute and has the content you're looking for
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            # Extract the generated text from the first candidate
            generated_text = response.candidates[0].content.parts[0].text
            return generated_text
        else:
            return "No suggestions found."

    except Exception as e:
        print(f"Error fetching suggestions: {e}")
        return "Error fetching suggestions."
