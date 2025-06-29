import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Check if the key was loaded
if not GOOGLE_API_KEY:
    print("ERROR: Could not find GOOGLE_API_KEY in your .env file.")
    exit()

# Configure the client
genai.configure(api_key=GOOGLE_API_KEY)

print("--- Attempting to list available models for your API key ---")

try:
    # This is the crucial part: we ask Google for the list of models
    for m in genai.list_models():
        # We check if the model supports the 'generateContent' method we need
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model found: {m.name}")

    print("\n--- Diagnostic Complete ---")
    print("If 'models/gemini-1.0-pro' or a similar 'pro' model is in the list above, the problem is in bot.py.")
    print("If you only see models like 'gemini-pro-vision' or 'embedding-001', your account may not have access to the chat model via the API.")

except Exception as e:
    print("\n--- AN ERROR OCCURRED ---")
    print(f"The API call failed with this error: {e}")
    print("This confirms the issue is with your API key, project configuration, or billing status.")
