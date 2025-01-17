from google.cloud import translate_v2 as translate
from google.oauth2 import service_account

# Path to your service account JSON key file
key_path = r"C:\Users\shiva\Downloads\ornate-course-445209-b8-5584f7706a67.json"  # Replace with the actual path

# Load credentials
credentials = service_account.Credentials.from_service_account_file(key_path)

# Initialize the Translate API client with credentials
client = translate.Client(credentials=credentials)

def translate_text(text, target_language):
    """Translates text into the target language."""
    result = client.translate(text, target_language=target_language)
    return result['translatedText']
