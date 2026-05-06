import firebase_admin
from firebase_admin import credentials, firestore

# Load your Firebase credentials
cred = credentials.Certificate("serviceAccountKey.json")

# Prevent multiple initialization
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Firestore DB instance
db = firestore.client()