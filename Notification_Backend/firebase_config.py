import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

firebase_creds = json.loads(os.getenv("FIREBASE_CREDENTIALS"))

cred = credentials.Certificate(firebase_creds)

firebase_admin.initialize_app(cred)

db = firestore.client()

#for local testing
# import firebase_admin
# from firebase_admin import credentials, firestore, messaging

# cred = credentials.Certificate("serviceAccountKey.json")

# firebase_admin.initialize_app(cred)

# db = firestore.client()