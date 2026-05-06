from fastapi import FastAPI
from datetime import datetime
from firebase_config import db
from notification_service import notify_chat, notify_feedback

app = FastAPI(title="Notification System")


# ✅ Test route
@app.get("/")
def root():
    return {"message": "Notification system running"}


# =========================
# 🟢 CHAT SYSTEM
# =========================
@app.post("/send-message")
def send_message(sender_id: str, receiver_id: str, sender_name: str, text: str):

    # Save message
    db.collection("messages").add({
        "senderId": sender_id,
        "receiverId": receiver_id,
        "senderName": sender_name,
        "text": text,
        "timestamp": datetime.utcnow()
    })

    # Send notification
    notify_chat(receiver_id, sender_name, text)

    return {"status": "message sent + notification triggered"}


# =========================
# 🟢 FEEDBACK SYSTEM
# =========================
@app.post("/add-feedback")
def add_feedback(patient_id: str, doctor_id: str, feedback: str):

    results_ref = db.collection("patients") \
                    .document(patient_id) \
                    .collection("combinedResults")

    docs = results_ref.order_by("timestamp", direction="DESCENDING").limit(1).stream()

    updated = False

    for doc in docs:
        doc.reference.update({
            "doctorFeedback": feedback,
            "doctorId": doctor_id
        })
        updated = True

    if not updated:
        return {"error": "No results found"}

    # Send notification
    notify_feedback(patient_id)

    return {"status": "feedback added + notification sent"}