from fastapi import FastAPI
from datetime import datetime
from firebase_config import db
from notification_service import notify_chat, notify_feedback
from pydantic import BaseModel

app = FastAPI(title="Notification System")


# ✅ Test route
@app.get("/")
def root():
    return {"message": "Notification system running"}


# =========================
# 🟢 CHAT SYSTEM
# =========================
class MessageRequest(BaseModel):
    chat_id: str
    sender_id: str
    receiver_id: str
    sender_name: str
    message: str


@app.post("/send-message")
def send_message(data: MessageRequest):

    try:

        # Save inside:
        # chats/{chat_id}/messages/{message_id}

        db.collection("chats") \
          .document(data.chat_id) \
          .collection("messages") \
          .add({
              "senderId": data.sender_id,
              "receiverId": data.receiver_id,
              "senderName": data.sender_name,
              "text": data.message,
              "timestamp": datetime.utcnow()
          })

        print("✅ MESSAGE SAVED IN CORRECT CHAT ROOM")

        # Trigger notification
        notify_chat(
            data.receiver_id,
            data.sender_name,
            data.message
        )

        return {"status": "message sent successfully"}

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}
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


