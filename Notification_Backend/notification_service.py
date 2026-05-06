from firebase_admin import messaging
from firebase_config import db
from datetime import datetime


# 🔹 Get user FCM token
def get_user_token(user_id: str):
    doc = db.collection("users").document(user_id).get()

    if not doc.exists:
        return None

    return doc.to_dict().get("fcm_token")


# 🔹 Save notification history (optional but good)
def save_notification(user_id: str, title: str, body: str):
    db.collection("notifications").add({
        "user_id": user_id,
        "title": title,
        "body": body,
        "timestamp": datetime.utcnow()
    })


# 🔹 Core send function
def send_notification_to_user(user_id: str, title: str, body: str):

    fcm_token = get_user_token(user_id)

    if not fcm_token:
        return {"error": "No FCM token found"}

    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        token=fcm_token
    )

    response = messaging.send(message)

    save_notification(user_id, title, body)

    return {"status": "sent", "response": response}


# 🔹 Chat notification
def notify_chat(receiver_id: str, sender_name: str, text: str):
    return send_notification_to_user(
        user_id=receiver_id,
        title="New Message",
        body=f"{sender_name}: {text}"
    )


# 🔹 Feedback notification
def notify_feedback(patient_id: str):
    return send_notification_to_user(
        user_id=patient_id,
        title="Doctor Feedback",
        body="Your doctor added feedback to your screening"
    )