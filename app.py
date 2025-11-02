"""
==========================================================
DL Face Recognition Attendance System
Using facenet-pytorch: MTCNN + InceptionResnetV1
==========================================================

Requirements:
pip install facenet-pytorch torch torchvision numpy opencv-python sqlite3
"""

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime

# =============================
# CONFIGURATION
# =============================
THRESHOLD = 0.8  # Euclidean distance threshold for matching
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DB_PATH = 'attendance.db'
SNAPSHOT_FOLDER = 'unknown_faces'

# =============================
# INITIALIZE MODELS
# =============================
mtcnn = MTCNN(keep_all=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# =============================
# DATABASE FUNCTIONS
# =============================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            student_id INTEGER,
            embedding BLOB,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            student_id INTEGER,
            timestamp TEXT,
            status TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    conn.commit()
    conn.close()

def serialize_embedding(embedding):
    return embedding.tobytes()

def deserialize_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32)

def register_student(name, img_paths):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO students (name) VALUES (?)", (name,))
    conn.commit()
    c.execute("SELECT id FROM students WHERE name=?", (name,))
    student_id = c.fetchone()[0]

    for img_path in img_paths:
        emb = extract_face_embedding(img_path)
        if emb is not None:
            c.execute("INSERT INTO embeddings (student_id, embedding) VALUES (?, ?)", 
                      (student_id, serialize_embedding(emb)))
    conn.commit()
    conn.close()
    print(f"[INFO] Registered {name}")

def load_known_faces():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT students.name, embeddings.embedding FROM students JOIN embeddings ON students.id=embeddings.student_id")
    rows = c.fetchall()
    conn.close()

    names = []
    embeddings = []

    for name, emb_blob in rows:
        names.append(name)
        embeddings.append(deserialize_embedding(emb_blob))
    return names, embeddings

def mark_attendance_db(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM students WHERE name=?", (name,))
    res = c.fetchone()
    if res:
        student_id = res[0]
        c.execute("INSERT INTO attendance (student_id, timestamp, status) VALUES (?, ?, ?)",
                  (student_id, datetime.now().isoformat(), "PRESENT"))
        conn.commit()
        print(f"[ATTENDANCE] Marked PRESENT for {name}")
    conn.close()

# =============================
# HELPER FUNCTIONS
# =============================
def extract_face_embedding(img):
    """Detect face and get embedding vector"""
    if isinstance(img, str):  # image path
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return None
    x1, y1, x2, y2 = map(int, boxes[0])
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))
    face_tensor = torch.tensor(face/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
    embedding = resnet(face_tensor).detach().cpu().numpy()[0].astype(np.float32)
    return embedding

def compare_embeddings(emb1, emb2):
    """Compare two embeddings"""
    dist = np.linalg.norm(emb1 - emb2)
    match = dist < THRESHOLD
    confidence = max(0, 100*(1 - dist))
    return match, dist, confidence

# =============================
# LIVE ATTENDANCE
# =============================
def live_attendance():
    known_names, known_embeddings = load_known_faces()
    if not known_names:
        print("[ERROR] No registered students.")
        return

    if not os.path.exists(SNAPSHOT_FOLDER):
        os.makedirs(SNAPSHOT_FOLDER)

    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None:
            for (x1, y1, x2, y2) in boxes:
                face = img_rgb[int(y1):int(y2), int(x1):int(x2)]
                emb = extract_face_embedding(face)
                matched = False

                for name, known_emb in zip(known_names, known_embeddings):
                    match, dist, conf = compare_embeddings(emb, known_emb)
                    if match:
                        matched = True
                        mark_attendance_db(name)
                        label = f"{name} ({conf:.1f}%)"
                        color = (0,255,0)
                        break

                if not matched:
                    label = "Unknown"
                    color = (0,0,255)
                    # save snapshot
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(os.path.join(SNAPSHOT_FOLDER, f"unknown_{ts}.jpg"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    init_db()
    register_student("Vignesh bala R", ["test_images/vignesh bala.jpg"])
    register_student("Vasantha Kumar", ["test_images/vasantha kumar.jpg"])

    live_attendance()