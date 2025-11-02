import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle

DB_PATH = "students.db"
UPLOAD_FOLDER = "uploads"
SNAPSHOT_FOLDER = "attendance_snapshots"
THRESHOLD = 0.8  # Euclidean distance threshold

# ===========================
# DATABASE FUNCTIONS
# ===========================
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
        CREATE TABLE IF NOT EXISTS face_encodings (
            student_id INTEGER,
            encoding BLOB,
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

def serialize_encoding(encoding):
    return pickle.dumps(encoding)

def deserialize_encoding(blob):
    return pickle.loads(blob)

# ===========================
# DL FACE RECOGNITION INIT
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ===========================
# REGISTER STUDENT
# ===========================
def register_student():
    name = simpledialog.askstring("Student Name", "Enter student name/ID:")
    if not name:
        return

    image_paths = filedialog.askopenfilenames(
        title=f"Select images for {name}",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not image_paths:
        messagebox.showwarning("No Images", "No images selected!")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO students (name) VALUES (?)", (name,))
    conn.commit()
    c.execute("SELECT id FROM students WHERE name=?", (name,))
    student_id = c.fetchone()[0]

    registered_count = 0
    for img_path in image_paths:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        faces = mtcnn(img)
        if faces is None:
            print(f"[WARNING] No face detected in {img_path}")
            continue
        for face in faces:
            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
            c.execute("INSERT INTO face_encodings (student_id, encoding) VALUES (?, ?)",
                      (student_id, serialize_encoding(embedding)))
            registered_count += 1

    conn.commit()
    conn.close()
    messagebox.showinfo("Registration Complete", f"Registered {registered_count} face encodings for {name}")

# ===========================
# LOAD STUDENTS
# ===========================
def load_students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT students.id, students.name, face_encodings.encoding FROM students JOIN face_encodings ON students.id=face_encodings.student_id")
    rows = c.fetchall()
    conn.close()

    known_face_encodings = []
    known_face_names = []

    for student_id, name, encoding_blob in rows:
        known_face_encodings.append(deserialize_encoding(encoding_blob))
        known_face_names.append(name)

    return known_face_encodings, known_face_names

# ===========================
# FACE MATCHING
# ===========================
def match_face(embedding, known_encodings, known_names):
    if len(known_encodings) == 0:
        return None, None, None
    distances = np.linalg.norm(np.array(known_encodings) - embedding, axis=1)
    best_index = np.argmin(distances)
    if distances[best_index] < THRESHOLD:
        return known_names[best_index], distances[best_index], (1 - distances[best_index]/THRESHOLD)*100
    else:
        return None, distances[best_index], 0

# ===========================
# MARK ATTENDANCE VIA WEBCAM
# ===========================
def mark_attendance():
    known_face_encodings, known_face_names = load_students()
    if len(known_face_encodings) == 0:
        messagebox.showwarning("No Registered Students", "No registered students found!")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if not os.path.exists(SNAPSHOT_FOLDER):
        os.makedirs(SNAPSHOT_FOLDER)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access the webcam")
        return

    messagebox.showinfo("Info", "Press 'q' to quit webcam after marking attendance.")

    marked_students = set()  # Keep track of already marked students

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = rgb_frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                face_tensor = torch.tensor(face_img/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu().numpy()[0]

                name, dist, confidence = match_face(embedding, known_face_encodings, known_face_names)

                if name and name not in marked_students:
                    marked_students.add(name)
                    c.execute("SELECT id FROM students WHERE name=?", (name,))
                    student_id = c.fetchone()[0]
                    timestamp = datetime.now().isoformat()
                    c.execute("INSERT INTO attendance (student_id, timestamp, status) VALUES (?, ?, ?)",
                              (student_id, timestamp, "PRESENT"))
                    conn.commit()

                    # Save snapshot
                    student_folder = os.path.join(SNAPSHOT_FOLDER, name)
                    if not os.path.exists(student_folder):
                        os.makedirs(student_folder)
                    snapshot_path = os.path.join(student_folder, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(snapshot_path, frame[y1:y2, x1:x2])
                    cv2.putText(frame, f"{name} PRESENT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                    # Stop after first recognition OR continue for multiple faces
                    cap.release()
                    cv2.destroyAllWindows()
                    conn.close()
                    messagebox.showinfo("Done", f"Attendance marked for {name}")
                    return  # exit function after marking

                elif not name:
                    unknown_folder = os.path.join(SNAPSHOT_FOLDER, "Unknown")
                    if not os.path.exists(unknown_folder):
                        os.makedirs(unknown_folder)
                    snapshot_path = os.path.join(unknown_folder, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(snapshot_path, frame[y1:y2, x1:x2])
                    cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

# ===========================
# VIEW REGISTERED STUDENTS
# ===========================
def view_students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name FROM students")
    rows = c.fetchall()
    conn.close()
    student_list = "\n".join([f"{student_id}: {name}" for student_id, name in rows])
    messagebox.showinfo("Registered Students", student_list or "No students registered.")

# ===========================
# VIEW ATTENDANCE LOGS
# ===========================
def view_attendance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT students.name, attendance.timestamp, attendance.status
        FROM attendance
        JOIN students ON attendance.student_id = students.id
        ORDER BY attendance.timestamp DESC
    """)
    rows = c.fetchall()
    conn.close()
    log_text = "\n".join([f"{name} | {timestamp} | {status}" for name, timestamp, status in rows])
    messagebox.showinfo("Attendance Logs", log_text or "No attendance recorded.")

# ===========================
# GUI SETUP
# ===========================
def main():
    init_db()
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    root = tk.Tk()
    root.title("DL Face Recognition Attendance System")

    tk.Label(root, text="Deep Learning Face Recognition Attendance System", font=("Helvetica", 16)).pack(pady=10)
    tk.Button(root, text="Register Student", command=register_student, width=30, height=2).pack(pady=5)
    tk.Button(root, text="Mark Attendance (Webcam)", command=mark_attendance, width=30, height=2).pack(pady=5)
    tk.Button(root, text="View Registered Students", command=view_students, width=30, height=2).pack(pady=5)
    tk.Button(root, text="View Attendance Logs", command=view_attendance, width=30, height=2).pack(pady=5)
    tk.Label(root, text="Attendance is marked automatically when a face is detected.", font=("Helvetica", 10)).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
