from django.shortcuts import render,HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, HttpResponse
import cv2
import face_recognition
import pickle
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
import datetime
import pymongo


import os

if os.environ.get('USE_FACE') == 'true':
    import face_recognition
# Create your views here.
def home(request):
    return render(request,'index.html')
@csrf_exempt
def name_receive(request):
    if request.method == 'POST':
        try:
            if request.content_type.startswith('multipart/form-data'):
                emp_name = request.POST.get('name')
                print(emp_name)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Create dataset folder
            student_path = f"dataset/{emp_name}"
            os.makedirs(student_path, exist_ok=True)

            cam = cv2.VideoCapture(0)
            count = 0

            print(f"Capturing 50 images for {emp_name}. Please keep your face steady...")
            while count < 50:
                ret, frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    img_path = f"{student_path}/{count}.jpg"
                    cv2.imwrite(img_path, face_img)
                    count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("Face Capture", frame)
                if cv2.waitKey(1) == 27:  # Press ESC to exit
                    break

            cam.release()
            cv2.destroyAllWindows()
            print(f"Face capture complete! Images saved in {student_path}/")
        except Exception as e:
            return HttpResponse("Image captured successfully")
@csrf_exempt
def train_model(request):
    data = {}

    print("Training model...")

    for student in os.listdir("dataset"):
        student_path = os.path.join("dataset", student)
        images = []

        if not os.path.isdir(student_path):
            print(f"Skipping {student_path} (Not a directory)")
            continue

        for img_file in os.listdir(student_path):
            img_path = os.path.join(student_path, img_file)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("cv2.imread failed, img is None")

                # Convert BGR (OpenCV) to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            except Exception as e_cv2:
                try:
                    pil_image = Image.open(img_path)

                    if pil_image.mode not in ['RGB', 'L']:
                        print(f"Unsupported mode {pil_image.mode} in {img_path}, skipping.")
                        continue

                    pil_image = pil_image.convert('RGB')
                    img = np.array(pil_image)
                except UnidentifiedImageError:
                    print(f"Unidentified image: {img_path}")
                    continue
                except Exception as e_pil:
                    print(f"Error processing {img_path} with PIL: {e_pil}")
                    continue

            try:
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    images.append(encodings[0])
                else:
                    print(f"No faces found in {img_path}, skipping.")
            except Exception as e:
                print(f"Error encoding {img_path}: {e}")
                continue

        if images:
            data[student] = images
        else:
            print(f"No valid images found for {student}")

    with open("face_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Training model completed")

    return HttpResponse("Model training complete!")


@csrf_exempt
def attendance(reques):
    url = 'mongodb://localhost:27017/'
    client = pymongo.MongoClient(url)
    db = client['face_data']
    collection = db['attendance']

    with open("face_data.pkl", "rb") as f:
        data = pickle.load(f)

    cam = cv2.VideoCapture(0)


    attendance_file = "attendance.csv"
    try:
        attendance_df = pd.read_csv(attendance_file)
    except FileNotFoundError:
        attendance_df = pd.DataFrame(columns=["Name", "Time"])

    print("Starting attendance system...")

    while True:
        ret, frame = cam.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in face_encodings:
            for student, encodings in data.items():
                matches = face_recognition.compare_faces(encodings, encoding, tolerance=0.5)
                if True in matches:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    
                    if not ((attendance_df["Name"] == student) & (attendance_df["Time"].str.startswith(timestamp[:10]))).any():
                        new_entry = {"Name": student, "Time": timestamp}
                        attendance_df = attendance_df._append(new_entry, ignore_index=True)
                        collection.insert_one(new_entry)
                        attendance_df.to_csv(attendance_file, index=False)

                    cv2.putText(frame, student, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    print("Attendance marked successfully!")
    return HttpResponse("Attedance marked successfully")