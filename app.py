import cv2
import os
from flask import Flask, request, render_template
from datetime import date
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from PIL import Image, ImageEnhance


app = Flask(__name__)
nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll')
        
def totalreg():
    return len(os.listdir('static/faces'))

def equalize_histogram_color(image):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb_img)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb_img)
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img

def gaussian_blur(image):
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_img

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []
def identify_face(facearray):
    model = joblib.load('static/knn.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/knn.pkl')
    
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    l = len(df)
    return names, rolls, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid}')
            
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)
    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, l

@app.route('/')

def home():
    names, rolls, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, l = extract_attendance()
    if 'knn.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')
    
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            break
        
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])

def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                frame[y:y+h, x:x+w] = equalize_histogram_color(frame[y:y+h, x:x+w])
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                bright_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                bright_frame = ImageEnhance.Brightness(bright_frame).enhance(1.5)
                bright_frame = cv2.cvtColor(np.array(bright_frame), cv2.COLOR_RGB2BGR)
                cv2.imwrite(userimagefolder+'/'+newusername+'_'+str(i)+'_bright.jpg', bright_frame[y:y+h, x:x+w])
                dark_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                dark_frame = ImageEnhance.Brightness(dark_frame).enhance(0.5)
                dark_frame = cv2.cvtColor(np.array(dark_frame), cv2.COLOR_RGB2BGR)
                cv2.imwrite(userimagefolder+'/'+newusername+'_'+str(i)+'_dark.jpg', dark_frame[y:y+h, x:x+w])
                
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)