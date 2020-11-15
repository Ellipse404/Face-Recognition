import cv2,  os
import numpy as np

# Initializing Algorithm :
algo = "HaarCascade Algorithm.xml"
haar = cv2.CascadeClassifier(algo)

data = 'E:\\PRANK...@@\\Dark Code [Illegal]\\AI MASTERCLASS\\Image Processing\\Face Detection\\Database'
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(data):
    for subdir in dirs:
        names[id] = subdir
        #path = 'E:\\PRANK...@@\\Dark Code [Illegal]\\AI MASTERCLASS\\Image Processing\\Face Detection\\'
        subjectpath = data + '\\' + subdir
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)
(images, labels) = [np.array(lis) for lis in [images, labels]]
#print(images, labels)

#  Loading Recognizer : use any one of them ..
model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()
# Training :
model.train(images, labels)
print("Training Completed")
# Capturing & Proccessing current frame :

cam = cv2.VideoCapture(0)
cnt=0
while True:
    _, img = cam.read()
    grayImg =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(img, 1.4, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face = grayImg[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)   # returns 2 things 1st-> lebels(0/1/2) 2nd-> Accuracy
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # checking prediction [right or not] :
        if prediction[1] < 800:
            cv2.putText(img, '%s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (10, 30, 255), 2)
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if cnt>100:
                print("Unknown Person !! ")
                cv2.imwrite("Unknown.jpg", img)
                cnt=0
    cv2.imshow("    <<<    FACE  RECOGNITION    >>>    ", img)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
    
cam.release()
cv2.destroyAllWindows()
