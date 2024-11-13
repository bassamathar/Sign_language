from ultralytics import YOLO
import cv2
import numpy as np
cap = cv2.VideoCapture(0)

class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

class_names2 = [
    "Hello",
    "love you",
    "No",
    "Thank_you",
    "Yes"
]


model = YOLO('bestllee.pt')
model1 = YOLO('best (1).pt')
#image = np.ones((300, 300), dtype=np.uint8) * 255
detected_class_names = []
detected_class_names.clear()
while True:
    ret, frame = cap.read()
    #fin = np.zeros_like(frame)
    if not ret:
        print("Failed to grab frame, exiting...")
        break
    
    result = model(frame, stream=True)
    result1 = model1(frame, stream=True)
    for r in result:
        boxes = r.boxes  

        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()  
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), 4)
            class_idx = int(b.cls)
            class_name = class_names[class_idx]
            cv2.putText(frame, class_names[int(b.cls)], (x1 + 10, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 20, 30), 3)
            detected_class_names.append(class_name)
        for r in result1:
            boxes = r.boxes
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), 4)
                class_idx = int(b.cls)
                class_namess = class_names2[class_idx]
                cv2.putText(frame, class_names[int(b.cls)], (x1 + 10, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 20, 30), 3)
                detected_class_names.append(class_namess)

                if class_namess == "love you":
                    detected_class_names.remove('love you')

        image = np.ones((700, 700, 3), dtype=np.uint8) * (100, 30, 30)
        frame_resizes = cv2.resize(frame, (500, 500))
        y_offset = image.shape[0] - frame_resizes.shape[0]
        x_offset = image.shape[1] - frame_resizes.shape[1]

        image[y_offset:y_offset + frame_resizes.shape[0], x_offset: x_offset + frame_resizes.shape[1]] = frame_resizes

        cv2.rectangle(image, (0, 200), (700, 0), (255, 0, 0), 3)
        
    y_pos = 20
    line_height = 30 
    for class_name in detected_class_names:
        
        (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

        if y_pos + text_height > image.shape[0]:
            break  

        
        cv2.putText(image, class_name, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 222, 111), 1)
        

        
        y_pos += line_height



    image = image.astype(np.uint8)
           
    cv2.imshow('frame', frame)
    cv2.imshow('frames', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

