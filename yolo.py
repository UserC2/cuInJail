import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import in_zone
import alert

safe_x_1 = 180
safe_x_2 = 480
safe_y_1 = 120
safe_y_2 = 350

# Load the YOLO model
yolo = YOLO('yolov8s.pt')

# Initialize Flask app
app = Flask(__name__)

# Load the video capture (0 for the first webcam device)
videoCap = cv2.VideoCapture(0)

# Function to get class colors (used to color bounding boxes)
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

cookieList = ['doughnut', 'cookie', 'cell phone'] #<> remove celll phopne - was for testing
alerted = False

# Function to generate frames for video streaming
def generate_frames():
    while True:
        cookieCount = 0

        # Capture a frame from the webcam
        ret, frame = videoCap.read()
        if not ret:
            continue
        
        # Perform YOLO inference on the captured frame
        results = yolo.track(frame, stream=True)

        for result in results:
            classes_names = result.names  # Class names returned by the model

            # Iterate over detected objects (bounding boxes)
            for box in result.boxes:
                if box.conf[0] > 0.4:  # Only consider detections with high confidence   

                    [x1, y1, x2, y2] = box.xyxy[0]  # Get bounding box coordinates
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])  # Get class index
                    class_name = classes_names[cls]  # Get the class name

                    out = False

                    # If it's a cookie and it's out of the safe zone, flag it
                    for name in cookieList:
                        if name == class_name and not in_zone.inZone(safe_x_1, safe_x_2, safe_y_1, safe_y_2, x1, x2, y1, y2):
                            cookieCount += 1
                            out = True

                    if not out:
                        colour = (0, 255, 0) # green
                    else:
                        colour = (0, 0, 255) # red

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                    if cookieCount > 1:  # If more than one cookie detected out of safe zone
                        alerted = True # Trigger an alert

        # draw cookie safe zone
        cv2.rectangle(frame, (safe_x_1, safe_y_1), (safe_x_2, safe_y_2), (0,0,255), 3)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        # Convert the frame to bytes
        frame = buffer.tobytes()
        
        # Yield the frame as a multipart HTTP response (for video streaming)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to display the video stream on the website
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the main page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
