import cv2
import numpy as np


def detect_white_plate(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=50,
                               maxRadius=200)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle
            # Extract the region of interest (ROI)
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), radius, 255, -1)
            roi = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert ROI to HSV and calculate white pixel ratio
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            white_ratio = cv2.countNonZero(white_mask) / (np.pi * radius ** 2)

            # If the white ratio is significant, highlight the detected circle
            if white_ratio > 0.7:  # Adjust threshold as necessary
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 4)
                cv2.putText(frame, "White Plate Detected",
                            (x - 50, y - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def main():
    # Access the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect white plate
        output_frame = detect_white_plate(frame)

        # Display the result
        cv2.imshow("White Plate Detection", output_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
