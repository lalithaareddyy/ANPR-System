import cv2
import threading
import imutils
import easyocr
import numpy as np
import argparse

class ANPRCamera:
    def __init__(self, video_source=0):
        self.stream = cv2.VideoCapture(video_source)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        (self.grabbed, self.frame) = self.stream.read()
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.stopped = False
        self.last_texts = []

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def get_frame(self):
        frame = imutils.resize(self.frame, width=640)
        results = self.reader.readtext(frame)
        self.last_texts = []

        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.last_texts.append(text)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_detected_texts(self):
        return self.last_texts

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action='store_true', help="Use live camera feed")
    args = parser.parse_args()

    if args.camera:
        cam = ANPRCamera()
        try:
            while True:
                frame_data = cam.get_frame()
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imshow("ANPR Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cam.stop()
            cv2.destroyAllWindows()
    else:
        print("Please provide --camera argument to start the ANPR camera.")
