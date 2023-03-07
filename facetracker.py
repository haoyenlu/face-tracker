import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class RealTimeDetection():
    def __init__(self,model):
        self.cap = cv2.VideoCapture(0)
        self.facetracker = model
        
    def play(self):
        while self.cap.isOpened():
            _ , frame = self.cap.read()
            frame = frame[50:500, 50:500,:]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (120,120))

            yhat = self.facetracker.predict(np.expand_dims(resized/255,0))
            sample_coords = yhat[1][0]

            if yhat[0] > 0.8: 
                # Controls the main rectangle
                cv2.rectangle(frame, 
                              tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                              tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                                    (255,0,0), 2)
                # Controls the label rectangle
                cv2.rectangle(frame, 
                              tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                            [0,-30])),
                              tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                            [80,0])), 
                                    (255,0,0), -1)

                # Controls the text rendered
                cv2.putText(frame, f'Face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                                       [0,-5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('EyeTrack', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
        
    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    facetracker = load_model('facetracker.h5')
    print(facetracker.summary())
    detector = RealTimeDetection(model=facetracker)
    detector.play()
    