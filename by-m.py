import cv2
import numpy as np
import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from plyer import filechooser


class MyBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MyBoxLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'

        self.video_feed = Image(allow_stretch=True, keep_ratio=True)
        self.add_widget(self.video_feed)

        choose_video_button = Button(
            text='Choose Video', size_hint_y=None, height=40)
        choose_video_button.bind(on_press=self.choose_video)
        self.add_widget(choose_video_button)

        self.cap = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.learning_rate = 0.01
        self.min_distance = 300
        self.cars = []
        self.next_car_id = 1
        self.prev_car_info = {}
        self.start_time = time.time()
        self.is_running = False

    def choose_video(self, instance):
        file_path = filechooser.open_file(title="Choose a video file")[0]
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            Clock.schedule_interval(self.update, 1.0/30.0)

    def update(self, dt):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return
        fgmask = self.fgbg.apply(frame, learningRate=self.learning_rate)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_cars = []
        current_time = time.time()
        for contour in contours:
            if cv2.contourArea(contour) < 700:
                continue
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
            else:
                centroid_x, centroid_y = 0, 0
            found = False
            for car_id, (prev_centroid_x, prev_centroid_y, prev_time) in self.prev_car_info.items():
                if abs(centroid_x - prev_centroid_x) < self.min_distance and abs(centroid_y - prev_centroid_y) < self.min_distance:
                    time_diff = current_time - prev_time
                    distance = np.sqrt(
                        (centroid_x - prev_centroid_x) ** 2 + (centroid_y - prev_centroid_y)**2)
                    speed = distance / time_diff
                    self.cars.append((centroid_x, centroid_y, car_id, speed))
                    new_cars.append((centroid_x, centroid_y, car_id, speed))
                    found = True
                    break
            if not found:
                self.cars.append((centroid_x, centroid_y, self.next_car_id, 0))
                new_cars.append((centroid_x, centroid_y, self.next_car_id, 0))
                self.prev_car_info[self.next_car_id] = (
                    centroid_x, centroid_y, current_time)
                self.next_car_id += 1
        self.cars = new_cars
        for (centroid_x, centroid_y, car_id, speed) in self.cars:
            cv2.rectangle(frame, (centroid_x - 20, centroid_y - 20),
                          (centroid_x + 20, centroid_y + 20), (0, 255, 0), 2)
            cv2.putText(frame, f"Car {car_id} - Speed: {speed:.2f} km/h", (centroid_x,
                        centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.video_feed.texture = texture


class MyApp(App):
    def build(self):
        return MyBoxLayout()


if __name__ == '__main__':
    MyApp().run()
