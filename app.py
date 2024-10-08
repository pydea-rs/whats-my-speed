import cv2
import numpy as np
from collections import deque

def abs(n: float):
    return n if n >= 0 else -n

def precision(estimate, actual):
    return 100 * (1 - abs(estimate - actual)/actual)

def load_yolo():
    config_path = "yolo/yolov4-tiny.cfg"
    weights_path = "yolo/yolov4-tiny.weights"
    names_path = "yolo/coco.names"

    net = cv2.dnn.readNet(weights_path, config_path)
    
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

def create_tracker():
    return cv2.TrackerKCF_create()

def detect_vehicles(video_path):
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    real_world_width = 10.0
    last_positions = {}
    vehicle_speeds = {}
    trackers = {}
    window_size = 10
    speed_buffer = deque(maxlen=window_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == 'car':
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                vehicle_id = f'vehicle_{i}'
                current_position = (x + w // 2, y + h // 2)

                if vehicle_id in last_positions:
                    ok, bbox = trackers[vehicle_id].update(frame)
                    if ok:
                        x, y, w, h = [int(v) for v in bbox]
                        current_position = (x + w // 2, y + h // 2)

                        last_x, last_y = last_positions[vehicle_id]
                        distance_pixels = np.sqrt((current_position[0] - last_x)**2 + (current_position[1] - last_y)**2)

                        meters_per_pixel = real_world_width / width
                        distance_meters = distance_pixels * meters_per_pixel

                        time_diff = 1 / frame_rate
                        speed_m_s = distance_meters / time_diff
                        speed_km_h = speed_m_s * 3.6

                        speed_buffer.append(speed_km_h)
                        smoothed_speed = np.mean(speed_buffer)

                        vehicle_speeds[vehicle_id] = (speed_m_s, smoothed_speed)
                else:
                    tracker = create_tracker()
                    bbox = (x, y, w, h)
                    tracker.init(frame, bbox)
                    trackers[vehicle_id] = tracker

                last_positions[vehicle_id] = current_position

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                estimate = vehicle_speeds.get(vehicle_id, (0.0, 0.0))[1]
                print('70:', precision(estimate, 70), "80: ", precision(estimate, 80), "90: ", precision(estimate, 90))
                speed_text = f'{estimate:.2f} km/h'
                cv2.putText(frame, speed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Traffic Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "File0010.mp4"
    detect_vehicles(video_path)
