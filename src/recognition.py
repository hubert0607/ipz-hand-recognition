import cv2
import mediapipe as mp
from ultralytics import YOLO
from keypoint_classifier import KeyPointClassifier
import copy
import itertools
import asyncio
import json
import websockets


global detected_persons
detected_persons = {}

class Detection:
    def __init__(self):
        #zmien to potem na cuda
        self.person_detector = YOLO('yolov8n.pt').to('mps')
        #padding zostal dodany dlatego ze yolo czasami ucinalo delikatnie koncowki palcu kiedy ramiona byly szeroko wyprostowane
        self.padding_percent = 0.05  # 5% padding

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            model_complexity=1
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.static_gesture_labels = ['Open', 'Close', 'OK', 'Peace Sign']


    def detect(self, frame):
        results_dict = {}

        detected_objects = self.person_detector(frame, verbose=False)[0]
        
        for person_id, box in enumerate(detected_objects.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = box
            
            if int(class_id) != 0:  # Skip if not a person
                continue

            padded_x1, padded_y1, padded_x2, padded_y2 = self._add_padding_to_box([x1, y1, x2, y2], frame)

            person_frame = frame[padded_y1:padded_y2, padded_x1:padded_x2]
            if person_frame.size == 0:
                continue

            rgb_person_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
                        
            person = {'box': [x1, y1, x2, y2]}
            person["hands"] = self._find_hands_landmarks(rgb_person_frame, padded_x1, padded_y1)


            results_dict[f'person_{person_id}'] = person

        return results_dict


    def _add_padding_to_box(self, box, frame):
        frame_height, frame_width = frame.shape[:2]
        padding_x = int(frame_width * self.padding_percent)
        padding_y = int(frame_height * self.padding_percent)
        
        x1, y1, x2, y2 = map(int, box)
        
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(frame_width, x2 + padding_x)
        y2 = min(frame_height, y2 + padding_y)
        
        return x1, y1, x2, y2


    def _find_hands_landmarks(self,rgb_person_frame, padded_x1, padded_y1):
        hands_results = self.hands.process(rgb_person_frame)
        hands = {}

        if hands_results.multi_hand_landmarks:
            for hand_id, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                hand_points = []
                for point in hand_landmarks.landmark:
                    # Convert relative coordinates to absolute
                    x = int(point.x * rgb_person_frame.shape[1] + padded_x1)
                    y = int(point.y * rgb_person_frame.shape[0] + padded_y1)
                    hand_points.append([x, y])

                pre_processed_landmarks = self._pre_process_landmark(hand_points)
                hand_sign_id = self.keypoint_classifier(pre_processed_landmarks)
                hand_sign_text = self.static_gesture_labels[min(hand_sign_id, len(self.static_gesture_labels)-1)]
                
                
                hands[f'hand_{hand_id}'] = {
                    'landmarks': hand_points,
                    'normalized_landmarks': pre_processed_landmarks,
                    'gesture': hand_sign_text,
                }
        return hands


    def _pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to 1D list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalize
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value if max_value != 0 else 0

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list



def draw_landmarks(image, hand_data):
    if not hand_data:
        return image
    
    landmark_points = hand_data['landmarks']
    gesture = hand_data['gesture']
    
    # Define connections for hand skeleton
    connections = [
        # Thumb
        (2,3), (3,4),
        # Index finger
        (5,6), (6,7), (7,8),
        # Middle finger
        (9,10), (10,11), (11,12),
        # Ring finger
        (13,14), (14,15), (15,16),
        # Pinky
        (17,18), (18,19), (19,20),
        # Palm
        (0,1), (1,2), (2,5), (5,9), (9,13), (13,17), (17,0)
    ]
    
    # Draw connections
    for start, end in connections:
        cv2.line(image, tuple(landmark_points[start]), tuple(landmark_points[end]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_points[start]), tuple(landmark_points[end]),
                (255, 255, 255), 2)

    # Draw landmarks
    for i, point in enumerate(landmark_points):
        radius = 8 if i in [4,8,12,16,20] else 5  # Bigger circles for fingertips
        cv2.circle(image, tuple(point), radius, (255,255,255), -1)
        cv2.circle(image, tuple(point), radius, (0,0,0), 1)
    
    # Draw gesture text
    text_position = (landmark_points[0][0], landmark_points[0][1] - 10)  # Above the wrist
    cv2.putText(image, gesture, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(image, gesture, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    return image


async def send_data(websocket):
    """Send data every 25ms to connected WebSocket clients."""
    data = detected_persons
    try:
        while True:
            # Serialize data to JSON
            message = json.dumps(data)
            await websocket.send(message)
            print(f"Sent: {message}")
            await asyncio.sleep(0.025)  # 25ms interval
    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed by the client.")

async def websocket_handler(websocket, path):
    """Handle incoming WebSocket connections."""
    print(f"Client connected: {path}")
    await send_data(websocket)

async def main_async():
    detection = Detection()
    cap = cv2.VideoCapture(0)
    
    # Start WebSocket server in the background
    server = await websockets.serve(websocket_handler, "127.0.0.1", 8765)
    print("Server running on ws://127.0.0.1:8765")

    # Set optimal resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detected_persons = detection.detect(frame)
            
            for person in detected_persons.values():
                # Draw person box
                box = person['box']
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw hand landmarks
                for hand_data in person['hands'].values():
                    frame = draw_landmarks(frame, hand_data)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Give other tasks a chance to run
            await asyncio.sleep(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        server.close()
        await server.wait_closed()

def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    main()