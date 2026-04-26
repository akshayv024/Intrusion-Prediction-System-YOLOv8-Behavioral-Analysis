import cv2
import numpy as np
import time
import threading
import smtplib
import os

from email.message import EmailMessage
from ultralytics import YOLO
from collections import defaultdict, deque


# ---------------- EMAIL CONFIG ----------------
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")


# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")


# ---------------- VIDEO CONFIG ----------------
VIDEO_PATH = "input/input_video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 0 else 30

dt = 1 / fps
delay = int(1000 / fps)

WINDOW_NAME = "AI Intrusion Prediction System"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


# ---------------- BOUNDARY ----------------
boundary_points = []


def draw_boundary(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        boundary_points.append((x, y))


cv2.setMouseCallback(WINDOW_NAME, draw_boundary)


# ---------------- TEMP STORAGE ----------------
track_history = defaultdict(lambda: deque(maxlen=5))
risk_memory = defaultdict(float)
approach_consistency = defaultdict(float)

in_session = defaultdict(bool)
alert_sent = defaultdict(bool)
exit_timer = defaultdict(float)

intrusion_count = 0

os.makedirs("screenshots", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ---------------- EMAIL ALERT ----------------
def send_alert(frame_copy, person_id):

    def worker():
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            image_path = (
                f"screenshots/intrusion_{person_id}_{timestamp}.jpg"
            )

            cv2.imwrite(image_path, frame_copy)

            if not all([SENDER_EMAIL, RECEIVER_EMAIL, APP_PASSWORD]):
                print("Email credentials not configured.")
                return

            msg = EmailMessage()
            msg["Subject"] = "Intrusion Detected"
            msg["From"] = SENDER_EMAIL
            msg["To"] = RECEIVER_EMAIL

            msg.set_content(
                "A person has entered the restricted area."
            )

            with open(image_path, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype="image",
                    subtype="jpg",
                    filename=os.path.basename(image_path)
                )

            with smtplib.SMTP_SSL(
                "smtp.gmail.com",
                465
            ) as server:
                server.login(
                    SENDER_EMAIL,
                    APP_PASSWORD
                )
                server.send_message(msg)

            print("Alert sent successfully.")

        except Exception as e:
            print("Email Error:", e)

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


# ---------------- GEOMETRY ----------------
def point_inside_polygon(point, polygon):

    if len(polygon) < 3:
        return False

    return cv2.pointPolygonTest(
        np.array(polygon, np.int32),
        point,
        False
    ) >= 0


def distance_to_polygon(point, polygon):

    def segment_distance(p, a, b):
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)

        ab = b - a

        t = np.clip(
            np.dot(p - a, ab) /
            (np.dot(ab, ab) + 1e-6),
            0,
            1
        )

        projection = a + t * ab

        return np.linalg.norm(
            p - projection
        )

    return min(
        segment_distance(
            point,
            polygon[i],
            polygon[(i + 1) % len(polygon)]
        )
        for i in range(len(polygon))
    )


# ---------------- UI HELPERS ----------------
def draw_corner_box(
    img,
    x1,
    y1,
    x2,
    y2,
    color,
    thickness=2,
    corner=18
):

    cv2.line(
        img,
        (x1, y1),
        (x1 + corner, y1),
        color,
        thickness
    )
    cv2.line(
        img,
        (x1, y1),
        (x1, y1 + corner),
        color,
        thickness
    )

    cv2.line(
        img,
        (x2, y1),
        (x2 - corner, y1),
        color,
        thickness
    )
    cv2.line(
        img,
        (x2, y1),
        (x2, y1 + corner),
        color,
        thickness
    )

    cv2.line(
        img,
        (x1, y2),
        (x1 + corner, y2),
        color,
        thickness
    )
    cv2.line(
        img,
        (x1, y2),
        (x1, y2 - corner),
        color,
        thickness
    )

    cv2.line(
        img,
        (x2, y2),
        (x2 - corner, y2),
        color,
        thickness
    )
    cv2.line(
        img,
        (x2, y2),
        (x2, y2 - corner),
        color,
        thickness
    )


def draw_label_panel(
    img,
    x1,
    y1,
    lines,
    color
):

    panel_height = 18 * len(lines)
    panel_width = 220

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x1, y1 - panel_height - 6),
        (x1 + panel_width, y1),
        (20, 20, 20),
        -1
    )

    img[:] = cv2.addWeighted(
        overlay,
        0.6,
        img,
        0.4,
        0
    )

    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x1 + 8, y1 - panel_height + 18 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )


# ---------------- MAIN LOOP ----------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (960, 540)
    )

    overlay = frame.copy()

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=False
    )

    if len(boundary_points) >= 3:
        cv2.fillPoly(
            overlay,
            [np.array(boundary_points)],
            (0, 0, 255)
        )

        frame = cv2.addWeighted(
            overlay,
            0.25,
            frame,
            0.75,
            0
        )

    global_prediction = 0

    for result in results:

        if result.boxes.id is None:
            continue

        for box, person_id in zip(
            result.boxes.xyxy,
            result.boxes.id
        ):

            person_id = int(person_id)

            x1, y1, x2, y2 = map(
                int,
                box
            )

            center_x = (x1 + x2) // 2
            foot_point = (center_x, y2)

            track_history[
                person_id
            ].append(
                foot_point
            )

            if len(boundary_points) < 3:
                continue

            current_distance = distance_to_polygon(
                foot_point,
                boundary_points
            )

            inside = point_inside_polygon(
                foot_point,
                boundary_points
            )

            approach = 0

            if len(track_history[person_id]) >= 2:

                previous_point = track_history[
                    person_id
                ][-2]

                previous_distance = distance_to_polygon(
                    previous_point,
                    boundary_points
                )

                approach = np.clip(
                    (
                        previous_distance -
                        current_distance
                    ) / max(previous_distance, 1),
                    -1,
                    1
                )

            if approach > 0.02:
                approach_consistency[
                    person_id
                ] = min(
                    1.0,
                    approach_consistency[
                        person_id
                    ] + 0.05
                )
            else:
                approach_consistency[
                    person_id
                ] = max(
                    0.0,
                    approach_consistency[
                        person_id
                    ] - 0.03
                )

            distance_score = np.clip(
                1 - current_distance / 400,
                0,
                1
            )

            eta_score = np.clip(
                1 - current_distance / 300,
                0,
                1
            )

            raw_risk = (
                0.4 * distance_score +
                0.3 * eta_score +
                0.3 * approach_consistency[person_id]
            )

            risk_memory[person_id] = (
                0.25 * raw_risk +
                0.75 * risk_memory[person_id]
            )

            risk = int(
                np.clip(
                    risk_memory[person_id] * 99,
                    0,
                    99
                )
            )

            if inside:

                risk = 100

                if not in_session[person_id]:
                    intrusion_count += 1
                    in_session[person_id] = True

                if not alert_sent[person_id]:
                    send_alert(
                        frame.copy(),
                        person_id
                    )
                    alert_sent[person_id] = True
                    exit_timer[person_id] = 0

            else:

                exit_timer[person_id] += dt

                if exit_timer[person_id] > 2:
                    in_session[person_id] = False
                    alert_sent[person_id] = False

            global_prediction = max(
                global_prediction,
                risk
            )

            if risk < 40:
                color = (80, 200, 120)
                status = "LOW RISK"

            elif risk < 70:
                color = (0, 215, 255)
                status = "APPROACHING"

            else:
                color = (0, 80, 255)
                status = "HIGH RISK"

            draw_corner_box(
                frame,
                x1,
                y1,
                x2,
                y2,
                color
            )

            draw_label_panel(
                frame,
                x1,
                y1,
                [
                    f"PERSON ID : {person_id}",
                    f"STATUS : {status}",
                    f"RISK : {risk}%"
                ],
                color
            )

            cv2.circle(
                frame,
                foot_point,
                4,
                (255, 0, 0),
                -1
            )

    cv2.putText(
        frame,
        f"PREDICTION LEVEL : {global_prediction}%",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"TOTAL INTRUSIONS : {intrusion_count}",
        (650, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    for point in boundary_points:
        cv2.circle(
            frame,
            point,
            5,
            (255, 255, 255),
            -1
        )

    if len(boundary_points) >= 2:
        cv2.polylines(
            frame,
            [np.array(boundary_points)],
            True,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    if cv2.waitKey(delay) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
