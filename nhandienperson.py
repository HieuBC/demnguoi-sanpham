import cv2
import numpy as np
from collections import OrderedDict

# Lớp CentroidTracker để theo dõi các đối tượng và ID của chúng qua các khung hình
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()  # ID của đối tượng và tâm của chúng
        self.disappeared = OrderedDict()  # Theo dõi thời gian biến mất của một đối tượng
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        # Đăng ký một đối tượng mới
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # Hủy đăng ký một đối tượng
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        # Nếu không có tâm nào được phát hiện
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Nếu không có đối tượng nào được theo dõi
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        # Nếu đang có đối tượng được theo dõi
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Tính khoảng cách giữa tâm hiện tại và tâm mới
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Cập nhật vị trí các đối tượng
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Xử lý các đối tượng không có tâm trùng khớp
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# Tải mô hình MobileNet-SSD đã được huấn luyện
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Nhãn cho các lớp trong tập dữ liệu COCO, chỉ số 15 là 'person' (người)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
           

# Tải video
cap = cv2.VideoCapture('test.mp4')

# Biến đếm người đi lên và xuống
up_count = 0
down_count = 0

# Tạo một đối tượng của CentroidTracker
ct = CentroidTracker()

# Từ điển để lưu tọa độ y trước đó của từng đối tượng
previous_y = {}

# Đặt vị trí đường ngang (ở giữa màn hình)
line_position = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

# Vòng lặp qua các khung hình video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Lấy kích thước khung hình
    (h, w) = frame.shape[:2]

    # Chuẩn bị đầu vào để phát hiện đối tượng
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Thực hiện forward pass để nhận kết quả phát hiện
    detections = net.forward()

    # Khởi tạo danh sách chứa các tâm phát hiện được
    centroids = []

    # Vòng lặp qua các đối tượng phát hiện
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Chỉ xử lý những phát hiện có độ tin cậy trên 50%
        if confidence > 0.5:
            # Lấy nhãn lớp
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            # Tính toán hộp bao quanh đối tượng
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Tính toán tâm của đối tượng
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)

            # Vẽ hộp bao và nhãn
            label = f"Person {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Cập nhật CentroidTracker với các tâm mới
    objects = ct.update(centroids)

    # Vòng lặp qua các đối tượng đang theo dõi
    for (object_id, centroid) in objects.items():
        current_y = centroid[1]

        # Kiểm tra nếu đối tượng đã từng được phát hiện trước đó
        if object_id in previous_y:
            prev_y = previous_y[object_id]

            # Kiểm tra xem người đó có di chuyển lên không (từ dưới đường lên trên)
            if prev_y > line_position and current_y < line_position:
                up_count += 1
                print(f"Person {object_id} moved up.")

            # Kiểm tra xem người đó có di chuyển xuống không (từ trên xuống dưới)
            elif prev_y < line_position and current_y > line_position:
                down_count += 1
                print(f"Person {object_id} moved down.")

        # Lưu tọa độ y hiện tại làm tọa độ y trước đó cho khung hình tiếp theo
        previous_y[object_id] = current_y

    # Vẽ đường ngang qua khung hình
    cv2.line(frame, (0, line_position), (w, line_position), (0, 0, 255), 2)

    # Hiển thị số lượng người đi lên và đi xuống trên khung hình
    cv2.putText(frame, f"Up: {up_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Down: {down_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị khung hình với phát hiện và số đếm
    cv2.imshow("People Counting", frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
