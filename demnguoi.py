import cv2
import numpy as np
from collections import OrderedDict

# Lớp CentroidTracker để theo dõi các đối tượng và ID của chúng qua các khung hình
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()  # ID đối tượng và tọa độ centroid của chúng
        self.disappeared = OrderedDict()  # Theo dõi thời gian một đối tượng biến mất
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
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Tính khoảng cách giữa các centroid hiện tại và các centroid đầu vào
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

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


# Tải mô hình MobileNet-SSD đã được huấn luyện sẵn
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Danh sách các nhãn cho các lớp trong tập dữ liệu COCO, chỉ số 15 là 'person'
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Tải luồng video
cap = cv2.VideoCapture('test.mp4')

# Bộ đếm cho số người di chuyển lên và xuống
up_count = 0
down_count = 0

# Tạo một instance của CentroidTracker
ct = CentroidTracker()

# Từ điển để lưu trữ tọa độ y trước đó cho mỗi đối tượng
previous_y = {}

# Đặt vị trí của đường ngang (ở giữa màn hình)
line_position = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

# Vòng lặp qua các khung hình của video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Lấy kích thước khung hình
    (h, w) = frame.shape[:2]

    # Chuẩn bị blob đầu vào cho việc phát hiện đối tượng
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Thực hiện bước truyền tiếp để lấy các kết quả phát hiện
    detections = net.forward()

    # Khởi tạo danh sách để giữ các centroid được phát hiện
    centroids = []

    # Vòng lặp qua các kết quả phát hiện
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Chỉ xử lý những kết quả có độ tin cậy lớn hơn 50%
        if confidence > 0.5:
            # Lấy nhãn lớp
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            # Tính toán hộp bao quanh đối tượng
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Tính toán centroid
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)

            # Vẽ hộp bao quanh và nhãn
            label = f"Person {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Cập nhật centroid tracker với các centroid mới
    objects = ct.update(centroids)

    # Vòng lặp qua các đối tượng đang được theo dõi
    for (object_id, centroid) in objects.items():
        current_y = centroid[1]

        # Kiểm tra xem đối tượng này đã được thấy trước đó chưa
        if object_id in previous_y:
            prev_y = previous_y[object_id]

            # Kiểm tra xem người đó có đang di chuyển lên (từ dưới đường đến trên đường) hay không
            if prev_y > line_position and current_y < line_position:
                up_count += 1
                print(f"Người {object_id} di chuyển lên.")

            # Kiểm tra xem người đó có đang di chuyển xuống (từ trên đường xuống dưới đường) hay không
            elif prev_y < line_position and current_y > line_position:
                down_count += 1
                print(f"Người {object_id} di chuyển xuống.")

        # Lưu tọa độ y hiện tại là tọa độ y trước đó cho khung hình tiếp theo
        previous_y[object_id] = current_y

    # Vẽ đường ngang trên khung hình
    cv2.line(frame, (0, line_position), (w, line_position), (0, 0, 255), 2)

    # Hiển thị số người di chuyển lên và xuống trên khung hình
    cv2.putText(frame, f"len: {up_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"xuong: {down_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị khung hình với các phát hiện và bộ đếm
    cv2.imshow("Đếm người", frame)

    # Thoát vòng lặp với phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng video và đóng các cửa sổ
cap.release()
cv2.destroyAllWindows
