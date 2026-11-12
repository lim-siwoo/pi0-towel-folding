"""이미지에 픽셀 격자를 그려 키포인트 좌표를 읽기 쉽게 한다. 사용: grid.py in.png out.png [x0 y0 x1 y1 scale]"""
import sys
import cv2
img = cv2.imread(sys.argv[1]); out = sys.argv[2]
x0, y0, x1, y1 = (int(v) for v in sys.argv[3:7]) if len(sys.argv) > 6 else (0, 0, img.shape[1], img.shape[0])
s = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
step = int(sys.argv[8]) if len(sys.argv) > 8 else 20
crop = img[y0:y1, x0:x1]
crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
for x in range((x0 // step + 1) * step, x1, step):
    major = x % (step * 5) == 0
    X = int((x - x0) * s)
    cv2.line(crop, (X, 0), (X, crop.shape[0]), (0, 255, 255) if major else (0, 160, 0), 1)
    if major: cv2.putText(crop, str(x), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
for y in range((y0 // step + 1) * step, y1, step):
    major = y % (step * 5) == 0
    Y = int((y - y0) * s)
    cv2.line(crop, (0, Y), (crop.shape[1], Y), (0, 255, 255) if major else (0, 160, 0), 1)
    if major: cv2.putText(crop, str(y), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
cv2.imwrite(out, crop)
