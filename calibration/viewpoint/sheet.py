"""여러 프레임의 같은 영역을 격자와 함께 세로로 쌓는다. sheet.py out.png cam x0 y0 x1 y1 scale frame..."""
import sys
import cv2
import numpy as np
out, cam = sys.argv[1], sys.argv[2]
x0, y0, x1, y1 = (int(v) for v in sys.argv[3:7]); s = float(sys.argv[7]); frames = sys.argv[8:]
tiles = []
for fr in frames:
    img = cv2.imread(f"pi_frames/{fr}_{cam}.png") if not fr.startswith("/") else cv2.imread(fr)
    c = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for x in range((x0 // 10 + 1) * 10, x1, 10):
        X = int((x - x0) * s); maj = x % 50 == 0
        cv2.line(c, (X, 0), (X, c.shape[0]), (0, 255, 255) if maj else (0, 110, 0), 1)
        if maj: cv2.putText(c, str(x), (X + 2, 12), 0, 0.4, (0, 255, 255), 1)
    for y in range((y0 // 10 + 1) * 10, y1, 10):
        Y = int((y - y0) * s); maj = y % 50 == 0
        cv2.line(c, (0, Y), (c.shape[1], Y), (0, 255, 255) if maj else (0, 110, 0), 1)
        if maj: cv2.putText(c, str(y), (2, Y - 2), 0, 0.4, (0, 255, 255), 1)
    cv2.putText(c, fr.split("/")[-1], (c.shape[1] - 160, c.shape[0] - 8), 0, 0.6, (255, 0, 255), 2)
    tiles.append(c)
cv2.imwrite(out, np.vstack(tiles))
