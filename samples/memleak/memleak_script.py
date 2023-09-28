import time
import numpy as np
import cv2

frame_w = 1280
frame_h = 720

noise_width = 250
noise_height = 250
start_top = 200
start_left = 200

noise = np.random.rand(noise_height, noise_width, 3) * 255
noise = noise.astype(np.uint8)
alpha = np.empty((noise_height, noise_width, 1), dtype=np.uint8)
alpha.fill(128)
noise = np.concatenate((noise, alpha), axis=2)


for i in range(5000):
    frame_np = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    frame_np[:, :, 3] = 255
    frame_mat = cv2.cuda.GpuMat(frame_np)

    stream = cv2.cuda.Stream()

    overlay = cv2.cuda.GpuMat(noise_height, noise_width, cv2.CV_8UC4)
    overlay.upload(noise, stream)

    row_range = start_top, start_top + noise_height
    col_range = start_left, start_left + noise_width
    frame_roi = cv2.cuda.GpuMat(frame_mat, row_range, col_range)

    # leak
    cv2.cuda.alphaComp(overlay, frame_roi, cv2.cuda.ALPHA_OVER, frame_roi, stream=stream)

    # no leak
    # cv2.cuda.alphaComp(overlay, frame_roi, cv2.cuda.ALPHA_OVER, frame_roi)

    # result = frame_mat.download(stream)
    stream.waitForCompletion()
    # cv2.imwrite(f"/data/memleak_script/result_{i}.jpg", result)

print('done, waiting for Ctrl+C')
# wait
while True:
    time.sleep(0.2)
