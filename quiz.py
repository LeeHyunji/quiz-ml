import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = cv2.imread('sample.jpeg')
results = model(img)
h, w, c = img.shape
print(f'width:  {w}px, height: {h}px')

result = results.pandas().xyxy[0].to_numpy()
result = [item for item in result if item[6]=='person']

tmp_img = cv2.imread('sample.jpeg')
for i, item in enumerate(result):
    cropped = img[int(item[1]):int(item[3]), int(item[0]):int(item[2])]
    cv2.imwrite(f'person{i+1}.png', cropped)
    cv2.rectangle(tmp_img, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255,255,255))
cv2.imwrite('result1.png', tmp_img)