from ultralytics import YOLO
from PIL import Image

#load the model
model = YOLO('F:/NewJourney/Python/pythonProject/objectDetection/runs/detect/train/weights/best.pt')

#predict with the model
results = model('F:/NewJourney/Python/pythonProject/objectDetection/test')

#show the result
for r in results:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[...,::-1]) #RGB PIL Image
    im.show()
    im.save('result.jpg')
