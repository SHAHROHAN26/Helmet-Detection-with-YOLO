import cv2
import numpy as np
import sys
import glob
import queue, threading, time
class VideoCapture:
	def __init__(self,name):
		self.cap = cv2.VideoCapture(name)
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()
	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()
				except queue.Empty:
					pass
			self.q.put(frame)
	def read(self):
		return self.q.get()

#arg1=sys.argv[1]
# Load Yolo
net = cv2.dnn.readNet("custom-yolov4-tiny-detector_final.weights", "custom-yolov4-tiny-detector.cfg")
classes = []
with open("_darknet.labels", "r") as f:
	classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
video_capture = VideoCapture(0)

#process_this_frame=True
'''path="./*.png"
files = glob.glob (path)
for myFile in files:
	img = cv2.imread(myFile)'''
while True:
	img = video_capture.read()
	#print(myFile)
#path = input("enter the name of the image file without extention: ")
#img = cv2.imread(path)
	#img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape

# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(output_layers)

# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
				

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	#print(indexes)

	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[1]
			
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
			print(label)
	cv2.imshow("video", img)

	#cv2.imshow("Image", img)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break

	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#cv2.imwrite("./result"+myFile,img)