from __future__ import print_function
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from PIL import ImageDraw
from PIL import ImageFont
from pycuda.tools import make_default_context
from data_processing_tiny import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import os
import common

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)
#vs = VideoStream(src="/dev/video0").start()
cap = cv2.VideoCapture('video2.mp4')

TRT_LOGGER = trt.Logger()
t0=time.time()
t1=time.time()
fps=0
def draw_bboxes(image_raw,bboxes, confidences, categories, all_categories, bbox_color='red'):
    """Draw the bounding boxes on the original input image and return it.
    
    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    global t0,t1,fps
    draw = ImageDraw.Draw(image_raw)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))
        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        font = ImageFont.truetype("openSans.ttf", 15)
        draw.text((left, top - 15), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color, font=font)
    font = ImageFont.truetype("openSans.ttf", 28)
    
    draw.text((2, 30), '{:.2f}FPS'.format(fps), fill='white', font=font)
    return image_raw

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main(inputSize):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    global vs, outputFrame, lock,t0,t1,fps
    cuda.init()
    device = cuda.Device(0)
    onnx_file_path = 'yolov3-tiny-416.onnx'
    engine_file_path = 'yolov3-tiny-{}.trt'.format(inputSize)
    h, w = (inputSize,inputSize)
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (inputSize, inputSize)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    
    # Output shapes expected by the post-processor
    output_shapes = [(1, 255, 13, 13), 
                     (1, 255, 26, 26)]

    # Do inference with TensorRT
    cuda.init()  # Initialize CUDA
    ctx = make_default_context()  # Create CUDA context
    
    postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)],
                          "obj_threshold": 0.4, 
                          "nms_threshold": 0.5,
                          "yolo_input_resolution": input_resolution_yolov3_HW}
    postprocessor = PostprocessYOLO(**postprocessor_args)
    
    
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        print("performing inference")
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        while True:
            
            trt_outputs = []
            #image_raw=vs.read()
            ret, image_raw = cap.read()
            if image_raw is not None:
                image_raw, image = preprocessor.process(image_raw)
                shape_orig_WH = image_raw.size
                inputs[0].host = image
                t0=time.time()
                trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                
                
                trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
                
                
                boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH),0)
                t1=time.time()
                t_inf = t1-t0
                fps = 1 / t_inf
                draw = True
                if(boxes is None):
                    print("no bboxes")
                    draw = False
                if(classes is None):
                    print("no classes")
                    draw = False
                if(scores is None):
                    print("no scores")
                    draw = False
                if draw :    
                    obj_detected_img = draw_bboxes(image_raw,bboxes=boxes, confidences=scores, categories=classes, all_categories=ALL_CATEGORIES)
                else:
                    obj_detected_img = image_raw
    #now stream this image
                
                with lock:
                    outputFrame = np.array(obj_detected_img)
                
    ctx.pop()

#-------------------------------------------------------------------------------------------------------#
    
@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")


@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
        




    
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, default="0.0.0.0",
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, default=8000,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-s", "--size", type=int, default=416,
		help="size of Yolo input image")
	args = vars(ap.parse_args())
	print(args["ip"])
	print(args["port"])
	print(args["size"])
    # start a thread that will perform motion detection
	t = threading.Thread(target=main, args=(args["size"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
#vs.stop()
