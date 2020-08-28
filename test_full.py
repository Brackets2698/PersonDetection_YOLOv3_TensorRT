from __future__ import print_function
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import imutils
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from PIL import ImageDraw,Image
from PIL import ImageFont
from PIL import Image
from pycuda.tools import make_default_context
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
#from data_processing_tiny import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import os
import common
import onnxruntime as rt
import json
import torch

from torchvision import transforms as T
from net import *

"""transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])"""

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#torch.set_default_dtype(torch.float)
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

"""def load_network(network):
    print("Loading network")
    save_path = "checkpts/market/resnet50/net_last.pth"
    print("Path set")
    network.load_state_dict(torch.load(save_path))
    
    return network
"""

class predict_decoder(object):

    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)['market']
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)['market']
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred,dictionary):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
             #   print('{}: {}'.format(name, chooce[pred[idx]]))
                dictionary[name] = chooce[pred[idx]]
    
                
sess = None 
input_name = None
label_name = None
outputFrame = None
lock = threading.Lock()

app = Flask(__name__)
#vs = VideoStream(src="/dev/video0").start()
cap = cv2.VideoCapture('video1.mp4')

TRT_LOGGER = trt.Logger()
t0=time.time()
t1=time.time()
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
    global t0,t1,fps,sess,inout_name,label_name
    
    base = image_raw.convert('RGBA')
    #draw = ImageDraw.Draw(image_raw)
    
    layer = Image.new('RGBA',base.size,(255,255,255,0))
    draw = ImageDraw.Draw(layer)
    i = 0
    
    for box, score, category in zip(bboxes, confidences, categories):
        if(category == 0):
            
            out_dict={
                'gender':None,
                'hair length':None,
                'sleeve length':None,
                'length of lower-body clothing':None,
                'type of lower-body clothing':None,
                'wearing hat':None,
                'carrying backpack':None,
                'carrying bag':None,
                'carrying handbag':None,
                'age':None,
                'color of upper-body clothing':None,
                'color of lower-body clothing':None
            }
            
            x_coord, y_coord, width, height = box
            left = max(0, np.floor(x_coord + 0.5).astype(int))
            top = max(0, np.floor(y_coord + 0.5).astype(int))
            right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
            bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))
            #Detect Attributes here
            
            if(i == i):
                src = image_raw.crop((left-50,top-20,right+50,bottom+20))
                src = transforms(src)
                src = src.unsqueeze(dim=0)
                src = src.numpy()
            
            
                #out = sess.run([label_name], {input_name: src})[0]
                out = sess.run([label_name], {input_name: src.astype(np.float32)})[0]
            
                #out = model.forward(src)
                out = torch.from_numpy(out)
                pred = torch.gt(out, torch.ones_like(out)*0.4)  # threshold=0.5

                Dec = predict_decoder('market')
                Dec.decode(pred,out_dict)
                
                #print(out_dict)
                
            i = (i+1)%5
            if(out_dict['wearing hat']=='yes' or out_dict['carrying bag']=='yes' or out_dict['carrying backpack']=='yes' or out_dict['carrying handbag']=='yes'):
                bbox_color = 'blue'
                dangerous = True
            else:
                bbox_color = 'red'
                dangerous = False
            #Test before drawing
            if(dangerous):
                draw.rectangle(((left, top), (right, bottom)), outline=bbox_color,fill=(0,0,255,64))
            else:
                draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
            
    font = ImageFont.truetype("openSans.ttf", 28)
    
    draw.text((2, 30), '{:.2f}FPS'.format(fps), fill='white', font=font)
    
    image_raw = Image.alpha_composite(base, layer).convert('RGB')
    
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
            last_layer = network.get_layer(network.num_layers - 1)
            network.mark_output(last_layer.get_output(0))
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
    #Load PAR model
    
    
    
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    global vs, outputFrame, lock,t0,t1,fps,sess,input_name,label_name
    
    #model = ResNet50_nFC(30)

    #model = load_network(model)

    #torch.save(model.state_dict(), "model")
    #device = torch.device('cuda')
    #model.to(device)
    #model.eval()
    
    
    sess_options = rt.SessionOptions()

    # Set graph optimization level
    #sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    #sess_options.optimized_model_filepath = "resnet50_nFC.onnx"
            
    #sess = rt.InferenceSession("resnet50_nFC.onnx", sess_options)
    sess = rt.InferenceSession("PAR.onnx")
    #sess.set_providers(['CUDAExecutionProvider'])
    #sess.set_providers(['CPUExecutionProvider'])

            
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    
    cuda.init()
    device = cuda.Device(0)
    onnx_file_path = 'yolov3-{}.onnx'.format(inputSize)
    engine_file_path = 'yolov3-{}.trt'.format(inputSize)
    h, w = (inputSize,inputSize)
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (inputSize, inputSize)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    
    # Output shapes expected by the post-processor
    output_shapes = [(1, 255, h // 32, w // 32),
                     (1, 255, h // 16, w // 16),
                     (1, 255, h //  8, w //  8)]
    
    """output_shapes = [(1, 255, 13, 13), 
                     (1, 255, 26, 26)]"""

    # Do inference with TensorRT
    cuda.init()  # Initialize CUDA
    ctx = make_default_context()  # Create CUDA context
    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.5,                                              
                          "nms_threshold": 0.35,                                              
                          "yolo_input_resolution": input_resolution_yolov3_HW}
    
    """postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)],
                          "obj_threshold": 0.4, 
                          "nms_threshold": 0.5,
                          "yolo_input_resolution": input_resolution_yolov3_HW}"""

     
    postprocessor = PostprocessYOLO(**postprocessor_args)
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        print("performing inference")
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        while True:
            trt_outputs = []
            #image_raw=vs.read()
            T0 = time.time()
            
            ret, image_raw = cap.read()
            if image_raw is not None:
                image_raw, image = preprocessor.process(image_raw)
                shape_orig_WH = image_raw.size
                inputs[0].host = image
                T1 = time.time()
                t0 = time.time()
                trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                
                
                trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
                T2 = time.time()
                #here we have Yolo output
                
                boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
                t1 = time.time()
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
                T3 = time.time()
                total = T3 - T0
                """print("Total time per frame: {:.3f}s (~{:.2f}FPS)".format(total,1/total))
                print("Pre process: {:.2f}%".format((T1-T0)/total))
                print("Inference: {:.2f}%".format((T2-T1)/total))
                print("Post process: {:.2f}%".format((T3-T2)/total))"""
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
