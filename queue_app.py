import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import logging as log

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument("--queue_param", type=str)
    parser.add_argument('--confidence_level', default=0.65, type=float)
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--write_video', default=False)
    parser.add_argument('--batch_size', default=False, type=int)
    
    args=parser.parse_args()

    return args

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        check = {}
        for ii,coord in enumerate(coords):
            for i, q in enumerate(self.queues):
                if coord[0]>int(q[0]) and coord[2]<int(q[2]):
                    d[i+1]+=1
                    check[ii] = True
        return d, check

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU"):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device, )

        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape
    
    
    def sync_inference(self, image):
        '''
        Makes a synchronous inference request, given an input image.
        '''
        pass

    
    def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        pass


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        pass


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs[self.output_blob]


# class objects created by delegation
class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, net, model, threshold=0.60, args=None):
        
        self.network = net
        self.model = model
        self.args = args
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def load_model(self, num_requests=1):
        '''
        TODO: This method needs to be completed by you
        '''
        
        self.net_input_shape = self.model.inputs[self.input_blob].shape
        
    def predict(self, net, batch_images, request_id=0):
        '''
        TODO: This method needs to be completed by you
        '''
        for i in range(len(batch_images)):
            if "retail" in self.args.video:
                status = self.network.requests[2*i+1].wait(-1)
            else:
                status = self.network.requests[i].wait(-1)
        
    def draw_outputs(self, request_id=0):
        '''
        TODO: This method needs to be completed by you
        '''
        return self.network.requests[request_id].outputs[self.output_blob]
        
    def preprocess_outputs(self, frame, outputs, confidence_level=0.4):
        '''
        TODO: This method needs to be completed by you
        '''
        height, width = frame.shape[:2]
        if (len(outputs) > 0) and (len(outputs[0][0]) > 0):
            boxes = []
            confs = []
            for res in outputs[0][0]:
                _, label, conf, xmin, ymin, xmax, ymax = res
                if conf > confidence_level:
                    xmin = int(xmin*width)
                    ymin = int(ymin*height)
                    xmax = int(xmax*width)
                    ymax = int(ymax*height)
                    boxes.append([xmin, ymin, xmax, ymax])
                    confs.append(conf)
        
        return boxes, confs
        
    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        frame = cv2.resize(image, (self.net_input_shape[3], self.net_input_shape[2]), 
                   interpolation = cv2.INTER_AREA)
        frame = np.moveaxis(frame, -1, 0)
        
        return frame

# singleton 
def get_network(args):
    model = args.model
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    start = time.time()

    core = IECore()
    if CPU_EXTENSION and "CPU" in args.device:
        core.add_extension(CPU_EXTENSION, args.device)
    model = IENetwork(model_structure, model_weights)
    net = core.load_network(network=model, device_name=args.device, num_requests=args.batch_size)
    
    return net, model

def preprocess(model, frame, net_input_shape):
    # Get the name of the input node
    input_name=next(iter(model.inputs))

    # Reading and Preprocessing Image
    frame=cv2.resize(frame, (net_input_shape[3], net_input_shape[2]), interpolation = cv2.INTER_AREA)
    frame=np.moveaxis(frame, -1, 0)

    # Running Inference in a loop on the same image
    input_dict={input_name:frame}
    
    return frame, input_dict, input_name

def preprocess_outputs(frame, args, result, confidence_level=0.65):
    '''
    TODO: This method needs to be completed by you
    '''
    boxes = []
    confs = []
    height, width = frame.shape[:2]
    if len(result[0][0]) > 0:
        for res in result[0][0]:
            _, label, conf, xmin, ymin, xmax, ymax = res
            if conf > confidence_level:
                xmin = int(xmin*width)
                ymin = int(ymin*height)
                xmax = int(xmax*width)
                ymax = int(ymax*height)
                boxes.append([xmin, ymin, xmax, ymax])
                confs.append(conf)
    
    return boxes, confs

def draw_boxes(frame, coords, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    c = (255,0,0)
    for box in coords: # Output shape is 1x1x100x7
        xmin = int(box[0] * width)
        ymin = int(box[1] * height)
        xmax = int(box[2] * width)
        ymax = int(box[3] * height)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), c, 2)
    return frame

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output=args.output
    confidence_level = args.confidence_level
    args.batch_size = int(args.batch_size)

    start_model_load_time = time.time()
    net, model = get_network(args)
    pd = PersonDetect(net, model, threshold, args)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue = Queue()

    try:
        if args.queue_param:
            args.queue_param=np.load(args.queue_param)
            queue_param=args.queue_param
            for q in queue_param:
                queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, (initial_w, initial_h), True)
    
    counter = 0
    start_inference_time = time.time()
    
    p = np.array([])
    s = []
    num_outputs = []
    batch_images = []
    f_batch_images = []
    batch_size = args.batch_size

    net_input_shape = model.inputs[pd.input_blob].shape
    
    print("Video len: ", video_len)

    try:
        while cap.isOpened():
            # Read the next frame
            flag, frame = cap.read()
            if not flag:
                break
            key_pressed = cv2.waitKey(60)

            ### TODO: Pre-process the frame
            p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)

            ### TODO: Perform inference on the frame
            net.start_async(request_id=0, inputs={pd.input_blob: p_frame})

            ### TODO: Get the output of inference
            if net.requests[0].wait(-1) == 0:
                result = net.requests[0].outputs[pd.output_blob]
                # boxes, confs = preprocess_outputs(p_frame, args, result)
                coords, confs = preprocess_outputs(frame, args, result)
                num_people, check = queue.check_coords(coords)
                print(f"Total People in frame = {len(coords)}")
                print(f"Number of people in queue = {num_people}")
                p = np.append(p, list(num_people.values()))
                s.append(sum(list(num_people.values())))
                for ii,box in enumerate(coords):
                    if ii in check:
                        xmin, ymin, xmax, ymax = box
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                y_pixel = 25
                out_text = ""
                for k, v in num_people.items(): # Output shape is 1x1x100x7
                    out_text += f"No. of People in Queue {k} is {v} "
                    frame = cv2.putText(frame, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
                    y_pixel += 30
                # Write out the frame
                out.write(frame)

            # Break if escape key pressed
            if key_pressed == 27:
                break

        total_time = time.time()-start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter/total_inference_time
        
        print("Max people: ", max(p.flatten().tolist()))
        print("Max people in all queues: ", max(s))
        print("Total counter: ", len(s))
        
        print("Total Inference Time: ", str(total_inference_time))
        print("FPS: ", str(fps))
        print("Total Model Load Time: ", str(total_model_load_time))

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        raise e
        print("Could not run Inference: ", e)

if __name__=='__main__':
    args = parse_args()
    
    if "manufacturing" in args.video:
        args.batch_size = 64
    elif "retail" in args.video:
        args.batch_size = 112
    elif "transportation" in args.video:
        args.batch_size = 64
    else:
        raise Exception("exception on queue param")

    main(args)