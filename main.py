"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


MODEL_INPUT_HEIGHT_WIDTH = 300
MODELS_HUMAN_LABEL_ID = 1

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, port=MQTT_PORT, keepalive=MQTT_KEEPALIVE_INTERVAL)
    return client

def update_frame_with_bounding_boxes(frame, boxes_array, frame_width, frame_height, prob_threshold):
    number_of_applied_boxes = 0
    for image_id, label, conf, x_min, y_min, x_max, y_max in boxes_array:
        if conf >= prob_threshold:
            # x_min, y_min, x_max, y_max are percentages of width and height
            # so calculate pixel position below
            start_point = (int(x_min * frame_width), int(y_min * frame_height))
            end_point = (int(x_max * frame_width), int(y_max * frame_height))
            # Box color needs to be reversed as CV2 takes BGR instead of RGB
            box_color = (0, 0, 255)
            if label == MODELS_HUMAN_LABEL_ID: 
                frame = cv2.rectangle(frame, start_point, end_point, box_color, 2)
                number_of_applied_boxes += 1

    return frame, number_of_applied_boxes

def preprocess_frame(frame):
    # Pre-process the frame
    processed_frame = cv2.resize(frame, (MODEL_INPUT_HEIGHT_WIDTH, MODEL_INPUT_HEIGHT_WIDTH))
    processed_frame = processed_frame.transpose((2,0,1))
    return processed_frame.reshape(1, *processed_frame.shape)

def execute_inference(network, processed_frame):
    network.exec_net(processed_frame)
    # Get the output of inference
    network.wait()
    return network.get_output()

def handle_image(args, network):
    inputFilePath = args.input
    image = cv2.imread(inputFilePath)
    image_height = image.shape[0]
    image_width = image.shape[1]
    processed_image = preprocess_frame(image)
    result = execute_inference(network, processed_image)
    update_tuple = update_frame_with_bounding_boxes(image, result[0][0], image_width, image_height, args.prob_threshold)
    inputFileName = inputFilePath[(inputFilePath.rfind('/') + 1):]
    cv2.imwrite("./outputs/" + inputFileName, update_tuple[0])
    
def infer_on_stream(args, client, network):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    net_input_shape = network.get_input_shape()

    # Handle the input stream
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)

    # Initialise people count variable
    total_people = 0
    person_count = 0
    person_in_frame_duration = 0
    time_person_entered_frame = 0
    # stat_change initially set to true to publish inital values
    stat_change = True
    diff_in_people_frame_count = 0
    
    # Loop until stream is over
    while cap.isOpened():

        # Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the image as needed
        p_frame = preprocess_frame(frame)

        # Start asynchronous inference for specified request
        network.exec_net(p_frame)

        # Wait for the result
        if network.wait() == 0:

            # Get the results of the inference request
            result = network.get_output()
            
            # Extract any desired stats from the results
            out_frame, people_in_frame_count = update_frame_with_bounding_boxes(frame, result[0][0], frame_width, frame_height, args.prob_threshold)

            if people_in_frame_count - person_count != 0:
                diff_in_people_frame_count += 1

                ''' If there is a difference between people_in_frame_count and person_count for more than a second 
                    then update the statistics. This will prevent one false classification affecting
                    the statistics '''
                if diff_in_people_frame_count > frames_per_second:
                    stat_change = True
                    # If there is an increase in the number of people in the frame increment total_people
                    if people_in_frame_count - person_count > 0:
                        total_people += people_in_frame_count - person_count

                    # if there are people in the frame and we haven't started our clock then start
                    if people_in_frame_count > 0 and person_in_frame_duration == 0:
                        time_person_entered_frame = time.time()
                    elif people_in_frame_count == 0:
                        # if there is no longer anyone in the frame then we should calculate how long they were there
                        person_in_frame_duration = time.time() - time_person_entered_frame

                    # Update the person count
                    person_count = people_in_frame_count

            else:
                diff_in_people_frame_count = 0

            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if stat_change:
                client.publish("person", json.dumps({
                    "count": person_count,
                    "total": total_people
                }))

                if person_in_frame_duration > 0:
                    client.publish("person/duration", json.dumps({"duration": person_in_frame_duration}))
                    person_in_frame_duration = 0
                    time_person_entered_frame = 0
                
                stat_change = False

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(out_frame)  
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break
        
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    
    # Initialize the Inference Engine
    network = Network()
    # Load inference model
    network.load_model(args.model, args.device, args.cpu_extension)
    # Perform inference on the input stream
    if ".jpg" in args.input or ".png" in args.input: 
        handle_image(args, network)
    elif ".mp4" in args.input:
        # Connect to the MQTT server
        client = connect_mqtt()
        infer_on_stream(args, client, network)
    else:
        raise Exception("Currently this file type is unsupported")


if __name__ == '__main__':
    main()
