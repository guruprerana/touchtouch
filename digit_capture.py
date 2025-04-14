# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
import pprint
import time
import datetime

import cv2

from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler

logging.basicConfig(level=logging.DEBUG)

# Print a list of connected DIGIT's
digits = DigitHandler.list_digits()
print("Connected DIGIT's to Host:")
pprint.pprint(digits)

# Connect to a Digit device with serial number with friendly name
digit = Digit("D20118", "Left Gripper")
digit.connect()

# Print device info
print(digit.info())

# Change LED illumination intensity
digit.set_intensity(Digit.LIGHTING_MIN)
time.sleep(1)
digit.set_intensity(Digit.LIGHTING_MAX)

# Change DIGIT resolution to QVGA
qvga_res = Digit.STREAMS["QVGA"]
digit.set_resolution(qvga_res)

# Change DIGIT FPS to 15fps
fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
digit.set_fps(fps_30)

# Grab single frame from DIGIT
filepath = "data/objects/styrofoam"

while True:
        frame = digit.get_frame()
        #if ref_frame is not None:
        #    frame = self.get_diff(ref_frame)
        cv2.imshow(f"Digit View", frame)
        if cv2.waitKey(1) == 27:
            digit.save_frame(f"{filepath}/digit_frame_{datetime.datetime.now()}.png")
cv2.destroyAllWindows()

#while True:
#    input("press enter to take frame")
#    frame = digit.save_frame(f"digit_frame_{datetime.datetime.now()}.png")
#print(f"Frame WxH: {frame.shape[0]}{frame.shape[1]}")

# Display stream obtained from DIGIT
#digit.show_view(frame)

# Disconnect DIGIT stream
digit.disconnect()

# Find a Digit by serial number and connect manually
digit = DigitHandler.find_digit("D20140")
pprint.pprint(digit)
cap = cv2.VideoCapture(digit["dev_name"])
cap.release()
