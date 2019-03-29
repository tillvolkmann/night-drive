import os, re, sys
nightdrive_root = "/home/SharedFolder/git/tillvolkmann/night-drive"
sys.path.append(nightdrive_root)
from utils.eval_utils import get_CAM_resnet18
import cv2
import argparse
import glob
import subprocess
import numpy as np

"""
python3 .timeofday_cam.py --path --weights os.path.join(nightdrive_root,
                                      "classifier_timeofday/models/resnet18_timeofday_daynight_classifier_best.pth")
"""

if __name__ == "__main__":

    # Parse input:
    # Instantiate parser
    parser = argparse.ArgumentParser()
    # Arguments
    parser.add_argument('--path', type=str, required=True, help='Path to directory containing frames.')
    parser.add_argument('--weights', type=str, required=True, help='Path to model containing classification weights.')
    parser.add_argument('--suffix', default="", type=str, required=False,
                        help='Suffix identifying second version of frame, e.g. frame-1_suffix.')
    # Get arguments
    opt = parser.parse_args()
    out_dir = opt.path

    # Specifications:
    # confidence thresholds, set between 0 and 1 to enable module
    conf_thresh_classification = 0.5
    # mapping dict for weather predictions
    dict_timeoday = {
        0: "Time-of-day: Daytime",
        1: "Time-of-day: Night",
    }
    # suffix for tmp files and video
    suffix_cam = "cam"
    # specify device
    device = "cuda"

    # Process frames:
    # Get list of all frames to process, i.e. those that contain "opt.suffix"
    if opt.suffix == "":
        list_frames = [fn for fn in os.listdir(opt.path) if re.search(f"frame-{opt.suffix}[0-9]+", fn) is not None]  # actuallythis works for suffixes like empty (""), "transfer_AtoB-", etc, so the below option could be removed; and of course "opt.suffix" in the expression is redundant if the alternative conidtion is not removed
    else:
        list_frames = [fn for fn in os.listdir(opt.path) if opt.suffix in fn]
    if len(list_frames) == 0:
        raise Exception(f"No frames containing {opt.suffix} found in {opt.path}.")

    # Loop over frames
    num_frames = len(list_frames)
    for i, frame in enumerate(list_frames):

        # classify weather and get CAM
        pred_tod_class, pred_tod_score, cam_bgr = get_CAM_resnet18(frame, opt.weights,
                                                                   dict_timeoday, 2, device)
        # write timeofday on image
        timeofday_color = [(255, 255, 255) if pred_tod_score >= conf_thresh_classification else (80, 80, 80)]
        cv2.putText(cam_bgr,
                    f"{pred_tod_class} ({pred_tod_score:.2f})",
                    (5, 715),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    timeofday_color[0],
                    2)

        # write bgr (will be transformed to rgb by open cv)
        frame_out_name = os.path.basename(frame)
        frame_out_name, ext = os.path.splitext(frame_out_name)
        frame_out_name = re.sub(r"([0-9]+)", r"{}\1".format('cam-'), frame_out_name) + ext  # appends "cam" right before the frame counter
        cv2.imwrite(os.path.join(opt.path, frame_out_name),
                    cam_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Print progress message
        if i + 1 % 100 == 0:
            print("Cam-processed {} of {} frames.".format(i + 1, num_frames))