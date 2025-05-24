import argparse
import gc
import json
import os

# GPU-optimized environment settings for RTX 3070
# No CPU-forcing environment variables to allow GPU usage

import subprocess
import sys
import threading
import time
import traceback
import uuid
from enum import Enum

import queue
import cv2
from flask import Flask, request

import service.trans_dh_service

from h_utils.custom import CustomError
from y_utils.config import GlobalConfig
from y_utils.logger import logger

# GPU optimization: Check CUDA availability and set device
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"🚀 GPU Mode: Using {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        if hasattr(torch.backends, 'cudnn'):
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        # Optimize GPU memory usage
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, falling back to CPU")
except Exception as e:
    device = torch.device("cpu")
    print(f"⚠️  CUDA initialization failed: {e}, using CPU")

def get_args():
    parser = argparse.ArgumentParser(description="HeyGem Digital Human GPU Processing")
    parser.add_argument('--audio_path', type=str, default='example/audio.wav', help='path to audio file')
    parser.add_argument('--video_path', type=str, default='example/video.mp4', help='path to video file')
    parser.add_argument('--code', type=str, default='1004', help='processing code')
    parser.add_argument('--gpu', action='store_true', help='force GPU usage (default: auto-detect)')
    parser.add_argument('--cpu', action='store_true', help='force CPU usage')
    return parser.parse_args()

def main():
    opt = get_args()
    
    # Handle device preference
    if opt.cpu:
        print("🔄 CPU mode forced by user")
        # Set CPU-only environment variables if requested
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif opt.gpu:
        print("🚀 GPU mode forced by user")
        if not torch.cuda.is_available():
            print("❌ GPU requested but not available!")
            return
    
    if not os.path.exists(opt.audio_path):
        audio_url = "example/audio.wav"
        print(f"⚠️  Audio path not found, using default: {audio_url}")
    else:
        audio_url = opt.audio_path

    if not os.path.exists(opt.video_path):
        video_url = "example/video.mp4"
        print(f"⚠️  Video path not found, using default: {video_url}")
    else:
        video_url = opt.video_path
    
    sys.argv = [sys.argv[0]]
    
    try:
        print("🔄 Initializing TransDhTask...")
        task = service.trans_dh_service.TransDhTask()
        print("✅ TransDhTask initialized successfully")
        
        if torch.cuda.is_available() and not opt.cpu:
            print("⏱️  GPU processing - should be faster than CPU...")
            time.sleep(5)  # Shorter wait for GPU
        else:
            print("⏱️  CPU processing - waiting for initialization...")
            time.sleep(15)  # Longer wait for CPU
        
        code = opt.code
        print(f"🚀 Starting processing:")
        print(f"   Audio: {audio_url}")
        print(f"   Video: {video_url}")
        print(f"   Code: {code}")
        print(f"   Device: {'GPU' if torch.cuda.is_available() and not opt.cpu else 'CPU'}")
        
        start_time = time.time()
        task.work(audio_url, video_url, code, 0, 0, 0, 0)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"✅ Processing completed successfully!")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        # Check for output files
        output_files = []
        possible_outputs = [
            f"{code}-t.mp4",
            f"{code}_format.wav",
            f"{code}_format.mp4",
            f"{code}.mp4",
            f"{code}.wav"
        ]
        
        for file_path in possible_outputs:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                output_files.append(f"📁 {file_path} ({file_size} bytes)")
        
        if output_files:
            print("📂 Generated files:")
            for file_info in output_files:
                print(f"   {file_info}")
        else:
            print("⚠️  No output files found in current directory")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        traceback.print_exc()
        return
    
    print("🎉 Main function completed successfully!")

if __name__ == "__main__":
    main() 