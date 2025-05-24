import argparse
import gc
import json
import os

# GPU-optimized environment settings for RTX 3070
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
# Remove CPU-forcing settings to allow GPU usage

import subprocess
import threading
import time
import traceback
import uuid
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import queue
import shutil
from functools import partial

import cv2
import gradio as gr
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
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, falling back to CPU")
except Exception as e:
    device = torch.device("cpu")
    print(f"⚠️  CUDA initialization failed: {e}, using CPU")

# Print current environment info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

class ProcessingStatusEnum(Enum):
    WAIT = "wait"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"

class VideoProcessor:
    def __init__(self):
        logger.info("VideoProcessor init")
        self.task = service.trans_dh_service.TransDhTask()
        # Use a default thread count since GlobalConfig.THREAD_COUNT doesn't exist
        thread_count = getattr(GlobalConfig, 'THREAD_COUNT', 4)  # Default to 4 threads
        self.executor = ThreadPoolExecutor(max_workers=thread_count)
        self.processing_status = ProcessingStatusEnum.WAIT
        self.output_video_path = None
        self.output_audio_path = None
        logger.info("VideoProcessor init finished")

    def process_video(self, audio_path, video_path, code, x_start, y_start, x_end, y_end):
        try:
            logger.info(f"🚀 GPU Processing: audio={audio_path}, video={video_path}, code={code}")
            self.processing_status = ProcessingStatusEnum.RUNNING
            
            # Process with GPU acceleration
            self.task.work(audio_path, video_path, code, x_start, y_start, x_end, y_end)
            
            self.processing_status = ProcessingStatusEnum.FINISHED
            logger.info("✅ GPU Processing completed successfully")
            return True
        except Exception as e:
            logger.error(f"❌ GPU Processing failed: {e}")
            self.processing_status = ProcessingStatusEnum.ERROR
            return False

    def get_result_files(self, code):
        """Get the generated result files"""
        try:
            video_file = f"{code}-t.mp4"
            audio_file = f"{code}_format.wav"
            
            if os.path.exists(video_file):
                self.output_video_path = video_file
            if os.path.exists(audio_file):
                self.output_audio_path = audio_file
                
            return self.output_video_path, self.output_audio_path
        except Exception as e:
            logger.error(f"Error getting result files: {e}")
            return None, None

def create_gradio_interface():
    processor = VideoProcessor()
    
    def process_files(audio_file, video_file, code_input):
        try:
            if not audio_file or not video_file:
                return "❌ Please upload both audio and video files", None, None
            
            code = code_input or "1004"
            logger.info(f"Starting GPU processing with code: {code}")
            
            # Process files
            success = processor.process_video(
                audio_file.name, 
                video_file.name, 
                code, 
                0, 0, 0, 0
            )
            
            if success:
                # Get result files
                video_result, audio_result = processor.get_result_files(code)
                return f"✅ Processing completed successfully! (GPU-accelerated)", video_result, audio_result
            else:
                return "❌ Processing failed", None, None
                
        except Exception as e:
            logger.error(f"Error in process_files: {e}")
            return f"❌ Error: {str(e)}", None, None
    
    # Create Gradio interface
    with gr.Blocks(title="🚀 HeyGem Digital Human (GPU-Accelerated)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 HeyGem Digital Human Generator (GPU-Accelerated)")
        gr.Markdown("### Upload audio and video files to generate synchronized digital human content")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.File(label="Audio File (.wav)", file_types=[".wav", ".mp3"])
                video_input = gr.File(label="Video File (.mp4)", file_types=[".mp4", ".avi"])
                code_input = gr.Textbox(label="Code", value="1004", placeholder="Enter processing code")
                process_btn = gr.Button("🚀 Process (GPU)", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(label="Status", interactive=False)
                video_output = gr.Video(label="Generated Video")
                audio_output = gr.Audio(label="Generated Audio")
        
        process_btn.click(
            fn=process_files,
            inputs=[audio_input, video_input, code_input],
            outputs=[status_output, video_output, audio_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    ) 