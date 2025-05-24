# 🚀 HeyGem Digital Human - GPU Optimization Project

A high-performance digital human video generation application with GPU acceleration support for NVIDIA RTX series graphics cards.

![GPU Status](https://img.shields.io/badge/GPU-RTX%203070%20Ti-green)
![CUDA](https://img.shields.io/badge/CUDA-11.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)

## 🎯 **Project Overview**

This project transforms a CPU-only digital human application into a dual-environment system supporting both CPU and GPU processing, achieving **40% performance improvement** on NVIDIA RTX hardware.

### **Key Features**
- 🚀 **GPU Acceleration**: CUDA 11.8 + PyTorch 2.0.1 optimization
- 🔄 **Dual Environment**: CPU (stable) + GPU (performance) options
- 🌐 **Web Interface**: Gradio-based UI with real-time processing
- 📱 **CLI Support**: Command-line tools for batch processing
- 🛡️ **Error Recovery**: Graceful fallbacks and comprehensive error handling

## 📊 **Performance Improvements**

| Metric | CPU Version | GPU Version | Improvement |
|--------|-------------|-------------|-------------|
| **Audio Processing** | 8.75s | 1.51s | **82% faster** |
| **Total Processing** | 18.99s | 11.71s | **38% faster** |
| **Model Loading** | 15s | 5s | **67% faster** |

## 🔧 **Hardware Requirements**

### **Minimum Requirements**
- **OS**: Linux Ubuntu 18.04+
- **Python**: 3.8+
- **RAM**: 8GB+
- **Storage**: 10GB free space

### **GPU Requirements (Optional)**
- **GPU**: NVIDIA RTX 20/30/40 series
- **VRAM**: 6GB+ recommended
- **CUDA**: 11.8+ compatible drivers
- **Driver**: NVIDIA 450.80.02+

## 🚀 **Quick Start**

### **1. Clone Repository**
```bash
git clone https://github.com/agilealpha1/AiVideo.git
cd AiVideo
```

### **2. CPU Environment Setup (Stable)**
```bash
# Create CPU environment
python -m venv venv
source venv/bin/activate

# Install CPU dependencies
pip install -r requirements_updated.txt

# Run CPU web interface
python app.py
# Access at: http://localhost:7860
```

### **3. GPU Environment Setup (Performance)**
```bash
# Create GPU environment
python -m venv venv_gpu
source venv_gpu/bin/activate

# Install GPU-optimized PyTorch
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements_gpu_fixed.txt

# Additional required packages
pip install einops typeguard==2.13.3

# Run GPU web interface
python app_gpu.py
# Access at: http://localhost:7861
```

## 📋 **Usage Guide**

### **Web Interface**
1. **Upload Files**: Audio (.wav) + Video (.mp4)
2. **Select Code**: Processing identifier (default: 1004)
3. **Choose Environment**:
   - CPU: `http://localhost:7860` (stable)
   - GPU: `http://localhost:7861` (faster)
4. **Process**: Click "Process" button
5. **Download**: Get generated video/audio files

### **Command Line**
```bash
# CPU processing
source venv/bin/activate
python run.py --audio_path audio.wav --video_path video.mp4

# GPU processing
source venv_gpu/bin/activate
python run_gpu.py --audio_path audio.wav --video_path video.mp4 --gpu

# Force CPU mode in GPU environment
python run_gpu.py --audio_path audio.wav --video_path video.mp4 --cpu
```

## 🏗️ **Project Structure**

```
HeyGem-Linux-Python-Hack/
├── 📁 CPU Environment
│   ├── app.py                    # CPU web interface
│   ├── run.py                   # CPU command line
│   └── requirements_updated.txt # CPU dependencies
├── 📁 GPU Environment  
│   ├── app_gpu.py              # GPU web interface
│   ├── run_gpu.py              # GPU command line
│   └── requirements_gpu_fixed.txt # GPU dependencies
├── 📁 Core Modules
│   ├── service/                # Core processing logic
│   ├── face_lib/              # Face detection/processing
│   ├── landmark2face_wy/      # Neural network models
│   └── y_utils/               # Utility functions
├── 📁 Configuration
│   ├── config/                # Application settings
│   └── example/               # Sample input files
└── 📁 Documentation
    ├── README.md              # This file
    └── .gitignore            # Git exclusions
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **CUDA Compatibility Error**
```
RuntimeError: Unexpected error from cudaGetDeviceCount()
```
**Solution**: Use CPU environment or update NVIDIA drivers

#### **Port Already in Use**
```
OSError: Cannot find empty port in range: 7860-7860
```
**Solution**: Check for running processes:
```bash
ps aux | grep python
kill <process_id>  # If needed
```

#### **Module Not Found Errors**
```
ModuleNotFoundError: No module named 'einops'
```
**Solution**: Install missing packages:
```bash
pip install einops typeguard==2.13.3
```

#### **Queue Timeout Warnings**
```
_queue.Empty: timeout
```
**Status**: Non-critical - processing continues, files still generated

### **Performance Optimization**

#### **GPU Memory Issues**
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### **Check GPU Status**
```bash
nvidia-smi
```

## 🧪 **Testing**

### **Verify Installation**
```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test web interface
curl http://localhost:7860  # CPU version
curl http://localhost:7861  # GPU version
```

### **Performance Benchmark**
```bash
# Compare processing times
time python run.py --audio_path example/audio.wav --video_path example/video.mp4
time python run_gpu.py --audio_path example/audio.wav --video_path example/video.mp4 --gpu
```

## 📦 **Dependencies**

### **Core Dependencies**
- `torch>=2.0.1` - Deep learning framework
- `gradio>=4.44.1` - Web interface
- `opencv-python>=4.7.0` - Computer vision
- `numpy>=1.21.6,<1.23.0` - Numerical computing
- `scipy>=1.7.1,<1.8.0` - Scientific computing

### **GPU-Specific**
- `torch==2.0.1+cu118` - CUDA-enabled PyTorch
- `onnxruntime-gpu==1.19.2` - GPU inference runtime
- `einops==0.8.1` - Tensor operations
- `typeguard==2.13.3` - Type checking

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Original HeyGem Digital Human team for the base application
- NVIDIA for CUDA toolkit and GPU optimization guides
- PyTorch team for GPU acceleration framework
- Gradio team for the excellent web interface framework

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/agilealpha1/AiVideo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/agilealpha1/AiVideo/discussions)
- **Documentation**: See `/docs` folder for detailed guides

---

**⭐ Star this repository if it helped you optimize your digital human processing!**

[![GPU Acceleration Demo](https://img.shields.io/badge/Demo-GPU%20Acceleration-success)](http://localhost:7861)
