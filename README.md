# VOS Annotation App

A web-based video object segmentation (VOS) annotation application that uses SAM-3 for mask initialization and XMem for tracking.

## Overview

This application provides an interactive interface for:
- Uploading and processing videos
- Initializing masks using SAM-3 with text prompts
- Tracking objects across video frames using XMem
- Correcting and refining masks interactively
- Exporting annotation results

## Prerequisites

### System Requirements
- Python 3.11 or higher (3.11+ recommended)
- Node.js 16+ and npm
- CUDA-capable GPU (recommended for SAM-3 and XMem)
- **ffmpeg** (for video processing) - must be installed or loaded as a module

### External Dependencies
- **SAM-3**: Segment Anything Model 3 (must be installed as a Python package)
- **XMem**: XMem tracking repository (must be cloned to `./XMem` directory)
- **XMem Model**: Download XMem.pth model file to `./XMem/saves/XMem.pth`

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vos-annoation_app
```

### 2. Load Required Modules (Cluster/HPC Systems)

On cluster systems, you may need to load modules first. **Python 3.11 or higher is recommended:**

```bash
# Check available Python modules
module avail python
# or
module spider python

# Load Python module (use 3.11 or higher if available)
module load python/3.11
# or
module load python/3.12
# or
module load python3

# Load ffmpeg module (REQUIRED for video processing)
module load ffmpeg

### 3. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install PyTorch (Required for SAM-3)

SAM-3 requires PyTorch. Install it before installing SAM-3:

```bash
# Upgrade pip first
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
# Note: PyTorch 2.7.0 may not exist - check PyTorch website for latest version
# Adjust CUDA version (cu126, cu121, etc.) based on your system
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# If the above fails, try without version constraint:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# Or check https://pytorch.org/get-started/locally/ for the correct command
```

### 5. Install SAM-3

SAM-3 must be installed as a Python package. Clone and install it:

```bash
# Clone SAM-3 repository
# Note: On clusters, git may require module loading or authentication
# If git is not available, download as ZIP: wget https://github.com/facebookresearch/segment-anything-3/archive/refs/heads/main.zip
git clone https://github.com/facebookresearch/segment-anything-3.git sam3
cd sam3
pip install -e .
cd ..
```

**Note**: If SAM-3 uses a gated model, you'll need a Hugging Face token for the first download. Set it as an environment variable:

```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

### 6. Install XMem

Clone the XMem repository, download the model, and install dependencies:

```bash
# Clone XMem repository
git clone https://github.com/hkchengrex/XMem.git

# Download XMem model
mkdir -p XMem/saves
cd XMem/saves
# Download the model file from the XMem repository releases
# Check https://github.com/hkchengrex/XMem/releases for the download link
# For example:
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
# Or download it manually from the releases page
cd ../..

# Install XMem dependencies
cd XMem
pip install -r requirements.txt
cd ..

# Verify installation
ls -la XMem/                    # Should show the XMem directory
ls -la XMem/saves/XMem.pth      # Should show the model file
ls -la XMem/eval.py             # Should show the eval script
```

**Note**: The XMem repository must be in the same directory as `server.py` (i.e., `./XMem/`), and the model must be at `./XMem/saves/XMem.pth`.

### 7. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 8. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## Configuration

### Environment Variables

- `HUGGINGFACE_HUB_TOKEN`: (Optional) Hugging Face token for downloading gated SAM-3 models. Only needed for first download if model is not cached.

### Path Configuration

The following paths are configured in `server.py` and may need adjustment:

- `RUNS_ROOT`: Default is `/scratch/project_2016918/vos_annotation_runs` - change this to your preferred storage location
- `XMEM_REPO`: Default is `./XMem` - should point to your XMem repository
- `XMEM_MODEL`: Default is `./XMem/saves/XMem.pth` - should point to your XMem model file

### Backend URL

The frontend connects to the backend at `http://127.0.0.1:12212` by default. To change this, edit `frontend/src/api.js`:

```javascript
const BACKEND = "http://127.0.0.1:12212";
```

## Running the Application

### 1. Load ffmpeg Module (if on cluster/HPC system)

```bash
module load ffmpeg
```

### 2. Start the Backend Server

Make sure your virtual environment is activated, then run:

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 12212 --log-level info
```

Or for development with auto-reload:

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 12212 --log-level info --reload
```

### 3. Start the Frontend Development Server

In a separate terminal:

```bash
cd frontend
npm run dev
```

The frontend will typically run on `http://localhost:5173` (Vite default port).

### 4. Access the Application

Open your browser and navigate to the frontend URL (e.g., `http://localhost:5173`).

## Usage

1. **Upload Video**: Drag and drop a video file or select one from the interface
2. **Initialize with SAM**: Enter a text prompt (e.g., "cow") and click "Initialize with SAM"
3. **Assign IDs**: Review detected masks and assign IDs to objects
4. **Track**: Click "Track" to track objects across all frames using XMem
5. **Correct**: Use the correction tools to refine masks interactively
6. **Export**: Download the final annotations

## Project Structure

```
vos-annoation_app/
├── server.py              # FastAPI backend server
├── requirements.txt       # Python dependencies
├── frontend/             # React frontend application
│   ├── src/
│   │   ├── api.js        # API client (backend URL configured here)
│   │   ├── App.jsx       # Main application component
│   │   └── pages/        # Page components
│   ├── package.json      # Frontend dependencies
│   └── vite.config.js    # Vite configuration
├── XMem/                 # XMem tracking repository (must be cloned)
│   └── saves/
│       └── XMem.pth      # XMem model file (must be downloaded)
└── sam3/                 # SAM-3 repository (must be cloned and installed)
```

## Dependencies

### Python Dependencies
- `fastapi>=0.104.0` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `numpy>=1.24.0` - Numerical computing
- `opencv-python>=4.8.0` - Image/video processing
- `Pillow>=10.0.0` - Image processing
- `torch>=2.0.0` - PyTorch
- `torchvision>=0.15.0` - PyTorch vision utilities
- `huggingface-hub>=0.19.0` - Hugging Face model access
- `python-multipart>=0.0.6` - File upload support

### Frontend Dependencies
- `react^18.3.1` - React framework
- `react-dom^18.3.1` - React DOM
- `vite^5.4.2` - Build tool
- `@vitejs/plugin-react^4.3.1` - Vite React plugin

## Troubleshooting

### SAM-3 Import Errors
- Ensure SAM-3 is installed as a package: `pip install -e sam3/`
- Check that the SAM-3 directory contains the `sam3` package
- Verify Python can import: `python -c "from sam3.model_builder import build_sam3_image_model"`

### XMem Errors
- Verify XMem repository is cloned to `./XMem`
- Check that `XMem/saves/XMem.pth` exists
- Ensure XMem dependencies are installed in the XMem repository

### GPU Issues
- Verify CUDA is installed and accessible: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU memory availability
- SAM-3 and XMem will fall back to CPU if GPU is unavailable (much slower)

### Port Conflicts
- Change backend port: `uvicorn server:app --port <different-port>`
- Update frontend `BACKEND` URL in `frontend/src/api.js`
- Change frontend port in `frontend/vite.config.js`

### Storage Issues
- Update `RUNS_ROOT` in `server.py` to a location with sufficient space
- Videos and extracted frames can take significant disk space

## Development

### Backend Development
- The server uses uvicorn with auto-reload for development
- Logs are output to the console
- API endpoints are documented at `http://localhost:12212/docs` (FastAPI auto-docs)

### Frontend Development
- Uses Vite for fast hot module replacement
- React components are in `frontend/src/`
- API client is in `frontend/src/api.js`

## License

[Add your license information here]

## Acknowledgments

- SAM-3: Facebook Research
- XMem: HK Chengrex et al.
