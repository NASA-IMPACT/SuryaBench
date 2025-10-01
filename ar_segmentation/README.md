# Dataset Generator for Active Regions with Polarity Inversion Lines

## Description 

This project provides C++ and Python tools to generate:

1. **Binary maps** representing strong solar Active Regions (ARs) that contain Polarity Inversion Lines (PILs)
2. **Index files** indicating the locations of input FITS files and output HDF5 results

---

## Project Structure

The following files need to be modified based on your environment and data locations:

- `./scripts/fulldisklist.txt`  
  Specify the years and months for which you want to create binary maps.

- `config.yaml`  
  Update the `data_dir` field to point to the directory containing magnetogram files (FITS format).

- `./index_file/ar_index_creator.py`  
  Update the `file_path` variable to match the directory containing the generated binary maps (HDF5 format) created by SuryaBench.

- `./dataloader/ar.py`  
  example of dataloader using our binary maps and input data files (multi-channel 4096x4096 rasters)

```bash
ar_segmentation/  
├── main.cpp               # C++ code to generate binary maps  
├── config.yaml            # Configuration file (YAML format)  
├── dataloader/
│   └── ar.py  # Loads binary maps and 4096×4096 input raster images
├── scripts/  
│   ├── main_code_runner.sh    # Bash script to run main.cpp using Docker or local build  
│   └── fulldisklist.txt    # list of years and months in fits files  
├── index_file/  
│   └── ar_index_creator.py    # Python script to generate index  files for input/output paths   
├── src/  
│   ├── io.hpp  
│   └── utils.hpp          # Supporting header files  
├── docker/  
│   └── dockerfile         # Docker environment setup  
└── CMakeLists.txt         # Build system config  
```

## Features
- Creates solar AR segmentation maps from magnetogram rasters. Maps represent ARs with polarity inversion lines. 
- Docker or local options are available, with GPU and CPU options
- Iterates over folders, expects magnetogram rasters (Expected input raster size is 4096×4096.)
- AR with PIL detection is parametrized (configuration can be changed from `config.yaml`), more on the detection algorithm can be found in Khani et al. [https://arxiv.org/pdf/2508.17195].
- Python script for creating a data loader/index file is provided.


## Usage
### 0. **Build and Run Using Docker**
You can either install dependencies locally or use a Docker container.

Run the Docker container (with GPU support and volume mount):
```bash
docker build -t SuryaBench -f docker/dockerfile .
docker run --gpus all -it \
  --cap-add=SYS_PTRACE \
  -v /path/to/your/data:/data \
  --name SuryaBench_ar_seg \
  SuryaBench
```

Re-enter the running container (if it was stopped):
```bash
docker start SuryaBench_ar_seg
docker exec -it SuryaBench_ar_seg /bin/bash
```

Once inside the container, you can build and run the code as described in the steps below.

### 1. **Build the Project (CMake)**
```bash
cmake build .
cd build
make
```
### 2. Run the code
The program runs based on a folder name (e.g., a formatted date string including files), which is expected to contain input FITS files:
```bash
./SuryaBench 20220101
```

### 3. Using the Bash Script (rocommended)
You can run the full pipeline using the provided script:
```bash
./scripts/main_code_runner.sh
```

### 4. Generate Index Files
```bash
python ./index_file/ar_index_creator.py
```

### 5. Download and Train with Surya AR Segmentation Dataset
- Download the processed data from huggingface:  

For downloading index files,
```
hf download nasa-ibm-ai4science/surya-bench-ar-segmentation --repo-type dataset --local-dir surya_ar_segmentation --include "*.csv"
```
For downloading 4096x4096 binary maps,
```
hf download nasa-ibm-ai4science/surya-bench-ar-segmentation --repo-type dataset --local-dir surya_ar_segmentation --include "*.tar.gz"
```

- A link to train model using Surya/Unet https://github.com/NASA-IMPACT/Surya/tree/main/downstream_examples/ar_segmentation

## Requirements [Local Installation]

## Contact
Jinsu Hong [jhong36@gsu.edu]
Kang Yang [kyang30@student.gsu.edu]
Berkay Aydin [baydin2@gsu.edu]