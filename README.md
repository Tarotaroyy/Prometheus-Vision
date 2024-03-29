# Prometheus-Vision

## Introduction
Prometheus-Vision is a project that utilizes depth cameras to capture depth values from the environment.
And to classify objects using vision techniques.

## Important 
- DO NOT CHANGE any file or file points relating to pybackend,pyrealsense, and pyrsutils. It is related to pyrealsense2 setup.

## Usage
### Running getDepthValues.py
To run the `getDepthValues.py` script, follow these steps:

1. **Grant Permission**: Since the script interacts with the depth camera, you may need to run it with elevated privileges. Use the `sudo` command as follows:
    sudo python3 getDepthValues.py

2. **Exit the Program**: Press the 'q' key while the program is running to exit gracefully.



## Dependencies
- Python 3.7
- pyrealsense2 (built from source, and environment is already set up)
- numpy
- tensorflow
- scipy


## Dataset used
- NYU Depth Dataset V2
- You may see PotentialDataSetToUse.md for training on more datasets
