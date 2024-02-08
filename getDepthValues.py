import pyrealsense2 as rs
import numpy as np
import cv2

# Helper function to retrieve depth data
def get_depth(depth_frame):
    depth_data = np.asanyarray(depth_frame.get_data())
    return depth_data

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable RGB stream

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Visualize depth map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Concatenate depth map and RGB image horizontally
        combined_image = np.hstack((depth_colormap, color_image))

        # Display combined image
        cv2.imshow('Depth and RGB', combined_image)

        # Get depth values
        depth_data = get_depth(depth_frame)

        # Print depth values using vectorized operation
        print(depth_data)

        # Break the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
