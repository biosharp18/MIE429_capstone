import cv2
import os

# Parameters
image_folder = 'vizs/101m'  # Folder containing PNGs
output_video = 'demo_crack.mp4'
frame_rate = 25  # Frames per second

# Get sorted list of image files
images = sorted([img for img in os.listdir(image_folder) if img.endswith('.png')])
if len(images) == 0:
    raise ValueError("No PNG images found in the specified folder.")

# Read the first image to determine the frame size
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape
size = (width, height)

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video, fourcc, frame_rate, size)

# Add images to the video
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    out.write(frame)

# Release the video writer
out.release()
print(f"Video saved as {output_video}")
