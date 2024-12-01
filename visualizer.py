import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
from sewer_image_classifier import ResNetModel, SewerMLClassifier
from torchvision.models import resnet18, ResNet18_Weights
resnet_model = resnet18(pretrained=False)
model = ResNetModel(resnet_model)





# Transformation to process frames (assuming 224x224 input size)



weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
# Function to predict on a single frame
def predict(frame, model):
    # Preprocess the frame
    #[3, 288, 352]
    frame=frame.transpose(2,0,1)
    tensor = preprocess(torch.from_numpy(frame).to("cuda")).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(tensor)
        output = nn.functional.softmax(output,1)
    # Convert output to binary prediction (0 or 1)
    
    prediction = output[0,1]
    return prediction.item()

# Function to process video and store predictions
# Function to process video and store predictions with downsampling
def process_video(video_path, model, frame_skip_interval=2):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    predictions = []
    timestamps = []
    frames = []
    frame_count = 0

    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Downsample: Skip frames based on the frame_skip_interval
        if frame_count % frame_skip_interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get timestamp in seconds
            prediction = predict(frame, model)
            predictions.append(prediction)
            timestamps.append(timestamp)
            #Save plt figure of prediction progress
            #Save image
            frames.append(frame)
        frame_count += 1

    cap.release()
    return timestamps, predictions, frame_rate, frames

# Function to visualize the predictions
def visualize_predictions(timestamps, predictions, frame_rate, save_name):
    # Create timebar
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, predictions, color='blue', linewidth=3)
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.xlabel("Time (s)")
    plt.title("Predicted probability of defect")
    plt.savefig(save_name)

# Main function
def main():
    ckpt = torch.load('/scratch/ssd004/scratch/gaorory/best-model-epoch=2-val_loss=0.50.ckpt')

    #ckpt = torch.load('/checkpoint/gaorory/14016259/best-model-epoch=4-val_loss=0.47.ckpt')
    #Checkpoint saving is fucked for some reason. Fix the keys:
    # Create a new state dict with the updated keys
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove the 'model.model.' prefix if it exists
        if key.startswith('model.model.'):
            new_key = key[len('model.'):]  # Remove the prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.to("cuda")
    model.eval()  # Set model to evaluation mode\

    video_path = 'videos/defect_101m.mp4'
    save_name = f"pretrained_{video_path.replace("/", "_")}.png"
    timestamps, predictions, frame_rate, frames = process_video(video_path=video_path, model=model)
    visualize_predictions(timestamps, predictions, frame_rate, save_name)
    #Save plot viz for every frame:
    start_ind = 100
    end_ind = 600
    timestamps_sub = timestamps[start_ind:end_ind]
    predictions_sub = predictions[start_ind:end_ind]
    frames_sub = frames[start_ind:end_ind]
    for i,_ in enumerate(predictions_sub):
        print(i)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5), facecolor='black')
        ax[0].set_facecolor('black')
        ax[0].plot(timestamps_sub, predictions_sub, color='#7DDA58', linewidth=2)
        ax[0].plot(timestamps_sub[i], predictions_sub[i], color='red', marker="o", markersize=10)
        ax[0].set_yticks([0, 1], ['Negative', 'Positive'])
        ax[0].tick_params(axis='y', colors='white', labelsize=15)
        ax[0].tick_params(axis='x', colors='white', labelsize=15)
        ax[0].set_xlabel("Time (s)", color='white', fontsize=15)
        ax[0].set_title("Predicted probability of defect", color='white', fontsize=15)
        ax[1].imshow(frames_sub[i])
        plt.savefig(f"vizs/101m/frame_{i:04d}.png", dpi=100)
        plt.close()
if __name__ == "__main__":
    main()