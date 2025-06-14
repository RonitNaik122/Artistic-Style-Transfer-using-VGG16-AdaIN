import torch
from torch.autograd import Variable
from collections import namedtuple
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import sys
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed_everything(42)  # for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mean and standard deviation used for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

""" Pretrained VGG16 Model """
class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

""" Transformer Net """
class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)

""" Components of Transformer Net """
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )

    def forward(self, x):
        return self.block(x) + x

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2), 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x

def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors

def test_transform(image_size=None):
    """ Transforms for test image """
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform

def test_image(image_path, checkpoint_model, save_path):
    """Apply style transfer to a single image"""
    os.makedirs(os.path.join(save_path, "results"), exist_ok=True)
    
    # Load and transform the image
    transform = test_transform()
    
    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    # Load model with CPU compatibility
    transformer.load_state_dict(torch.load(checkpoint_model, map_location=device))
    transformer.eval()
    
    # Prepare input
    image = Image.open(image_path).convert('RGB')
    image_tensor = Variable(transform(image)).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()
    
    # Save image
    fn = os.path.basename(checkpoint_model).split('.')[0]
    output_path = os.path.join(save_path, f"results/{fn}-output.jpg")
    torch.save(stylized_image, output_path.replace('.jpg', '.pt'))
    
    # Convert to image format and save
    stylized_image_np = stylized_image[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255
    stylized_image_np = stylized_image_np.astype(np.uint8)
    
    # Save as JPEG
    Image.fromarray(stylized_image_np).save(output_path)
    print(f"Image saved to: {output_path}")
    
    return output_path

def test_video(video_path, checkpoint_model, save_path, max_dim=None):
    """
    Apply style transfer to a video
    
    Args:
        video_path: Path to the input video
        checkpoint_model: Path to the style transfer model checkpoint
        save_path: Directory to save the output
        max_dim: Maximum dimension (width or height) for resizing the video frames for processing.
                 None means no resizing. Lower values process faster but with lower quality.
    """
    os.makedirs(os.path.join(save_path, "results"), exist_ok=True)
    video_filename = os.path.basename(video_path).split('.')[0]
    model_name = os.path.basename(checkpoint_model).split('.')[0]
    output_video_path = os.path.join(save_path, f"results/{model_name}-{video_filename}.mp4")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
    
    # Calculate resize dimensions if max_dim is specified
    if max_dim is not None:
        if frame_width > frame_height:
            new_width = max_dim
            new_height = int(frame_height * (max_dim / frame_width))
        else:
            new_height = max_dim
            new_width = int(frame_width * (max_dim / frame_height))
        process_width, process_height = new_width, new_height
        print(f"Processing frames at reduced size: {process_width}x{process_height}")
    else:
        process_width, process_height = frame_width, frame_height
    
    # Define transform for video frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load model
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(checkpoint_model, map_location=device))
    transformer.eval()
    
    # Define video writer with proper codec
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Try 'avc1' or 'H264' if this doesn't work
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    start_time = time.time()
    frame_count = 0
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for processing if needed
        if max_dim is not None:
            process_frame = cv2.resize(frame_rgb, (process_width, process_height))
        else:
            process_frame = frame_rgb
            
        # Convert to PIL Image
        img_pil = Image.fromarray(process_frame)
        
        # Apply transform
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Stylize frame
        with torch.no_grad():
            stylized_tensor = transformer(img_tensor)
            stylized_tensor = denormalize(stylized_tensor).cpu()
            
            # Convert tensor to numpy array
            stylized_np = stylized_tensor.squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
            stylized_np = (stylized_np * 255).astype(np.uint8)
            
            # Resize back to original dimensions if needed
            if max_dim is not None:
                stylized_np = cv2.resize(stylized_np, (frame_width, frame_height))
                
            # Convert RGB back to BGR for OpenCV
            stylized_bgr = cv2.cvtColor(stylized_np, cv2.COLOR_RGB2BGR)
            
            # Write to output file
            out.write(stylized_bgr)
        
        # Show progress
        frame_count += 1
        if frame_count % 10 == 0 or frame_count == total_frames:
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
            estimated_total_time = total_frames / fps_processing if fps_processing > 0 else 0
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"Processed {frame_count}/{total_frames} frames "
                  f"({(frame_count/total_frames)*100:.1f}%) - "
                  f"Processing speed: {fps_processing:.2f} fps - "
                  f"Est. remaining time: {remaining_time/60:.1f} minutes")
    
    # Release resources
    cap.release()
    out.release()
    
    # Display final message
    total_time = time.time() - start_time
    print(f"\nStyle transfer complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average processing speed: {frame_count/total_time:.2f} fps")
    print(f"Stylized video saved to: {output_video_path}")
    
    return output_video_path