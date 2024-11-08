#cd D:\LAST_TRY\Image-Inpainting-master
# python -m streamlit run src/interface/app.py

# src/interface/app.py
import streamlit as st
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.append(src_dir)
    print(f"Added to path: {src_dir}")

# Now import from src directory
from src.models.pconv_unet import PConvUNet
from src.models.vgg16_extractor import VGG16Extractor
from src.utils_helper.image_helpers import UnNormalize
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
from src.metrics.psnr_metric import PSNR_Metric
from src.loss.loss_compute import LossCompute

class InpaintingApp:
    def __init__(self):
        print("Initializing InpaintingApp...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.sidebar.write(f"Using device: {self.device}")
        
        self.model = self.load_model()
        self.vgg = self.load_vgg()
        self.canvas_size = 512
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Initialize metrics
        self.psnr_metric = PSNR_Metric()
        self.loss_compute = LossCompute(self.vgg, {
            "loss_hole": 6.0,
            "loss_valid": 1.0,
            "loss_perceptual": 0.05,
            "loss_style_out": 120.0,
            "loss_style_comp": 120.0,
            "loss_tv": 0.1
        }, device=self.device)
        print("Initialization complete.")

    def load_model(self):
        try:
            print("Loading PConvUNet model...")
            model = PConvUNet()
            weights_path = os.path.join(src_dir, "weights", "pconv", "unet", "model_weights.pth")
            print(f"Loading weights from: {weights_path}")
            
            if not os.path.exists(weights_path):
                st.error(f"Model weights not found at: {weights_path}")
                return None
                
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            print("Model loaded successfully.")
            return model
        except Exception as e:
            st.error(f"Error loading PConvUNet model: {str(e)}")
            print(f"Error details: {str(e)}")
            return None

    def load_vgg(self):
        try:
            print("Loading VGG model...")
            vgg = VGG16Extractor()
            vgg.to(self.device)
            vgg.eval()
            print("VGG loaded successfully.")
            return vgg
        except Exception as e:
            st.error(f"Error loading VGG model: {str(e)}")
            print(f"Error details: {str(e)}")
            return None

    def process_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        try:
            print("Processing image...")
            # Resize input
            image = image.resize((self.canvas_size, self.canvas_size))
            
            # Prepare image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Convert mask to the correct format (1 = keep, 0 = inpaint)
            mask = 1 - (mask > 0).astype(np.float32)  # Invert the mask
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = mask_tensor.repeat(1, 3, 1, 1).to(self.device)

            with torch.no_grad():
                output = self.model((img_tensor, mask_tensor))
                
            # Calculate metrics
            metrics = {}
            metrics['psnr'] = self.psnr_metric.compute_psnr(
                output.cpu(), 
                img_tensor.cpu()
            ).item()
            
            loss_fn = self.loss_compute.loss_total(mask_tensor)
            total_loss, loss_dict = loss_fn(img_tensor, output)
            metrics.update(loss_dict)
            
            # Convert output tensor to image
            output_image = self.unorm(output.cpu())[0]
            
            # Display metrics
            st.sidebar.write("### Metrics")
            st.sidebar.metric("PSNR", f"{metrics['psnr']:.2f} dB")
            for key, value in loss_dict.items():
                st.sidebar.metric(key, f"{value:.4f}")
            
            print("Image processing complete.")
            return Image.fromarray(output_image)
            
        except Exception as e:
            st.error(f"Error during image processing: {str(e)}")
            print(f"Processing error details: {str(e)}")
            return None

    def run(self):
        st.title("Image Inpainting with Partial Convolutions")
        
        # Sidebar controls
        st.sidebar.title("Controls")
        stroke_width = st.sidebar.slider("Brush width:", 1, 100, 30)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    input_image = Image.open(uploaded_file)
                    st.image(input_image, width=self.canvas_size)
                
                with col2:
                    st.subheader("Draw Mask")
                    canvas_result = st_canvas(
                        fill_color="rgb(255, 255, 255)",
                        stroke_width=stroke_width,
                        stroke_color="rgb(255, 255, 255)",
                        background_image=input_image.resize((self.canvas_size, self.canvas_size)),
                        height=self.canvas_size,
                        width=self.canvas_size,
                        drawing_mode="freedraw",
                        key="canvas",
                    )

                if canvas_result.image_data is not None:
                    # Extract mask from canvas
                    mask = canvas_result.image_data[:,:,0]
                    
                    # Add process button
                    if st.button("Process Image"):
                        if np.sum(mask) == 0:
                            st.warning("Please draw a mask first!")
                        else:
                            with st.spinner("Processing..."):
                                result = self.process_image(input_image, mask)
                                if result is not None:
                                    st.image(result, caption="Inpainting Result", width=self.canvas_size)
            
            except Exception as e:
                st.error(f"Error in main app loop: {str(e)}")
                print(f"Main loop error details: {str(e)}")

def main():
    print("Starting application...")
    app = InpaintingApp()
    app.run()

if __name__ == "__main__":
    main()