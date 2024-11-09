# python -m streamlit run src/interface/app.py --logger.level=debug
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from PIL import Image
import torch
import numpy as np
import sys
import os
import traceback

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.append(src_dir)
    print(f"Added to path: {src_dir}")

from src.models.pconv_unet import PConvUNet
from src.models.vgg16_extractor import VGG16Extractor
from src.utils_helper.image_helpers import UnNormalize
from src.utils_helper.state_dict_converter import convert_state_dict
from torchvision import transforms
from src.metrics.psnr_metric import PSNR_Metric
from src.loss.loss_compute import LossCompute
from src.interface.components.ui_components import UIComponents
from src.interface.components.canvas_handler import CanvasHandler

class InpaintingApp:
    def __init__(self):
        print("Initializing InpaintingApp...")
        self.canvas_size = 512
        
        # Initialize components
        self.ui = UIComponents()
        self.canvas_handler = CanvasHandler(canvas_size=self.canvas_size)
        
        # Setup device and UI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ui.setup_styles()
        st.sidebar.write(f"Using device: {self.device}")
        
        # Load models and initialize transforms
        self.model = self.load_model()
        self.vgg = self.load_vgg()
        self.setup_transforms()
        self.setup_metrics()
        
        print("Initialization complete.")

    def setup_transforms(self):
        """Initialize image transforms"""
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def setup_metrics(self):
        """Initialize metrics and loss computation"""
        self.psnr_metric = PSNR_Metric(device=self.device)
        self.loss_compute = LossCompute(self.vgg, {
            "loss_hole": 6.0,
            "loss_valid": 1.0,
            "loss_perceptual": 0.05,
            "loss_style_out": 120.0,
            "loss_style_comp": 120.0,
            "loss_tv": 0.1
        }, device=self.device)

    def load_model(self):
        try:
            print("Loading PConvUNet model...")
            model = PConvUNet()
            weights_path = os.path.join(src_dir, "weights", "pconv", "unet", "model_weights.pth")
            print(f"Loading weights from: {weights_path}")
            
            if not os.path.exists(weights_path):
                st.error(f"Model weights not found at: {weights_path}")
                return None
                
            old_state_dict = torch.load(weights_path, map_location=self.device)
            new_state_dict = convert_state_dict(old_state_dict)
            model.load_state_dict(new_state_dict)
            model.to(self.device)
            model.eval()
            print("Model loaded successfully.")
            return model
        except Exception as e:
            self.ui.handle_errors(e)
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
            self.ui.handle_errors(e)
            return None

    def process_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        try:
            if self.model is None:
                st.error("Model not loaded properly. Cannot process image.")
                return None

            print("Processing image...")
            # Resize input
            image = image.resize((self.canvas_size, self.canvas_size))
            
            # Prepare image and mask
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            mask = 1 - (mask > 0).astype(np.float32)  # Invert the mask
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = mask_tensor.repeat(1, 3, 1, 1).to(self.device)

            # Process image
            with torch.no_grad():
                output = self.model((img_tensor, mask_tensor))
                
            # Calculate and display metrics
            metrics = self.calculate_metrics(output, img_tensor, mask_tensor)
            self.ui.display_metrics(metrics)
            
            # Convert output tensor to image
            output_image = self.unorm(output.cpu())[0]
            print("Image processing complete.")
            return Image.fromarray(output_image)
            
        except Exception as e:
            self.ui.handle_errors(e)
            return None

    def calculate_metrics(self, output, img_tensor, mask_tensor):
        """Calculate PSNR and loss metrics"""
        metrics = {
            'psnr': self.psnr_metric.compute_psnr(output.cpu(), img_tensor.cpu()).item()
        }
        loss_fn = self.loss_compute.loss_total(mask_tensor)
        total_loss, loss_dict = loss_fn(img_tensor, output)
        metrics.update(loss_dict)
        return metrics

    def run(self):
        st.title("Image Inpainting with Partial Convolutions")
        
        try:
            # Display instructions and create controls
            self.ui.display_instructions()
            controls = self.ui.create_sidebar_controls()
            
            # File uploader
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                try:
                    input_image = Image.open(uploaded_file)
                    
                    # Create display columns and get mask
                    mask, resized_image = self.canvas_handler.display_canvases(input_image, controls)
                    
                    # Process the image if mask is valid
                    if mask is not None:
                        if self.canvas_handler.validate_mask(mask):
                            if st.button("Process Image"):
                                with st.spinner("Processing..."):
                                    result = self.process_image(resized_image, mask)
                                    if result is not None:
                                        st.markdown("""
                                            <div style='text-align: center; font-size: 1.2rem; font-weight: bold; margin: 1rem 0;'>
                                                Inpainting Result
                                            </div>
                                        """, unsafe_allow_html=True)
                                        st.image(result, use_container_width=True)
                
                except Exception as e:
                    self.ui.handle_errors(e)
                    print(traceback.format_exc())
                    
        except Exception as e:
            self.ui.handle_errors(e)

def main():
    print("Starting application...")
    app = InpaintingApp()
    app.run()

if __name__ == "__main__":
    main()