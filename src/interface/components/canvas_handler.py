import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import traceback
import sys

class CanvasHandler:
    def __init__(self, canvas_size: int = 512):
        self.canvas_size = canvas_size

    def process_mask(self, canvas_result) -> np.ndarray:
        """Process canvas result to get the mask"""
        try:
            if canvas_result.image_data is None:
                return None
                
            # Extract mask from canvas (take first channel as they're all same for white)
            mask = canvas_result.image_data[:,:,0]
            
            # Debug print
            print("Mask shape:", mask.shape)
            print("Mask dtype:", mask.dtype)
            print("Mask unique values:", np.unique(mask))
            
            # Ensure binary mask (0 or 255)
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)
            
            return mask
        except Exception as e:
            print("Error in process_mask:", str(e))
            print(traceback.format_exc())
            st.error(f"Error processing mask: {str(e)}")
            st.code(traceback.format_exc(), language="python")
            return None

    def validate_mask(self, mask: np.ndarray) -> bool:
        """Validate if mask is suitable for inpainting"""
        try:
            if mask is None:
                return False
                
            # Debug print
            print("Validating mask:")
            print("Mask shape:", mask.shape)
            print("Mask dtype:", mask.dtype)
            print("Mask unique values:", np.unique(mask))
            
            # Convert mask to binary (0 or 1)
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Count white pixels
            white_pixels = int(np.sum(binary_mask))  # Convert to Python int
            total_pixels = self.canvas_size * self.canvas_size
            mask_area = white_pixels / total_pixels
            
            print(f"White pixels: {white_pixels}")
            print(f"Total pixels: {total_pixels}")
            print(f"Mask area: {mask_area}")
            
            # Validate mask coverage
            if white_pixels == 0:
                st.warning("Please draw a mask first!")
                return False
            if mask_area > 0.95:
                st.warning("Mask covers too much of the image. Please draw a smaller mask.")
                return False
                
            return True
        except Exception as e:
            print("Error in validate_mask:", str(e))
            print(traceback.format_exc())
            st.error(f"Error validating mask: {str(e)}")
            st.code(traceback.format_exc(), language="python")
            return False
    
    def combine_masks(self, mask1: np.ndarray, mask2: np.ndarray = None) -> np.ndarray:
        """
        Combine multiple masks if needed.
        Args:
            mask1: First mask
            mask2: Second mask (optional)
        Returns:
            Combined mask as a binary numpy array
        """
        try:
            if mask2 is None:
                return mask1
                
            # Convert masks to binary (0 or 1)
            mask1_binary = (mask1 > 0).astype(np.uint8)
            mask2_binary = (mask2 > 0).astype(np.uint8)
            
            # Combine masks using logical OR
            combined_binary = np.logical_or(mask1_binary, mask2_binary)
            
            # Convert back to 0-255 range
            return combined_binary.astype(np.uint8) * 255
            
        except Exception as e:
            print("Error in combine_masks:", str(e), file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            st.error(f"Error combining masks: {str(e)}")
            return mask1  # Return first mask if combination fails

    def display_canvases(self, input_image: Image.Image, controls: dict):
        """Create vertically stacked canvases"""
        try:
            print("Input image size:", input_image.size)
            print("Input image mode:", input_image.mode)
            
            with st.container():
                # Resize input image for display
                resized_image = input_image.resize((self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS)
                if resized_image.mode != 'RGB':
                    resized_image = resized_image.convert('RGB')
                
                # Section 1: Original Image
                st.markdown("""
                    <div style='text-align: center; font-size: 1.2rem; font-weight: bold; margin: 1rem 0; padding-top: 1rem;'>
                        Original Image
                    </div>
                """, unsafe_allow_html=True)
                
                st.image(resized_image, use_container_width=True)
                
                # Section 2: Drawing Canvas
                st.markdown("""
                    <div style='text-align: center; font-size: 1.2rem; font-weight: bold; margin: 1rem 0; padding-top: 1rem;'>
                        Draw Mask Here
                    </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Convert PIL Image to numpy array and ensure correct format
                    background_array = np.array(resized_image)
                    
                    # Ensure the background array is the correct type and shape
                    if background_array.dtype != np.uint8:
                        background_array = background_array.astype(np.uint8)
                    
                    # Create the canvas with explicit parameter types
                    canvas_result = st_canvas(
                        fill_color="rgb(255, 255, 255)",  # White fill
                        stroke_width=int(controls["stroke_width"]),  # Ensure integer
                        stroke_color="rgb(255, 255, 255)",  # White stroke
                        background_color=None,  # No background color when using image
                        background_image=background_array,  # Properly formatted numpy array
                        drawing_mode=str(controls["drawing_mode"]),  # Ensure string
                        height=self.canvas_size,
                        width=self.canvas_size,
                        key="canvas",
                    )
                    
                    # Process and display mask if it exists
                    mask = None
                    if canvas_result is not None and canvas_result.image_data is not None:
                        print("Canvas result shape:", canvas_result.image_data.shape)
                        print("Canvas result dtype:", canvas_result.image_data.dtype)
                        
                        mask = self.process_mask(canvas_result)
                        if mask is not None and np.sum(mask) > 0:
                            st.markdown("""
                                <div style='text-align: center; font-size: 1.2rem; font-weight: bold; margin: 1rem 0; padding-top: 1rem;'>
                                    Extracted Mask
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Create mask display
                            mask_display = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
                            mask_display[..., :] = mask[..., np.newaxis]  # Broadcast mask to all channels
                            st.image(mask_display, caption="White areas will be inpainted", use_container_width=True)
                    
                    return mask, resized_image
                    
                except Exception as e:
                    print(f"Error with canvas creation: {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    st.error(f"Error creating canvas: {str(e)}")
                    return None, None
            
        except Exception as e:
            print("Error in display_canvases:", str(e), file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            st.error(f"Error displaying canvases: {str(e)}")
            return None, None