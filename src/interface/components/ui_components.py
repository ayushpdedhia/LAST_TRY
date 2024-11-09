import streamlit as st
import traceback

class UIComponents:
    @staticmethod
    def setup_styles():
        """Setup CSS styles for the application"""
        st.markdown("""
            <style>
                .block-container {
                    max-width: 1200px;
                    padding-top: 2rem;
                }
                [data-testid="stHorizontalBlock"] {
                    align-items: center;
                    gap: 2rem;
                }
                .stButton > button {
                    width: 100%;
                    margin-top: 1rem;
                }
                .canvas-container {
                    display: flex;
                    justify-content: center;
                    margin: 1rem 0;
                }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def create_sidebar_controls():
        st.sidebar.title("Inpainting Controls")
        
        controls = {}
        
        controls["drawing_mode"] = st.sidebar.radio(
            "Drawing Tool:",
            options=["freedraw", "rect"],
            format_func=lambda x: "Freehand" if x == "freedraw" else "Rectangle"
        )
        
        controls["stroke_width"] = st.sidebar.slider(
            "Brush size:",
            min_value=1,
            max_value=100,
            value=30
        )
        
        return controls

    @staticmethod
    def display_instructions():
        st.markdown("""
        ### Instructions
        1. Upload an image using the file uploader
        2. Use the drawing tools to mark areas for inpainting:
            - **Freehand**: Draw freely over areas to remove
            - **Rectangle**: Click and drag to create rectangular masks
        3. White areas in the mask will be inpainted
        4. Click 'Process Image' when ready
        """)

    @staticmethod
    def display_metrics(metrics):
        if not metrics:
            return
            
        st.sidebar.markdown("### Metrics")
        if 'psnr' in metrics:
            st.sidebar.metric("PSNR", f"{metrics['psnr']:.2f} dB")
        
        for key, value in metrics.items():
            if key != 'psnr':
                st.sidebar.metric(key, f"{value:.4f}")

    @staticmethod
    def handle_errors(error, show_traceback=True):
        """Enhanced error handling with full traceback"""
        st.error(f"Error: {str(error)}")
        if show_traceback:
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc(), language="python")