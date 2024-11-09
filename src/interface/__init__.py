"""
Interface module for the Image Inpainting Application.
Provides the main application class and UI components.
"""

__version__ = '1.0.0'

from src.interface.components.ui_components import UIComponents
from src.interface.components.canvas_handler import CanvasHandler
from src.interface.app import InpaintingApp

__all__ = [
    'InpaintingApp',
    'UIComponents',
    'CanvasHandler'
]