# BeVIS

### Let Bevis be with you

BeVIS is an innovative AI project inspired by JARVIS, designed to seamlessly integrate with people's everyday lives through a moving camera. With BeVIS, you can simply use your fingers to select an object or area in mid-air and receive an explanation about it. Whether it’s a question you can't solve, information about a historical monument, or something in a museum, BeVIS will provide the answers. BeVIS can also act as the eyes for the visually impaired, enabling them to perceive and understand the world more easily.

BeVIS aims to make a significant contribution to educational equity, providing everyone with access to the best education resources and learning opportunities.

## Features

- **Interactive Content Recognition**: By framing with your fingers, BeVIS can recognize and explain objects or information, making learning and exploration more intuitive and accessible.
- **Support for the Visually Impaired**: BeVIS assists visually impaired users by identifying objects in the environment and providing audio descriptions.

## How It Works

BeVIS leverages advanced computer vision and machine learning techniques to achieve its functionality:

1. **Lightweight YOLOv5 for Hand extraction**: Uses YOLOv5, a state-of-the-art object detection model, optimized for detecting hands. This allows BeVIS to detect hand gestures accurately while maintaining high performance on resource-constrained devices.

2. **Simple IOU-Based Hand Tracking**: BeVIS incorporates a simple Intersection over Union (IOU) approach to track hand movements, ensuring the camera follows gestures smoothly and reliably.

3. **ResNet50 for Keypoint Detection**: Employs self-trained ResNet50, a deep neural network model, to identify the key points of the hand, enabling precise tracking of finger positions and motions.

4. **Gesture Semantics Using 2D Angle Analysis**: BeVIS defines gesture semantics by calculating the angles between hand keypoints, interpreting different gestures to trigger actions or responses.

5. **Spatio-Temporal Motion Analysis**: Analyzes the motion trajectory of hand movements over time to understand and differentiate complex interactions.

6. **Gesture Semantics Combined with Motion Trajectories**: By combining gesture detection with motion trajectory analysis, BeVIS enables natural and expressive user interactions, allowing users to point, drag, or select virtual objects seamlessly.

## Getting Started

### Requirements

- Python 3.8 or above
- OpenCV
- PyTorch
- YOLOv5 pretrained models
- &#x20;ResNet50 self-trained skeleton recognition

### Installation

1. Clone the repository:
   ```bash
   git clone 
   cd BeVIS
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pretrained models for YOLOv5 and ResNet50 and place them in the appropriate directory.

### Usage

To run BeVIS:

```bash
python inference.py
```

Ensure your camera is connected, and BeVIS will start recognizing your hand gestures. Use your fingers to frame and select the objects you are interested in, and BeVIS will provide explanations or descriptions in real time.

### Example Use Cases

- **Educational Tours**: Use BeVIS during museum tours to learn more about exhibits by framing them with your fingers.
- **Home Assistance**: Point at an appliance, and BeVIS can provide operational information.
- **Support for Visually Impaired Users**: BeVIS can detect and describe objects, providing real-time feedback to users with visual impairments.

- **ChatGPT**: Utilized for generating explanations and enhancing the interaction experience with natural language understanding. Built upon TTX and GPT-4, ChatGPT provides BeVIS with full-scene perception capabilities, significantly enhancing its ability to understand and respond to diverse scenarios.

- **YOLOv5**: Lightweight version for hand detection, ensuring efficient recognition even on devices with limited processing power.
- **ResNet50**: Used for detailed keypoint detection, crucial for accurate hand tracking.
- **OpenCV**: Facilitates real-time image processing and camera operations.
- **PyTorch**: For deep learning and model inference.

## Reference:



---

Let BeVIS be with you, transforming the way we interact with the world around us.

