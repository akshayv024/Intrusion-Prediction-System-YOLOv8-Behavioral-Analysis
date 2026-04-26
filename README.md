# AI-Based Intrusion Prediction System

An intelligent surveillance system designed to predict and detect unauthorized intrusion in restricted areas using YOLOv8 and behavioral movement analysis.

## Features

- Real-time human detection using YOLOv8
- Person tracking with unique IDs
- Polygon-based restricted boundary monitoring
- Movement behavior analysis
- Intrusion risk prediction before boundary crossing
- Automatic screenshot capture
- Email alert system
- Event logging for security records
  

## How It Works

1. Capture video input from CCTV or webcam
2. Detect humans using YOLOv8
3. Track human movement across frames
4. Monitor restricted boundary area
5. Analyze movement direction and approach consistency
6. Predict intrusion risk in real time
7. Trigger alerts when risk threshold is reached
   

## Tech Stack

- Python
- YOLOv8
- OpenCV
- NumPy
- SMTP (Email Alerts)
  

## Installation

```bash
git clone <https://github.com/akshayv024/AI-Intrusion-Prediction-System>
cd Ai-Intrusion-Prediction-System
pip install -r requirements.txt
python intrusion_system.py
```


## Output

- Bounding box detection
- Risk percentage display
- Intrusion alert status
- Screenshot evidence
- Email notifications
- Event logs


## Research Publication

Published in International Journal of Sciences and Innovation Engineering  
DOI: https://doi.org/10.70849/ijsci03032600536


## Future Improvements

- Live CCTV integration
- Multi-camera support
- Cloud logging
- Mobile notifications
