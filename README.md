# Connect4 with Dice Game Analysis

A computer vision system for analyzing Connect4 with Dice gameplay videos.
![End Frame](<End Frame hard.png>)

For more details read [report.md](<report.md>)

## Project Structure

```
project2/
├── data/           # Store input video files
├── src/            # Source code
├── output/         # Output files (logs, annotated videos)
└── tests/          # Test files
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your game video in the `data` directory.
2. Run the main script:
```bash
python src/main.py --video data/your_video.mp4 --config config_tuned.yaml
```

3. Check the `output` directory for:
   - Game event logs (JSON format)
   - Annotated video (if enabled)

## Configuration

To customize the system's behavior, modify the configuration file (`config.yaml`) or use `app.py` by running:
```bash
streamlit run src/app.py --server.maxUploadSize 2000
```  
and customize parameters here.

## Features

- Video stabilization and preprocessing
- Dice value detection using computer vision
- Counter detection and placement tracking
- Game state management and winning condition detection
- Event logging with timestamps

## Development

To run tests:
```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
