import streamlit as st
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
import os

from data_handling import VideoHandler
from main import process_frame, DebugVisualizer, load_config, preprocess_frame
from game_state import GameState, GameController

def load_and_save_config(config_path: str, tuned_config_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load configuration and save any modifications to tuned config file."""
    if Path(tuned_config_path).exists():
        config = load_config(tuned_config_path)
    else:
        config = load_config(config_path)
        # Save initial config as tuned config
        with open(config_path, 'r') as f:
            initial_config = yaml.safe_load(f)
        with open(tuned_config_path, 'w') as f:
            yaml.dump(initial_config, f)
    return config

def update_config(tuned_config_path: str, section: str, param: str, value: any) -> None:
    """Update a specific parameter in the tuned config file while preserving other settings."""
    with open(tuned_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if section not in config:
        config[section] = {}
    if param not in config[section]:
        config[section][param] = {}
        
    # Deep update to preserve nested structure
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    deep_update(config[section][param], value)
    
    with open(tuned_config_path, 'w') as f:
        yaml.dump(config, f)

def show_frame_with_params(title: str, image: np.ndarray, params: Optional[Dict] = None, config_path: Optional[str] = None):
    """Show a frame with its parameters if any and save changes automatically."""
    if not st.session_state.show_preprocessing and title.startswith("1."):
        return None
    
    if not st.session_state.dice_debug and title.startswith("6."):
        return None
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(title)
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image,
            caption=title
        )
    
    if params:
        with col2:
            st.write("Parameters")
            values = {k: st.slider(v["label"], v["min"], v["max"], v["value"], key=f"{title}_{k}")
                     for k, v in params.items()}
            
            # Save parameters if they changed and config path is provided
            if config_path and values != {k: v["value"] for k, v in params.items()}:
                if "Grid Mask" in title:
                    update_config(config_path, "detection", "grid", {
                        "grid_color_range": {
                            "lower": [values['h_lower'], values['s_lower'], values['v_lower']],
                            "upper": [values['h_upper'], values['s_upper'], values['v_upper']]
                        }
                    })
                elif "Detected Circles" in title:
                    update_config(config_path, "detection", "grid", {
                        "circle_detection": {
                            "min_dist": values['min_dist'],
                            "param1": values['param1'],
                            "param2": values['param2'],
                            "min_radius": values['min_radius'],
                            "max_radius": values['max_radius']
                        }
                    })
                elif "Final Result" in title:
                    update_config(config_path, "detection", "grid", {
                        "color_threshold": values['color_threshold']
                    })
                elif "Grid Detection - Processed Contour" in title:
                    update_config(config_path, "detection", "grid", {
                        "epsilon_factor": values['epsilon_factor']
                    })
                elif "Dice Detection" in title:
                    update_config(config_path, "detection", "dice", {
                        "canny_high": values['canny_high'],
                        "canny_low": values['canny_low'],
                        "max_radius": values['max_radius'],
                        "min_radius": values['min_radius']
                    })
                elif "Red Mask" in title:
                    update_config(config_path, "detection", "grid", {
                        "red_ranges": [
                            {
                                "lower": [values['h_lower1'], values['s_lower1'], values['v_lower1']],
                                "upper": [values['h_upper1'], values['s_upper1'], values['v_upper1']]
                            },
                            {
                                "lower": [values['h_lower2'], values['s_lower2'], values['v_lower2']],
                                "upper": [values['h_upper2'], values['s_upper2'], values['v_upper2']]
                            }
                        ]
                    })
                elif "Yellow Mask" in title:
                    update_config(config_path, "detection", "grid", {
                        "yellow_range": {
                            "lower": [values['h_lower'], values['s_lower'], values['v_lower']],
                            "upper": [values['h_upper'], values['s_upper'], values['v_upper']]
                        }
                    })
                elif "White Mask" in title:
                    update_config(config_path, "detection", "dice", {
                        "color_range": {
                            "lower": [values['h_lower'], values['s_lower'], values['v_lower']],
                            "upper": [values['h_upper'], values['s_upper'], values['v_upper']]
                        }
                    })
                elif "Dice - Edges" in title:
                    update_config(config_path, "detection", "dice", {
                        "canny_low": values['canny_low'],
                        "canny_high": values['canny_high']
                    })
                st.rerun()
            return values
    return {}

def main():
    st.set_page_config(layout="wide")
    st.title("Connect4 with Dice Game Analysis")
    
    # Initialize session state for debug controls
    if 'show_preprocessing' not in st.session_state:
        st.session_state.show_preprocessing = False
    if 'counter_debug' not in st.session_state:
        st.session_state.counter_debug = False
    if 'dice_debug' not in st.session_state:
        st.session_state.dice_debug = False
    
    # Set up paths
    config_path = "config.yaml"
    tuned_config_path = "config_tuned.yaml"
    
    # Load configuration
    config = load_and_save_config(config_path, tuned_config_path)
    video_config = config["video"]
    output_config = config["output"]
    detection_config = config["detection"]
    grid_params = detection_config["grid"]

    game_state = GameState()
    game_controller = GameController(game_state)
    
    # Video input
    video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi'])
    
    # Get video files from data folder
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    video_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mp4')])
    
    selected_video = st.selectbox(
        "Or select a video from data folder",
        options=video_files,
        index=video_files.index("hard2.mp4") if "hard2.mp4" in video_files else 0
    )
    
    # Use uploaded file if available, otherwise use selected video from data folder
    video_path = video_file.name if video_file else os.path.join("data", selected_video)
    
    if video_file is not None or selected_video:
        # Save uploaded video to temp file
        temp_video_path = "temp_video.mp4"
        if video_file:
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())
        else:
            with open(os.path.join("data", selected_video), "rb") as f:
                with open(temp_video_path, "wb") as g:
                    g.write(f.read())
        
        # Initialize video capture
        video_handler = VideoHandler(temp_video_path)
        if not video_handler.load_video():
            st.error("Failed to load video")
            return
        
        # Frame selection slider
        total_frames = int(video_handler.frame_count)
        frame_number = st.slider("Frame", 0, total_frames-1, 0)
        
        # Debug controls
        st.sidebar.header("Debug Controls")
        st.session_state.show_preprocessing = st.sidebar.checkbox("Show Preprocessing", value=st.session_state.show_preprocessing)
        st.session_state.counter_debug = st.sidebar.checkbox("Show Counter Detection", value=st.session_state.counter_debug)
        st.session_state.dice_debug = st.sidebar.checkbox("Show Dice Detection", value=st.session_state.dice_debug)

        # Extract and process frame
        frame = video_handler.extract_frame(frame_number)
        if frame is not None:
            # Initialize components
            debug_visualizer = DebugVisualizer(enabled=True, streamlit_mode=True)

            # Calculate average corners from first 100 frames if needed
            average_corners = None
            if frame_number >= 10:
                prev_gray = None
                prev_points = None
                corners_history = []
                for i in range(10):
                    temp_frame = video_handler.extract_frame(i)

                    adjusted, transform, gray, points = preprocess_frame(temp_frame, prev_gray, prev_points, None)
                    _, _, corners = process_frame(
                        adjusted, grid_params,
                        debug_visualizer=None,  # No debug visualization for temp frames
                        transform=transform,
                        counter_debug=st.session_state.counter_debug,
                        game_state=game_state,
                        game_controller=game_controller,
                        detection_config=detection_config
                    )
                    if corners is not None:
                        corners_history.append(corners)
                    prev_gray = gray
                    prev_points = points
                
                if corners_history:
                    average_corners = np.mean(corners_history, axis=0)
                    st.info(f"Using average corners from {len(corners_history)} valid frames")

            # Preprocess frame
            prev_frame = video_handler.extract_frame(frame_number-1) if frame_number > 0 else None
            _, _, prev_gray, prev_points = preprocess_frame(prev_frame, None, None, None) if prev_frame is not None else (None, None, None, None)
            adjusted, transform, _, _ = preprocess_frame(frame, prev_gray, prev_points, debug_visualizer)
            if adjusted is None:
                st.error("Failed to preprocess frame")
                return

            # Process frame with average corners if available
            states, visualized, _ = process_frame(
                adjusted, grid_params,
                debug_visualizer=debug_visualizer,
                override_corners=average_corners if frame_number >= 10 else None,
                transform=transform,
                counter_debug=st.session_state.counter_debug,
                game_state=game_state,
                game_controller=game_controller,
                detection_config=detection_config
            )

            # Show debug images with parameters where needed
            debug_images = debug_visualizer.get_debug_images()
            for title, image in debug_images.items():
                if "Grid Mask" in title:
                    grid_params_update = show_frame_with_params(
                        f"{title} {grid_params['grid_color_range']['lower']}-{grid_params['grid_color_range']['upper']}",
                        image,
                        {
                            "h_lower": {"label": "H Lower", "min": 0, "max": 180, "value": grid_params['grid_color_range']['lower'][0]},
                            "s_lower": {"label": "S Lower", "min": 0, "max": 255, "value": grid_params['grid_color_range']['lower'][1]},
                            "v_lower": {"label": "V Lower", "min": 0, "max": 255, "value": grid_params['grid_color_range']['lower'][2]},
                            "h_upper": {"label": "H Upper", "min": 0, "max": 180, "value": grid_params['grid_color_range']['upper'][0]},
                            "s_upper": {"label": "S Upper", "min": 0, "max": 255, "value": grid_params['grid_color_range']['upper'][1]},
                            "v_upper": {"label": "V Upper", "min": 0, "max": 255, "value": grid_params['grid_color_range']['upper'][2]}
                        },
                        tuned_config_path
                    )
                    if grid_params_update:
                        grid_params['grid_color_range']['lower'] = [
                            grid_params_update['h_lower'],
                            grid_params_update['s_lower'],
                            grid_params_update['v_lower']
                        ]
                        grid_params['grid_color_range']['upper'] = [
                            grid_params_update['h_upper'],
                            grid_params_update['s_upper'],
                            grid_params_update['v_upper']
                        ]
                elif "Red Mask" in title:
                    red_params_update = show_frame_with_params(
                        f"{title} Range1: {grid_params['red_ranges'][0]['lower']}-{grid_params['red_ranges'][0]['upper']}, Range2: {grid_params['red_ranges'][1]['lower']}-{grid_params['red_ranges'][1]['upper']}",
                        image,
                        {
                            "h_lower1": {"label": "H Lower (Range 1)", "min": 0, "max": 180, "value": grid_params['red_ranges'][0]['lower'][0]},
                            "s_lower1": {"label": "S Lower (Range 1)", "min": 0, "max": 255, "value": grid_params['red_ranges'][0]['lower'][1]},
                            "v_lower1": {"label": "V Lower (Range 1)", "min": 0, "max": 255, "value": grid_params['red_ranges'][0]['lower'][2]},
                            "h_upper1": {"label": "H Upper (Range 1)", "min": 0, "max": 180, "value": grid_params['red_ranges'][0]['upper'][0]},
                            "s_upper1": {"label": "S Upper (Range 1)", "min": 0, "max": 255, "value": grid_params['red_ranges'][0]['upper'][1]},
                            "v_upper1": {"label": "V Upper (Range 1)", "min": 0, "max": 255, "value": grid_params['red_ranges'][0]['upper'][2]},
                            "h_lower2": {"label": "H Lower (Range 2)", "min": 0, "max": 180, "value": grid_params['red_ranges'][1]['lower'][0]},
                            "s_lower2": {"label": "S Lower (Range 2)", "min": 0, "max": 255, "value": grid_params['red_ranges'][1]['lower'][1]},
                            "v_lower2": {"label": "V Lower (Range 2)", "min": 0, "max": 255, "value": grid_params['red_ranges'][1]['lower'][2]},
                            "h_upper2": {"label": "H Upper (Range 2)", "min": 0, "max": 180, "value": grid_params['red_ranges'][1]['upper'][0]},
                            "s_upper2": {"label": "S Upper (Range 2)", "min": 0, "max": 255, "value": grid_params['red_ranges'][1]['upper'][1]},
                            "v_upper2": {"label": "V Upper (Range 2)", "min": 0, "max": 255, "value": grid_params['red_ranges'][1]['upper'][2]}
                        },
                        tuned_config_path
                    )
                    # Update config if parameters changed
                    if red_params_update:
                        # Update red range 1
                        grid_params['red_ranges'][0]['lower'] = [
                            red_params_update['h_lower1'],
                            red_params_update['s_lower1'],
                            red_params_update['v_lower1']
                        ]
                        grid_params['red_ranges'][0]['upper'] = [
                            red_params_update['h_upper1'],
                            red_params_update['s_upper1'],
                            red_params_update['v_upper1']
                        ]
                        # Update red range 2
                        grid_params['red_ranges'][1]['lower'] = [
                            red_params_update['h_lower2'],
                            red_params_update['s_lower2'],
                            red_params_update['v_lower2']
                        ]
                        grid_params['red_ranges'][1]['upper'] = [
                            red_params_update['h_upper2'],
                            red_params_update['s_upper2'],
                            red_params_update['v_upper2']
                        ]
                elif "Yellow Mask" in title:
                    yellow_params_update = show_frame_with_params(
                        f"{title} {grid_params['yellow_range']['lower']}-{grid_params['yellow_range']['upper']}",
                        image,
                        {
                            "h_lower": {"label": "H Lower", "min": 0, "max": 180, "value": grid_params['yellow_range']['lower'][0]},
                            "s_lower": {"label": "S Lower", "min": 0, "max": 255, "value": grid_params['yellow_range']['lower'][1]},
                            "v_lower": {"label": "V Lower", "min": 0, "max": 255, "value": grid_params['yellow_range']['lower'][2]},
                            "h_upper": {"label": "H Upper", "min": 0, "max": 180, "value": grid_params['yellow_range']['upper'][0]},
                            "s_upper": {"label": "S Upper", "min": 0, "max": 255, "value": grid_params['yellow_range']['upper'][1]},
                            "v_upper": {"label": "V Upper", "min": 0, "max": 255, "value": grid_params['yellow_range']['upper'][2]}
                        },
                        tuned_config_path
                    )
                    # Update config if parameters changed
                    if yellow_params_update:
                        # Update yellow parameters
                        grid_params['yellow_range']['lower'] = [
                            yellow_params_update['h_lower'],
                            yellow_params_update['s_lower'],
                            yellow_params_update['v_lower']
                        ]
                        grid_params['yellow_range']['upper'] = [
                            yellow_params_update['h_upper'],
                            yellow_params_update['s_upper'],
                            yellow_params_update['v_upper']
                        ]
                elif "Processed Contour" in title:
                    params_update = show_frame_with_params(
                        title,
                        image,
                        {
                            "epsilon_factor": {
                                "label": "Epsilon Factor",
                                "min": 0.0,
                                "max": 1.0,
                                "value": grid_params['epsilon_factor']
                            }
                        },
                        tuned_config_path
                    )
                    if params_update:
                        grid_params['epsilon_factor'] = params_update['epsilon_factor']
                elif "Final Result" in title:
                    threshold_update = show_frame_with_params(
                        title,
                        image,
                        {
                            "color_threshold": {
                                "label": "Color Threshold",
                                "min": 0.0,
                                "max": 1.0,
                                "value": grid_params['color_threshold']
                            }
                        },
                        tuned_config_path
                    )
                    if threshold_update:
                        grid_params['color_threshold'] = threshold_update['color_threshold']
                elif "White Mask" in title:
                    white_params_update = show_frame_with_params(
                        f"{title} {detection_config['dice']['color_range']['lower']}-{detection_config['dice']['color_range']['upper']}",
                        image,
                        {
                            "h_lower": {"label": "H Lower", "min": 0, "max": 180, "value": detection_config['dice']['color_range']['lower'][0]},
                            "s_lower": {"label": "S Lower", "min": 0, "max": 255, "value": detection_config['dice']['color_range']['lower'][1]},
                            "v_lower": {"label": "V Lower", "min": 0, "max": 255, "value": detection_config['dice']['color_range']['lower'][2]},
                            "h_upper": {"label": "H Upper", "min": 0, "max": 180, "value": detection_config['dice']['color_range']['upper'][0]},
                            "s_upper": {"label": "S Upper", "min": 0, "max": 255, "value": detection_config['dice']['color_range']['upper'][1]},
                            "v_upper": {"label": "V Upper", "min": 0, "max": 255, "value": detection_config['dice']['color_range']['upper'][2]}
                        },
                        tuned_config_path
                    )
                    # Update config if parameters changed
                    if white_params_update:
                        # Update white parameters
                        detection_config['dice']['color_range']['lower'] = [
                            white_params_update['h_lower'],
                            white_params_update['s_lower'],
                            white_params_update['v_lower']
                        ]
                        detection_config['dice']['color_range']['upper'] = [
                            white_params_update['h_upper'],
                            white_params_update['s_upper'],
                            white_params_update['v_upper']
                        ]
                        
                        # Update config file
                        update_config(tuned_config_path, "detection", "dice", {
                            "color_range": {
                                "lower": detection_config['dice']['color_range']['lower'],
                                "upper": detection_config['dice']['color_range']['upper']
                            }
                        })
                elif "Dice - Edges" in title:
                    edge_params_update = show_frame_with_params(
                        f"{title} Low: {detection_config['dice']['canny_low']} High: {detection_config['dice']['canny_high']}",
                        image,
                        {
                            "canny_low": {"label": "Canny Low", "min": 0, "max": 255, "value": detection_config['dice']['canny_low']},
                            "canny_high": {"label": "Canny High", "min": 0, "max": 255, "value": detection_config['dice']['canny_high']}
                        },
                        tuned_config_path
                    )
                    # Update config if parameters changed
                    if edge_params_update:
                        detection_config['dice']['canny_low'] = edge_params_update['canny_low']
                        detection_config['dice']['canny_high'] = edge_params_update['canny_high']
                        
                        # Update config file
                        update_config(tuned_config_path, "detection", "dice", {
                            "canny_low": detection_config['dice']['canny_low'],
                            "canny_high": detection_config['dice']['canny_high']
                        })
                else:
                    show_frame_with_params(title, image)

            debug_visualizer.clear_debug_images()

            if st.button("Process Video"):
                pass

        # Clean up
        video_handler.cap.release()
        Path(temp_video_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
