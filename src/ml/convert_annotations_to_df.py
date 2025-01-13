import json
import sys
import math
import re
import pandas as pd

category_id_to_label = {
    1: "toy",
    2: "water",
    3: "food",
    4: "teddy",
    5: "teddy_lying",
    6: "teddy_howling",
    7: "teddy_play",
}

cols = [
    "frame_idx", "teddy_state", "teddy_confidence", "teddy_x", "teddy_y",
    "teddy_w", "teddy_h", "teddy_dx", "teddy_dy", "closest_toy_confidence",
    "closest_toy_x", "closest_toy_y", "closest_toy_w", "closest_toy_h",
    "closest_toy_dx", "closest_toy_dy", "water_confidence", "water_x",
    "water_y", "water_w", "water_h",
]

def calculate_distance(x1, y1, w1, h1, x2, y2, w2, h2):
    center1_x = x1 + w1 / 2.0
    center1_y = y1 + h1 / 2.0
    center2_x = x2 + w2 / 2.0
    center2_y = y2 + h2 / 2.0
    return math.hypot(center2_x - center1_x, center2_y - center1_y)

def get_closest_toy(frame_detections, teddy_x, teddy_y, teddy_w, teddy_h, prev_toy_pos):
    if teddy_x == -1 or teddy_y == -1:
        return (-1, -1, -1, -1, -1, 0, 0), None
        
    closest_toy = (-1, -1, -1, -1, -1)
    min_dist = float("inf")
    
    for toy in frame_detections["toy"]:
        dist = calculate_distance(teddy_x, teddy_y, teddy_w, teddy_h, *toy[1:])
        if dist < min_dist:
            min_dist = dist
            closest_toy = toy
    
    toy_dx = toy_dy = 0
    new_toy_pos = None
    
    if closest_toy[0] != -1 and prev_toy_pos:
        toy_dx = closest_toy[1] - prev_toy_pos[0]
        toy_dy = closest_toy[2] - prev_toy_pos[1]
    
    if closest_toy[0] != -1:
        new_toy_pos = (closest_toy[1], closest_toy[2])
        
    return (*closest_toy, toy_dx, toy_dy), new_toy_pos

def process_frame_data(detection, prev_positions):
    prev_teddy_pos, prev_toy_pos = prev_positions
    
    # Process teddy data
    if detection["teddy"]:
        teddy_data = (*detection["teddy"], detection["teddy_state"])
    else:
        teddy_data = (-1, -1, -1, -1, -1, "none")
    
    teddy_dx = teddy_dy = 0
    if prev_teddy_pos and teddy_data[1] != -1:
        teddy_dx = teddy_data[1] - prev_teddy_pos[0]
        teddy_dy = teddy_data[2] - prev_teddy_pos[1]
    
    new_teddy_pos = (teddy_data[1], teddy_data[2]) if teddy_data[1] != -1 else None
    
    toy_data, new_toy_pos = get_closest_toy(
        detection, teddy_data[1], teddy_data[2], teddy_data[3], teddy_data[4], prev_toy_pos
    )
    
    water_data = detection["water"][0] if detection["water"] else (-1, -1, -1, -1, -1)
    
    return {
        "teddy": (*teddy_data[:5], teddy_dx, teddy_dy, teddy_data[5]),
        "toy": toy_data,
        "water": water_data,
    }, (new_teddy_pos, new_toy_pos)

def create_detections_dict(data_source, is_instances_default=False):
    detections = {}
    
    if not is_instances_default:
        pattern = re.compile(
            r"Frame\s+(\d+):\s+(\w+)\s*\(([\d.]+)\):\s+x=([\d.]+),\s+y=([\d.]+),\s+w=([\d.]+),\s+h=([\d.]+)"
        )
        
        with open(data_source, "r") as f:
            for line in f:
                match = pattern.match(line.strip())
                if not match:
                    continue
                    
                frame_idx, label, conf, x, y, w, h = match.groups()
                frame_idx = int(frame_idx)
                conf = float(conf)
                bbox = [float(v) for v in (x, y, w, h)]
                
                if frame_idx not in detections:
                    detections[frame_idx] = {"teddy": None, "toy": [], "water": [], "teddy_state": None}
                
                if "teddy" in label.lower():
                    detections[frame_idx]["teddy"] = (conf, *bbox)
                    detections[frame_idx]["teddy_state"] = label.lower()
                elif label.lower() == "toy":
                    detections[frame_idx]["toy"].append((conf, *bbox))
                elif label.lower() == "water":
                    detections[frame_idx]["water"].append((conf, *bbox))
    else:
        data = json.loads(data_source)
        
        for item in data["annotations"]:
            frame_idx = item["image_id"]
            
            if frame_idx not in detections:
                detections[frame_idx] = {"teddy": None, "toy": [], "water": [], "teddy_state": None}
            
            # label = category_id_to_label[item["category_id"]]
            label = category_id_to_label[item["category_id"]]
            conf = 1.0 
            bbox = item["bbox"]
            
            if "teddy" in label:
                detections[frame_idx]["teddy"] = (conf, *bbox)
                detections[frame_idx]["teddy_state"] = label
            elif label == "toy":
                detections[frame_idx]["toy"].append((conf, *bbox))
            elif label == "water":
                detections[frame_idx]["water"].append((conf, *bbox))
    
    return detections

def create_dataframe(detections):
    rows = []
    prev_positions = (None, None)  # (teddy_pos, toy_pos)
    
    for frame_idx in sorted(detections.keys()):
        processed_data, prev_positions = process_frame_data(detections[frame_idx], prev_positions)
        
        row = {
            "frame_idx": frame_idx,
            "teddy_state": processed_data["teddy"][7],
            "teddy_confidence": processed_data["teddy"][0],
            "teddy_x": processed_data["teddy"][1],
            "teddy_y": processed_data["teddy"][2],
            "teddy_w": processed_data["teddy"][3],
            "teddy_h": processed_data["teddy"][4],
            "teddy_dx": processed_data["teddy"][5],
            "teddy_dy": processed_data["teddy"][6],
            "closest_toy_confidence": processed_data["toy"][0],
            "closest_toy_x": processed_data["toy"][1],
            "closest_toy_y": processed_data["toy"][2],
            "closest_toy_w": processed_data["toy"][3],
            "closest_toy_h": processed_data["toy"][4],
            "closest_toy_dx": processed_data["toy"][5],
            "closest_toy_dy": processed_data["toy"][6],
            "water_confidence": processed_data["water"][0],
            "water_x": processed_data["water"][1],
            "water_y": processed_data["water"][2],
            "water_w": processed_data["water"][3],
            "water_h": processed_data["water"][4],
        }
        rows.append(row)
    
    return pd.DataFrame(rows, columns=cols)

def cv_output_to_df(file_path):
    detections = create_detections_dict(file_path, is_instances_default=False)
    return create_dataframe(detections)

def instances_default_to_df(instances_file):
    with open(instances_file, 'r') as f:
        instances_data = f.read()
    detections = create_detections_dict(instances_data, is_instances_default=True)
    return create_dataframe(detections)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main_ml.py <filepath>")
        sys.exit(1)

    if not sys.argv[1].endswith('.txt') and not sys.argv[1].endswith('.json'):
        print("Error: The provided file is not a TXT or JSON file.")
        sys.exit(1)

    filepath = sys.argv[1]
    if filepath.endswith(".json"):
        df = instances_default_to_df(filepath)
    else:
        df = cv_output_to_df(filepath)

    df.to_csv("output.csv", index=False)
