

import sys


priority_state = [
    {
        'STATE': 'TEDDY',
        'LABEL': 'teddy',
        'PRIORITY': 1
    },
    {
        'STATE': 'TEDDY_LYING',
        'label': 'teddy_lying',
        'PRIORITY': 2
    },
    {
        'STATE': 'TEDDY_PLAYING',
        'LABEL': 'teddy_play',
        'PRIORITY': 3
    },
    {
        'STATE': 'TEDDY_HOWLING',
        'LABEL': 'teddy_howling',
        'PRIORITY': 4
    },
    {
        'STATE': 'TEDDY_WATER',
        'PRIORITY': 5
    }
]

def generate_predictions(video_path):
    pass




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    generate_predictions(video_path)
