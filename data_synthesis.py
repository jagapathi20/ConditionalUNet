import os
import numpy as np
import cv2
import json
def create_synthetic_polygons(num_samples=1000, image_size=128):
    synthetic_data = []
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    color_values = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255)
    }
    
    os.makedirs('synthetic/inputs', exist_ok=True)
    os.makedirs('synthetic/outputs', exist_ok=True)
    
    for i in range(num_samples):
        # Random polygon parameters
        num_sides = np.random.randint(3, 9)  # 3 to 8 sides
        center = (image_size // 2, image_size // 2)
        radius = np.random.randint(50, 100)
        color_name = np.random.choice(colors)
        color_value = color_values[color_name]
        
        # Generate polygon points
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
        points = []
        for angle in angles:
            # Add some randomness to make irregular polygons
            r = radius + np.random.randint(-20, 20)
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Create input image (outline only)
        input_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        cv2.polylines(input_img, [points], True, (255, 255, 255), 2)
        
        # Create output image (filled polygon)
        output_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        cv2.fillPoly(output_img, [points], color_value)
        
        # Save images
        input_filename = f'synthetic_input_{i:04d}.png'
        output_filename = f'synthetic_output_{i:04d}.png'
        
        cv2.imwrite(f'synthetic/inputs/{input_filename}', input_img)
        cv2.imwrite(f'synthetic/outputs/{output_filename}', output_img)
        
        synthetic_data.append({
            'input': input_filename,
            'output': output_filename,
            'color': color_name
        })
    
    # Save synthetic data.json
    with open('synthetic/data.json', 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    print(f"Generated {num_samples} synthetic polygon samples")
    return synthetic_data