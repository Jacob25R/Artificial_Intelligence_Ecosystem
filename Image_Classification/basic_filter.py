import cv2
import numpy as np
import os

def apply_artistic_filter(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return

    # 1. Convert to LAB color space to boost saturation & contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge channels back
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Sharpen edges for high detail
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # 3. Add a Cool Cyan/Magenta Tint (Neon/Cyberpunk vibe)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Boost Saturation
    s = cv2.add(s, 40)
    
    final_hsv = cv2.merge((h, s, v))
    filtered_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # Save output image
    output_filename = "filtered_output.jpg"
    cv2.imwrite(output_filename, filtered_img)
    print(f"\n[+] Success! Filtered image saved as '{output_filename}'")

if __name__ == "__main__":
    print("--- Artistic Filter Tool ---")
    while True:
        img_name = input("\nEnter image filename (or 'exit' to quit): ").strip()
        if img_name.lower() == 'exit':
            break
        apply_artistic_filter(img_name)
