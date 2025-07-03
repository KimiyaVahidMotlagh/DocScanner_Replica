import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# -------------------- Step 1: Load & Preprocess --------------------
# Loads the original image manually (hardcoded path)
image = cv2.imread(f"C:/Users/ASUS/Downloads/CamScanner/sample.jpg")
if image is None:
    raise FileNotFoundError("Image not found at given path.")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(image_gray, d=9, sigmaColor=75, sigmaSpace=75)
edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

# -------------------- Step 2: Find Document Contour --------------------
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Helper to reorder points: top-left, top-right, bottom-right, bottom-left
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# -------------------- Step 3: Perspective Transform --------------------
if len(approx) == 4:
    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
else:
    raise ValueError("Could not find 4-point document contour.")

# -------------------- Step 4: Prepare for UI --------------------
window_name = "Document Filter Preview"
button_height = 50
filter_mode = 0
save_requested = False

# Get screen resolution dynamically
screen_res = (1920, 1080)
try:
    from screeninfo import get_monitors
    screen_res = (get_monitors()[0].width, get_monitors()[0].height)
except:
    pass
screen_width, screen_height = screen_res

# Button labels and associated modes
buttons = [
    ("Load", 0),
    ("Original", 1),
    ("Gray", 2),
    ("B&W", 3),
    ("Magic Color", 4),
    ("Save", 5)
]

# Image loader helper
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None, None

    # Step 1: Convert to grayscale
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Bilateral filter + Canny edge detection
    blurred = cv2.bilateralFilter(image_gray, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

    # Step 3: Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img.copy(), img.copy()

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        pts = approx.reshape(4, 2)
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return img.copy(), warped.copy()
    else:
        return img.copy(), img.copy()


# Update image canvas and sizes
def update_image(new_warped):
    global warped, width, height, canvas_height
    warped = new_warped
    height = warped.shape[0]
    width = warped.shape[1]
    canvas_height = height + button_height

    max_display_height = screen_height - 100
    if canvas_height > max_display_height:
        scale = max_display_height / canvas_height
        warped = cv2.resize(warped, (int(width * scale), int(height * scale)))
        height = warped.shape[0]
        width = warped.shape[1]
        canvas_height = height + button_height


update_image(warped)

# -------------------- Step 5: Filter Logic --------------------
def apply_filter(mode):
    if mode == 0:
        return warped
    elif mode == 1:
        return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    elif mode == 2:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 10)
    elif mode == 3:
        filtered = cv2.bilateralFilter(warped, 9, 75, 75)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(filtered, -1, sharpen_kernel)
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(cv2.multiply(s, 1.3), 0, 255).astype(np.uint8)
        v = np.clip(cv2.multiply(v, 1.2), 0, 255).astype(np.uint8)
        enhanced_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    else:
        return warped

# -------------------- Step 6: UI Buttons --------------------
def draw_buttons(canvas):
    for label, idx in buttons:
        x_start = int(idx * (width / len(buttons)))
        x_end = int((idx + 1) * (width / len(buttons)))
        color = (60, 120, 60) if idx == filter_mode else (30, 30, 30)
        if label == "Save":
            color = (70, 70, 130)
        cv2.rectangle(canvas, (x_start, canvas_height - button_height), (x_end, canvas_height), color, -1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = x_start + (x_end - x_start - text_size[0]) // 2
        text_y = canvas_height - (button_height - text_size[1]) // 2
        cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# -------------------- Step 7: Mouse Event --------------------
def get_button_index(x):
    segment_width = int(width / len(buttons))
    return int(x // segment_width)

def mouse_callback(event, x, y, flags, param):
    global filter_mode, save_requested
    if event == cv2.EVENT_LBUTTONDOWN and y >= canvas_height - button_height:
        idx = get_button_index(x)
        label = buttons[idx][0]
        if label == "Save":
            save_requested = True
        elif label == "Load":
            root = tk.Tk()
            root.withdraw()
            filetypes = [("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            filepath = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
            if filepath and os.path.exists(filepath):
                img_loaded, warped_loaded = load_image(filepath)
                if img_loaded is not None:
                    image = img_loaded
                    update_image(warped_loaded)
                    filter_mode = 0
                    print(f"[✅] Loaded image: {filepath}")
                    cv2.resizeWindow(window_name, width, canvas_height)
        else:
            filter_mode = buttons[idx][1]

# -------------------- Step 8: Main Loop --------------------
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)
cv2.resizeWindow(window_name, width, canvas_height)

while True:
    output = apply_filter(filter_mode)
    display_output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR) if len(output.shape) == 2 else output.copy()
    canvas = np.zeros((canvas_height, width, 3), dtype=np.uint8)
    canvas[:height, :, :] = display_output
    draw_buttons(canvas)
    cv2.imshow(window_name, canvas)

    if save_requested:
        filter_name = buttons[filter_mode][0].replace(" ", "_")
        filename = f"output_{filter_name}.jpg"
        cv2.imwrite(filename, output)
        print(f"[✅] Image saved as {filename}")
        save_requested = False

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()