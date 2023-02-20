# import the opencv library
import cv2
import numpy as np
import imagehash
import PIL

# define a video capture object
vid = cv2.VideoCapture(0)

TARGET_WIDTH = 768
TARGET_HEIGHT = 1024

def check_if_points_in_region(points, regions):
    if len(points) != 4:
        return False

    for region in regions:
        num_in_region = 0
        for point in points:
            point = point[0]
            if (point[1] > region[0][0]) and (point[1] < region[0][1]) and (point[0] > region[1][0]) and (point[0] < region[1][1]):
                num_in_region += 1
        if num_in_region != 1:
            return False

    return True

def extract_screen_content(frame, points):
    output_pts = np.float32([[0, 0], [0, TARGET_HEIGHT], [TARGET_WIDTH, TARGET_HEIGHT], [TARGET_WIDTH, 0]])
    matrix = cv2.getPerspectiveTransform(np.float32(points), output_pts)
    out = cv2.warpPerspective(frame, matrix, (TARGET_WIDTH, TARGET_HEIGHT), flags=cv2.INTER_LINEAR)
    return out


def detect_screen_position(frame):
    img_height, img_width, _ = frame.shape
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_treshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img_treshold = cv2.morphologyEx(img_treshold, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img_morphed = cv2.morphologyEx(img_treshold, cv2.MORPH_CLOSE, kernel)


    (cnts, _) = cv2.findContours(img_morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]


    found_screen = False
    capture_regions = [[[0, img_height/2], [0, img_width/2]],
                        [[0, img_height/2], [img_width/2, img_width]],
                       [[img_height/2, img_height], [0, img_width/2]],
                       [[img_height/2, img_height], [img_width/2, img_width]]]
    # loop over our contours
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if check_if_points_in_region(approx, capture_regions):
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            approx = np.squeeze(np.asarray(approx), 1)
            return (True, frame, approx)

    return (False, None, None)

def compare_images(reference, test):
    reference = PIL.Image.fromarray(reference)
    test = PIL.Image.fromarray(test)

    hash0 = imagehash.average_hash(reference)
    hash1 = imagehash.average_hash(test)
    cutoff = 5  # Can be changed according to what works best for your images

    hashDiff = hash0 - hash1  # Finds the distance between the hashes of images
    print(f"Diff: {hashDiff}")


img_cal = cv2.imread('/home/gregory-pc/Obrazy/black_csd/csd_test.png', cv2.IMREAD_COLOR)
img_cal = cv2.rotate(img_cal, cv2.ROTATE_90_CLOCKWISE)
is_detected, img, points = detect_screen_position(img_cal)
img_test = cv2.imread('/home/gregory-pc/Obrazy/black_csd/csd_test_w_content.png', cv2.IMREAD_COLOR)
img_test = cv2.rotate(img_test, cv2.ROTATE_90_CLOCKWISE)
img_reference = cv2.imread('/home/gregory-pc/Obrazy/black_csd/reference3.png', cv2.IMREAD_COLOR)
img = extract_screen_content(img_test, points)
compare_images(img, img_reference)
"""
cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    output = detect_screen_position(frame)
    # Display the resulting frame
    cv2.imshow('frame', output)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

"""