import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from open3d import core as o3c
from Constants import *
LAPLASS = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

def process_focus(image: np.ndarray):
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            gray_image = cv2.GaussianBlur(image, (5, 5), 0)
            return np.linalg.norm(cv2.filter2D(gray_image, 0, LAPLASS), ord="fro")
        elif image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            return np.linalg.norm(cv2.filter2D(gray_image, 0, LAPLASS), ord="fro")
        elif image.ndim == 4:
            gray_images = []
            ret_laplasses = []
            for not_gray in image:
                gray_image = cv2.cvtColor(not_gray, cv2.COLOR_RGB2GRAY)
                gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
                gray_images.append(gray_image)
            for gray_image in gray_images:
                ret_laplasses.append(np.linalg.norm(cv2.filter2D(gray_image, 0, LAPLASS), ord="fro"))
            return np.array(ret_laplasses)
        

def create_circle_grid(width, height, circle_grid_size=(5, 11), circle_r_pix=50):
    circle_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    rows, cols = circle_grid_size
    circle_coord = np.zeros((rows * cols, 2), dtype=np.uint32)
    position = 0
    for col in range(cols):
        pos_y = col * 3 * circle_r_pix + 3 * circle_r_pix
        for row in range(rows):
            if col % 2:
                pos_x = row * 6 * circle_r_pix + 3 * circle_r_pix
            else:
                pos_x = row * 6 * circle_r_pix 
            circle_coord[position] = [pos_x, pos_y]
            position += 1
    for point in circle_coord:
        if point[0] < width and point[1] < height and point[0] and point[1]:
            cv2.circle(circle_image, tuple(point), circle_r_pix, (0, 0, 0), cv2.FILLED)
    return circle_image

def get_optimal_grid(width: int, height: int, circle_radius: int):
    offset = 3 * circle_radius
    work_width = width - offset
    work_height = height - offset
    optimal_x = work_width // offset
    optimal_y = work_height // offset
    return optimal_y, optimal_x

def get_gray_codes(width=PROJECTOR_WIDTH, height=PROJECTOR_HEIGHT):
    maximum = max(width, height)
    number_of_bits = int(np.ceil(np.log2(maximum)))
    gray_codes = np.arange(maximum, dtype=np.uint16)
    gray_codes = (gray_codes >> 1) ^ gray_codes
    gray_codes.byteswap(True)

    return np.unpackbits(gray_codes.view(dtype=np.uint8)).reshape((-1, 16))[:, 16-number_of_bits:]

def get_gray_code_images(width=PROJECTOR_WIDTH, height=PROJECTOR_HEIGHT):
    gray_codes = get_gray_codes(width, height)
    images = np.zeros((4*len(gray_codes[0])+2,height, width), dtype=np.uint8)
    images[0, :, :] = 0
    images[1, :, :] = 255
    line_size = width // len(gray_codes)

    for i, code in enumerate(gray_codes):
        for j, bit in enumerate(code):
            start_position = i * line_size
            end_position = (i + 1) * line_size
            vertical_normal = 2*j + 2
            horizontal_normal = 2*(len(code) - (j+1)) + 3
            vertical_inverse = 2*j + 2*len(code) + 2
            horizontal_inverse = 2*(len(code) - (j+1)) + 3 + 2*len(code)

            images[vertical_normal, :, start_position:end_position] = 255 if bit == 1 else 0
            images[vertical_inverse, :, start_position:end_position] = 255 - images[vertical_normal, :, start_position:end_position]
            if i < height:
                images[horizontal_normal, start_position:end_position, :] = 255 if bit == 1 else 0
                images[horizontal_inverse, start_position:end_position, :] = 255 - images[horizontal_normal, start_position:end_position, :]

    return images

def get_Ld_Lg(images):
    black = images[0]
    white = images[1]

    pattern_len = int((len(images) - 2) / 4)
    horizontal_ids = np.array([2*pattern_len - 2, 2*pattern_len - 4, 2*pattern_len - 6, 4*pattern_len - 2, 4*pattern_len - 4, 4*pattern_len - 6], dtype=np.uint8)
    vertical_ids = np.array([1, 3, 5, 2*pattern_len + 1, 2*pattern_len + 3, 2*pattern_len + 5], dtype=np.uint8)
    b = (white + black)/white
    b_inv = white / (white + black)
    remaining_images = images[2:]
    L_max = np.max(np.take(remaining_images, horizontal_ids, axis=0), axis=0)
    L_min = np.min(np.take(remaining_images, vertical_ids, axis=0), axis=0)
    L_max = L_max.astype(np.float64)
    L_min = L_min.astype(np.float64)

    # L_d = (L_max - L_min) * b_inv
    # L_g = 2.0 * (L_max - L_d) * b_inv
    L_d = (L_max - L_min)/(1.0 - b)
    L_g = 2.0 * (L_min - b*L_max)/(1.0 - b**2)
    return L_d, L_g

def get_Lited(images, Ld, Lg, eps=100, m=2):
    pattern_len = (len(images) - 2) // 4
    pattern_images = images[2:]
    normal_patterns = pattern_images[:pattern_len*2]
    inverse_patterns = pattern_images[pattern_len*2:]

    horizontal_ids = [2*i + 1 for i in range(pattern_len)]
    vertical_ids = [2*i for i in range(pattern_len)]

    h_norm = np.take(normal_patterns, horizontal_ids, axis=0)
    v_norm = np.take(normal_patterns, vertical_ids, axis=0)
    h_inv = np.take(inverse_patterns, horizontal_ids, axis=0)
    v_inv = np.take(inverse_patterns, vertical_ids, axis=0)

    h_codes = np.zeros(h_norm.shape, dtype=np.int8) - 1
    v_codes = np.zeros(v_norm.shape, dtype=np.int8) - 1
    
    h_ims = np.zeros(h_norm.shape, dtype=np.uint8) + 100
    v_ims = np.zeros(v_norm.shape, dtype=np.uint8) + 100
    
    Ld = np.repeat(Ld[np.newaxis, :, :], pattern_len, axis=0)
    Lg = np.repeat(Lg[np.newaxis, :, :], pattern_len, axis=0)

    h_codes[np.where(Ld < m)] = -1
    v_codes[np.where(Ld < m)] = -1
    h_ims[np.where(Ld < m)] = 100
    v_ims[np.where(Ld < m)] = 100

    # h_codes[np.where((Ld > (Lg + eps)) & (h_norm > (h_inv + eps)))] = 1
    # v_codes[np.where((Ld > (Lg + eps)) & (v_norm > (v_inv + eps)))] = 1
    # h_ims[np.where((Ld > (Lg + eps)) & (h_norm > (h_inv + eps)))] = 255
    # v_ims[np.where((Ld > (Lg + eps)) & (v_norm > (v_inv + eps)))] = 255

    # h_codes[np.where((Ld > (Lg + eps)) & ((h_norm + eps) < h_inv))] = 0
    # v_codes[np.where((Ld > (Lg + eps)) & ((v_norm + eps) < v_inv))] = 0
    # h_ims[np.where((Ld > (Lg + eps)) & ((h_norm + eps) < h_inv))] = 0
    # v_ims[np.where((Ld > (Lg + eps)) & ((v_norm + eps) < v_inv))] = 0

    # h_codes[np.where(((h_norm + eps) < Ld) & (h_inv > (Lg + eps)))] = 0
    # v_codes[np.where(((v_norm + eps) < Ld) & (v_inv > (Lg + eps)))] = 0
    # h_ims[np.where(((h_norm + eps) < Ld) & (h_inv > (Lg + eps)))] = 0
    # v_ims[np.where(((v_norm + eps) < Ld) & (v_inv > (Lg + eps)))] = 0

    # h_codes[np.where((h_norm > (Lg + eps)) & ((h_inv + eps) < Ld))] = 1
    # v_codes[np.where((v_norm > (Lg + eps)) & ((v_inv + eps) < Ld))] = 1
    # h_ims[np.where((h_norm > (Lg + eps)) & ((h_inv + eps) < Ld))] = 255
    # v_ims[np.where((v_norm > (Lg + eps)) & ((v_inv + eps) < Ld))] = 255

    for image_index, image in enumerate(h_codes):
        print("h", image_index)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if Ld[image_index][row][col] > np.int32(Lg[image_index][row][col]) + eps and h_norm[image_index][row][col] > np.int32(h_inv[image_index][row][col]) + eps:
                    h_codes[image_index][row][col] = 1
                    h_ims[image_index][row][col] = 255
                elif Ld[image_index][row][col] > np.int32(Lg[image_index][row][col]) + eps and np.int32(h_norm[image_index][row][col]) + eps < h_inv[image_index][row][col]:
                    h_codes[image_index][row][col] = 0
                    h_ims[image_index][row][col] = 0
                elif np.int32(h_norm[image_index][row][col]) + eps < Ld[image_index][row][col] and h_inv[image_index][row][col] > np.int32(Lg[image_index][row][col]) + eps:
                    h_codes[image_index][row][col] = 0
                    h_ims[image_index][row][col] = 0
                elif h_norm[image_index][row][col] > np.int32(Lg[image_index][row][col]) + eps and np.int32(h_inv[image_index][row][col]) + eps > Ld[image_index][row][col]:
                    h_codes[image_index][row][col] = 1
                    h_ims[image_index][row][col] = 255
    
    for image_index, image in enumerate(v_codes):
        print("v", image_index)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if Ld[image_index][row][col] > np.int32(Lg[image_index][row][col]) + eps and v_norm[image_index][row][col] > np.int32(v_inv[image_index][row][col]) + eps:
                    v_codes[image_index][row][col] = 1
                    v_ims[image_index][row][col] = 255 
                elif Ld[image_index][row][col] > np.int32(Lg[image_index][row][col]) + eps and np.int32(v_norm[image_index][row][col]) + eps < v_inv[image_index][row][col]:
                    v_codes[image_index][row][col] = 0
                    v_ims[image_index][row][col] = 0
                elif np.int32(v_norm[image_index][row][col]) + eps < Ld[image_index][row][col] and v_inv[image_index][row][col] > np.int32(Lg[image_index][row][col]) + eps:
                    v_codes[image_index][row][col] = 0
                    v_ims[image_index][row][col] = 0
                elif v_norm[image_index][row][col] > np.int32(Lg[image_index][row][col]) + eps and np.int32(v_inv[image_index][row][col]) + eps > Ld[image_index][row][col]:
                    v_codes[image_index][row][col] = 1
                    v_ims[image_index][row][col] = 255

    return h_codes, v_codes, h_ims, v_ims

def gray_to_decimal(gray_codes):
    gray_code_list = [str(gray_codes[i]) for i in range(0, len(gray_codes))]
    gray_code_str = ''.join(gray_code_list)
    if '-1' in gray_code_str:
        return -1
    
    gray_code_binary = int(gray_code_str, 2)
    m = gray_code_binary >> 1
    while m:
        gray_code_binary ^= m
        m >>= 1
    return gray_code_binary

def plot_cloud(points, colors):
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points.T, o3c.float32))
    pcd = pcd.to_legacy()
    pcd.colors = o3d.cpu.pybind.utility.Vector3dVector(colors)

    print('Remove outlier in point cloud')
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    inlier_cloud = pcd.select_by_index(ind)

    print('Save ply file')
    o3d.io.write_point_cloud(os.path.join(PATH_END, 'cloud.ply'), inlier_cloud)

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        id = 0
        while os.path.exists(os.path.join(PATH_END, f'cloud_{id}.png')):
            id+=1
        plt.imsave(os.path.join(PATH_END, f'cloud_{id}.png'), np.asarray(image))
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([inlier_cloud], key_to_callback)

def generate_aruco():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
    image = board.generateImage(outSize=(1920, 1080), marginSize=50)
    cv2.imwrite("charuco_board.png", image)


def adaptive_illumination_correction(image):
    low_pass = cv2.GaussianBlur(image, (201, 201), 60)
    
    high_pass = image.astype(float) - low_pass.astype(float)
    
    corrected = cv2.normalize(high_pass + 127, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return corrected

def contrast(image, alpha=1.5, beta=-30):
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

if __name__ == "__main__":
    generate_aruco()
    # for i, image in enumerate(get_gray_code_images()):
    #     cv2.namedWindow(f"{i}", cv2.WINDOW_NORMAL)
    #     cv2.setWindowProperty(f"{i}", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #     cv2.imshow(f"{i}", image)
    #     cv2.waitKey(2000)
    #     cv2.destroyAllWindows()
    # image = cv2.imread("cal_proj.png")
    # cv2.imshow("test", image)
    # cv2.waitKey(0)
    # time.sleep(5)

    # cv2.destroyWindow("test")
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, circles = cv2.findCirclesGrid(gray, (4, 11), flags=cv2.CALIB_CB_SYMMETRIC_GRID)

    # if ret:
    #     frame = cv2.drawChessboardCorners(image, (4, 11), circles, ret)
    # else:
    #     frame = np.zeros((500, 500), dtype=np.uint8)

    # cv2.imshow("da", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


