import cv2
import numpy as np
from skimage import morphology, img_as_float, img_as_ubyte
import torch
from modelB4 import LDC

test_data = "CLASSIC"


# functions for traditional color-based section
def invert(image):
    image = (255 - image)
    return image


def find_contours_new(image=None):
    index = 0
    maxArea = 0
    img = image.copy()
    cnts, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(cnts)):
        if cv2.contourArea(cnts[i]) > maxArea:
            maxArea = cv2.contourArea(cnts[i])
            index = i

    # print(len(cnts))

    if len(cnts) < 1:
        centre = [320, 240]
    else:
        moments = cv2.moments(cnts[index])
        if moments['m00'] == 0:
            centre = [320, 240]
        else:
            centre = [int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])]
    return centre


# functions for DL edge-based section
def transform(img, gt, img_width, img_height, mean_bgr):
    # gt[gt< 51] = 0 # test without gt discrimination
    if test_data == "CLASSIC":
        print(f"actual size: {img.shape}, target size: {(img_height, img_width,)}")
        # img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.resize(img, (img_width, img_height))
        gt = None

    # Make images and labels at least 512 by 512
    elif img.shape[0] < 512 or img.shape[1] < 512:
        img = cv2.resize(img, (img_width, img_height))  # 512
        # gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height))  # 512

    # Make sure images and labels are divisible by 2^4=16
    elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
        img_width = ((img.shape[1] // 16) + 1) * 16
        img_height = ((img.shape[0] // 16) + 1) * 16
        img = cv2.resize(img, (img_width, img_height))
        gt = cv2.resize(gt, (img_width, img_height))
    else:
        img_width = img_width
        img_height = img_height
        img = cv2.resize(img, (img_width, img_height))
        gt = cv2.resize(gt, (img_width, img_height))
    # # For FPS
    # img = cv2.resize(img, (496,320))
    # if self.yita is not None:
    #     gt[gt >= self.yita] = 1
    img = np.array(img, dtype=np.float32)
    # if self.rgb:
    #     img = img[:, :, ::-1]  # RGB->BGR
    img -= mean_bgr
    img = img.transpose((2, 0, 1))
    # special process: unsqueeze
    img = torch.from_numpy(img.copy()).float().unsqueeze(0)

    if test_data == "CLASSIC":
        gt = np.zeros((img.shape[:2]))
        gt = torch.from_numpy(np.array([gt])).float()
    else:
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255.
        gt = torch.from_numpy(np.array([gt])).float()
    return img, gt


def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
          ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def return_processed_preds(tensor, img_shape=None):
    # print(img_shape)
    fuse_name = 'fused'
    av_name = 'avg'
    tensor2 = None
    tmp_img2 = None

    # output_dir_f = os.path.join(output_dir, fuse_name)
    # output_dir_a = os.path.join(output_dir, av_name)
    # os.makedirs(output_dir_f, exist_ok=True)
    # os.makedirs(output_dir_a, exist_ok=True)

    # 255.0 * (1.0 - em_a)
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)
    # print(f"tensor shape: {tensor.shape}")

    image_shape = [x.cpu().detach().numpy() for x in img_shape]
    # (H, W) -> (W, H)
    image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

    # (H, W) -> (W, H)
    # image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

    idx = 0
    for i_shape in image_shape:
        # print(i_shape)
        tmp = tensor[:, idx, ...]
        tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
        # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
        tmp = np.squeeze(tmp)
        tmp2 = np.squeeze(tmp2) if tensor2 is not None else None

        # Iterate our all 7 NN outputs for a particular image
        preds = []
        fuse_num = tmp.shape[0] - 1
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)
            # tmp_img[tmp_img < 0.0] = 0.0
            # tmp_img = 255.0 * (1.0 - tmp_img)
            if tmp2 is not None:
                tmp_img2 = tmp2[i]
                tmp_img2 = np.uint8(image_normalization(tmp_img2))
                tmp_img2 = cv2.bitwise_not(tmp_img2)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
                tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None

            if tmp2 is not None:
                tmp_mask = np.logical_and(tmp_img > 128, tmp_img2 < 128)
                tmp_img = np.where(tmp_mask, tmp_img2, tmp_img)
                preds.append(tmp_img)

            else:
                preds.append(tmp_img)

            if i == fuse_num:
                # print('fuse num',tmp.shape[0], fuse_num, i)
                fuse = tmp_img
                fuse = fuse.astype(np.uint8)
                if tmp_img2 is not None:
                    fuse2 = tmp_img2
                    fuse2 = fuse2.astype(np.uint8)
                    # fuse = fuse-fuse2
                    fuse_mask = np.logical_and(fuse > 128, fuse2 < 128)
                    fuse = np.where(fuse_mask, fuse2, fuse)
                    # print(fuse.shape, fuse_mask.shape)

        # Get the mean prediction of all the 7 outputs
        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))
        # output_file_name_f = os.path.join(output_dir_f, file_name)
        # output_file_name_a = os.path.join(output_dir_a, file_name)
        fuse = 255 - fuse
        average = 255 - average
        # cv2.imwrite(output_file_name_f, fuse)
        # cv2.imwrite(output_file_name_a, average)
        idx += 1
    return fuse, average


if __name__ == "__main__":
    # change source and mode together, accepted source: .jpg, .png, .avi, .mp4, 0, 1... and mode: 'image', 'camera'
    source = '470052.jpg'
    mode = 'image'
    frame_list = []
    save_all_steps_img = False  # applicable for mode='image'

    # parameter for tuning
    eta = 50  # threshold for filtering edge map predicted by DL model, edge-based section
    open_iter = 1  # number of iteration for performing morphological opening operation, edge-based section
    object_filter_size = 150  # objects smaller than this size will be filtered out, edge-based section
    dilate_iter = 1  # number of iteration for performing morphological dilation operation, edge-based section
    edge_area = 1000  # edge filter based on area
    threshold = 0.71  # threshold for pixel value filtering, color-based section
    if threshold < 0 or threshold > 1:
        print('Invalid pixel value threshold. Threshold range: 0 - 1')
    width_tolerance = 45  # tolerated distance for end point stabilization, x direction, color-based section
    height_tolerance = 45  # tolerated distance for end point stabilization, y direction, color-based section

    # basic cv parameters, unchangable, edge-based section
    a = np.zeros((3, 3), int)
    kernel = np.fill_diagonal(a, 1)
    kernel_dilate = np.ones((3, 3), np.uint8)
    rect_x_list = []
    rect_y_list = []
    rect_xw_list = []
    rect_yh_list = []

    # LDC parameter setting, unchangable, edge-based section
    mean_bgr = [103.939, 116.779, 123.68]
    img_height = 512
    img_width = 512
    label = None
    test_data = "CLASSIC"

    # model initialization, you may choose them in checkpoints folder
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    model = LDC().to(device)
    model.load_state_dict(
        torch.load('24_model.pth', map_location=device))
    model.eval()  # not recording gradient since we only infer images, not in training pipeline

    need_record = True  # only in video mode
    vid_writer = cv2.VideoWriter('003_comp_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))

    if mode == 'image':
        # load one image
        frame = cv2.imread(source)

        # color-based end-point estimation
        center_x = int(frame.shape[0] / 2)
        center_y = int(frame.shape[1] / 2)
        frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        frame_list.append(frame_gray.copy())  # append
        frame_inv = invert(frame_gray)
        frame_list.append(frame_inv.copy())  # append
        pMax = np.amax(frame)
        thres, result = cv2.threshold(frame_inv, (pMax * threshold), 255, cv2.THRESH_BINARY)
        frame_list.append(result.copy())  # append
        centre = find_contours_new(result)
        # print(len(centre))
        if centre is None:
            centre = [center_x, center_y]
        if (center_x - width_tolerance <= centre[0] <= center_x + width_tolerance) and (
                center_y - height_tolerance <= centre[1] <= center_y + height_tolerance):
            centre = [center_x, center_y]
        cv2.circle(frame, (int(centre[0]), int(centre[1])), 2, (0, 0, 255), 2, 8, 0)
        frame_list.append(frame.copy())  # append

        # DL-based outermost lumen estimation with filtering using color-based end-point
        image = frame.copy()
        im_shape = [image.shape[0], image.shape[1]]
        image, label = transform(img=image, gt=label, img_width=img_width, img_height=img_height, mean_bgr=mean_bgr)
        image_shape = [torch.tensor([im_shape[0]]), torch.tensor([im_shape[1]])]

        with torch.no_grad():
            img = image.to(device)
            preds = model(img)
            fuse, _ = return_processed_preds(preds, image_shape)
        frame_list.append(fuse.copy())  # append
        ret, fuse = cv2.threshold(fuse, eta, 255, cv2.THRESH_BINARY)
        frame_list.append(fuse.copy())  # append

        # postprocessing and filtering
        fuse = cv2.morphologyEx(fuse, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        frame_list.append(fuse.copy())  # append
        sk_image = img_as_float(fuse)
        sk_image = morphology.remove_small_objects(sk_image.astype(bool), min_size=object_filter_size, connectivity=2,
                                                   in_place=True)  # connectivity = 1 as 4-connect, = 2 as 8-connect
        sk_image = morphology.skeletonize(sk_image > 0)
        fuse = img_as_ubyte(sk_image)
        frame_list.append(fuse.copy())  # append

        fuse = cv2.dilate(fuse, kernel_dilate, iterations=dilate_iter)
        fuse = cv2.morphologyEx(fuse, cv2.MORPH_OPEN, kernel, iterations=1)
        frame_list.append(fuse.copy())  # append

        num_objects, labels, stats, centers = cv2.connectedComponentsWithStats(fuse, connectivity=8)

        output = np.zeros((fuse.shape[0], fuse.shape[1], 3), np.uint8)

        for i in range(1, num_objects):
            mask = labels == i
            output[:, :, 0][mask] = 0
            output[:, :, 1][mask] = 255 / i
            output[:, :, 2][mask] = 0

            x, y, w, h, area = stats[i]
            cx, cy = centers[i]

            # cv2.circle(output, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
            # rectangular bounding box
            if area > edge_area and x < centre[0] < x + w and y < centre[1] < y + h:
                # store x, y, x+w, y+h
                rect_x_list.append(x)
                rect_y_list.append(y)
                rect_xw_list.append(x + w)
                rect_yh_list.append(y + h)
                # draw all rectangle
                # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1, 8, 0)
                # cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 255, 0), 2, 8, 0)
                frame_list.append(frame.copy())  # append
        if len(rect_x_list) > 0:
            final_rect_x = int(min(rect_x_list))
            final_rect_y = int(min(rect_y_list))
            final_rect_xw = int(max(rect_xw_list))
            final_rect_yh = int(max(rect_yh_list))

            cv2.rectangle(frame, (final_rect_x, final_rect_y), (final_rect_xw, final_rect_yh), (255, 255, 0), 5, 8, 0)
            cv2.circle(frame, (int(final_rect_x + (final_rect_xw - final_rect_x) / 2), int(final_rect_y + (final_rect_yh - final_rect_y) / 2)), 2, (0, 255, 0), 2, 8, 0)
            frame_list.append(frame.copy())  # append
        if save_all_steps_img and len(frame_list) > 0:
            for i in len(frame_list):
                cv2.imwrite('output' + str(i) + '.jpg', frame_list[i])
        cv2.imshow('output', frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            pass
        elif key == ord('s'):
            cv2.imwrite('output.jpg', frame.copy)
        cv2.destryoAllWindows()
    elif mode == 'video':
        # opencv camera stream loading
        cap = cv2.VideoCapture(source)
        center_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
        center_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # color-based end-point estimation
                center_x = int(frame.shape[0] / 2)
                center_y = int(frame.shape[1] / 2)
                frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                frame_inv = invert(frame_gray)
                pMax = np.amax(frame)
                thres, result = cv2.threshold(frame_inv, (pMax * threshold), 255, cv2.THRESH_BINARY)
                centre = find_contours_new(result)
                print(len(centre))
                if centre is None:
                    centre = [center_x, center_y]
                if (center_x - width_tolerance <= centre[0] <= center_x + width_tolerance) and (
                        center_y - height_tolerance <= centre[1] <= center_y + height_tolerance):
                    centre = [center_x, center_y]
                cv2.circle(frame, (int(centre[0]), int(centre[1])), 2, (0, 0, 255), 2, 8, 0)

                # DL-based outermost lumen estimation with filtering using color-based end-point
                image = frame.copy()
                im_shape = [image.shape[0], image.shape[1]]
                image, label = transform(img=image, gt=label, img_width=img_width, img_height=img_height,
                                         mean_bgr=mean_bgr)
                image_shape = [torch.tensor([im_shape[0]]), torch.tensor([im_shape[1]])]

                with torch.no_grad():
                    img = image.to(device)
                    preds = model(img)
                    fuse, _ = return_processed_preds(preds, image_shape)

                ret, fuse = cv2.threshold(fuse, eta, 255, cv2.THRESH_BINARY)

                # postprocessing and filtering
                fuse = cv2.morphologyEx(fuse, cv2.MORPH_OPEN, kernel, iterations=open_iter)
                sk_image = img_as_float(fuse)
                sk_image = morphology.remove_small_objects(sk_image.astype(bool), min_size=object_filter_size,
                                                           connectivity=2,
                                                           in_place=True)  # connectivity = 1 as 4-connect, = 2 as 8-connect
                sk_image = morphology.skeletonize(sk_image > 0)
                fuse = img_as_ubyte(sk_image)

                fuse = cv2.dilate(fuse, kernel_dilate, iterations=dilate_iter)
                fuse = cv2.morphologyEx(fuse, cv2.MORPH_OPEN, kernel, iterations=1)

                num_objects, labels, stats, centers = cv2.connectedComponentsWithStats(fuse, connectivity=8)

                output = np.zeros((fuse.shape[0], fuse.shape[1], 3), np.uint8)

                for i in range(1, num_objects):
                    mask = labels == i
                    output[:, :, 0][mask] = 0
                    output[:, :, 1][mask] = 255 / i
                    output[:, :, 2][mask] = 0

                    x, y, w, h, area = stats[i]
                    cx, cy = centers[i]

                    # cv2.circle(output, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
                    # rectangular bounding box
                    if area > edge_area and x < centre[0] < x + w and y < centre[1] < y + h:
                        # store x, y, x+w, y+h
                        rect_x_list.append(x)
                        rect_y_list.append(y)
                        rect_xw_list.append(x + w)
                        rect_yh_list.append(y + h)
                        # draw all rectangle
                        # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1, 8, 0)
                        # cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 255, 0), 2, 8, 0)

                if len(rect_x_list) > 0:
                    final_rect_x = int(min(rect_x_list))
                    final_rect_y = int(min(rect_y_list))
                    final_rect_xw = int(max(rect_xw_list))
                    final_rect_yh = int(max(rect_yh_list))

                    cv2.rectangle(frame, (final_rect_x, final_rect_y), (final_rect_xw, final_rect_yh), (255, 255, 0), 5, 8, 0)
                    cv2.circle(frame, (int(final_rect_x + (final_rect_xw - final_rect_x) / 2), int(final_rect_y + (final_rect_yh - final_rect_y) / 2)), 2, (0, 255, 0), 2, 8, 0)

                cv2.imshow('output', frame)
                if need_record:
                    vid_writer.write(frame)

                # for each frame, reset the rectangle list
                rect_x_list = []
                rect_y_list = []
                rect_xw_list = []
                rect_yh_list = []

                key = cv2.waitKey(30)
                if key == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
