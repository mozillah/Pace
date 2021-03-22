
import os
import dlib
from skimage import io
import csv
import numpy as np
import shutil
import time
import cv2
import pandas as pd

#  Use frontal face detector of Dlib

path_photos_from_camera = "data/data_faces_from_camera/"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class Face_Register:
    
    # os.path.join("data/data_faces_from_camera/","")
    def __init__(self):

        self.font = cv2.FONT_ITALIC

        self.existing_faces_cnt = 0         # count for counting saved faces
        self.ss_cnt = 0                     # count for screen shots
        self.current_frame_faces_cnt = 0    # cnt for counting faces in current frame

        self.save_flag = 1                  # The flag to control if save
        self.press_n_flag = 0               # The flag to check if press 'n' before 's'

        # FPS
        self.frame_time = 0                 
        self.frame_start_time = 0
        self.fps = 0
        self.frame_cnt = 0


        # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.features_known_list = []
        # 存储录入人脸名字 / Save the name of faces in the database
        self.name_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标 / List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_centroid_list = []
        self.current_frame_centroid_list = []

        # 用来存储上一帧和当前帧检测出目标的名字 / List to save names of objects in frame N-1 and N
        self.last_frame_names_list = []
        self.current_frame_face_name_list = []

        # 上一帧和当前帧中人脸数的计数器 / cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离 / Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字 / Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征 / Save the features of people in current frame
        self.current_frame_face_features_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

    #  Make dir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save faces images and csv
        if os.path.isdir(path_photos_from_camera):
            pass
        else:
            os.mkdir(path_photos_from_camera)

    #  Delete the old data of faces
    def pre_work_del_old_face_folders(self):
        # "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(path_photos_from_camera+folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")

    # 如果有之前录入的人脸, 在之前 person_x 的序号按照 person_x+1 开始录入 / Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # 获取已录入的最后一个人脸序号 / Get the order of latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_num_list.append(int(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)

        # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入 / Start from person_1
        else:
            self.existing_faces_cnt = 0

    # 获取处理之后 stream 的帧数 / Update FPS of video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    #  PutText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some notes
        cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


    # 返回单张图像的 128D 特征 / Return 128D features for single image
    # Input:    path_img           <class 'str'>
    # Output:   face_descriptor    <class 'dlib.vector'>
    def return_128d_features(self,path_img):
        img_rd = io.imread(path_img)
        faces = detector(img_rd, 1)

        print("%-40s %-20s" % (" >> 检测到人脸的图像 / Image with faces detected:", path_img), '\n')

        # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
        # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
        if len(faces) != 0:
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
            print("no face")
        return face_descriptor


    # 返回 personX 的 128D 特征均值 / Return the mean value of 128D face descriptor for person X
    # Input:    path_faces_personX       <class 'str'>
    # Output:   features_mean_personX    <class 'numpy.ndarray'>
    def return_features_mean_personX(self,path_faces_personX):
        features_list_personX = []
        photos_list = os.listdir(path_faces_personX)
        if photos_list:
            for i in range(len(photos_list)):
                # 调用 return_128d_features() 得到 128D 特征 / Get 128D features for single image of personX
                print("%-40s %-20s" % (" >> 正在读的人脸图像 / Reading image:", path_faces_personX + "/" + photos_list[i]))
                features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
                # 遇到没有检测出人脸的图片跳过 / Jump if no face detected from image
                if features_128d == 0:
                    i += 1
                else:
                    features_list_personX.append(features_128d)
        else:
            print(" >> 文件夹内图像文件为空 / Warning: No images in " + path_faces_personX + '/', '\n')

        # 计算 128D 特征的均值 / Compute the mean
        # personX 的 N 张图像 x 128D -> 1 x 128D
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=int, order='C')
        return features_mean_personX


    def features_extraction_to_csv(self):
        # 获取已录入的最后一个人脸序号 / Get the order of latest person
        person_list = os.listdir(path_photos_from_camera)
        person_num_list = []
        for person in person_list:
            person_num_list.append(int(person.split('_')[-1]))
        person_cnt = max(person_num_list)

        with open("data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in range(person_cnt):
                # Get the mean/average features of face/personX, it will be a list with a length of 128D
                print(path_photos_from_camera + "person_" + str(person + 1))
                features_mean_personX = self.return_features_mean_personX(path_photos_from_camera + "person_" + str(person + 1))
                writer.writerow(features_mean_personX)
                print(" >> 特征均值 / The mean of features:", list(features_mean_personX), '\n')
            print("所有录入人脸数据存入 / Save all the features of faces registered into: data/features_all.csv")


    # 从 "features_all.csv" 读取录入人脸特征 / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.features_known_list.append(features_someone_arr)
                self.name_known_list.append("Person_" + str(i + 1))
            print("Faces in Database：", len(self.features_known_list))
            return 1
        else:
            print('##### Warning #####', '\n')
            print("'features_all.csv' not found!")
            print(
                "Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'",
                '\n')
            print('##### End Warning #####')
            return 0

    # 获取处理之后 stream 的帧数 / Get the fps of video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features


    # # 获取人脸 / Main process of face detection and saving
    # def process(self, stream):
    #     # 1. Create folders to save photos
    #     self.pre_work_mkdir()

    #     # 2. 删除 "/data/data_faces_from_camera" 中已有人脸图像文件 / Uncomment if want to delete the saved faces and start from person_1
    #     # if os.path.isdir(self.path_photos_from_camera):
    #     #     self.pre_work_del_old_face_folders()

    #     # 3. "/data/data_faces_from_camera" 
    #     self.check_existing_faces_cnt()

    #     while stream.isOpened():
    #         flag, img_rd = stream.read()        # Get camera video stream
    #         kk = cv2.waitKey(1)
    #         faces = detector(img_rd, 0)         # Use Dlib face detector

    #         # 4. Press 'n' to create the folders for saving faces
    #         if kk == ord('n'):
    #             self.existing_faces_cnt += 1
    #             current_face_dir = path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
    #             os.makedirs(current_face_dir)
    #             print('\n')
    #             print("Create folders: ", current_face_dir)

    #             self.ss_cnt = 0                 # Clear the cnt of screen shots
    #             self.press_n_flag = 1           # Pressed 'n' already

    #         # 5.Face detected
    #         if len(faces) != 0:
    #             # Show the ROI of faces
    #             for k, d in enumerate(faces):
    #                 #Compute the size of rectangle box
    #                 height = (d.bottom() - d.top())
    #                 width = (d.right() - d.left())
    #                 hh = int(height/2)
    #                 ww = int(width/2)

    #                 # 6. If the size of ROI > 480x640
    #                 if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
    #                     cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    #                     color_rectangle = (0, 0, 255)
    #                     save_flag = 0
    #                     if kk == ord('s'):
    #                         print("Please adjust your position")
    #                 else:
    #                     color_rectangle = (255, 255, 255)
    #                     save_flag = 1

    #                 cv2.rectangle(img_rd,
    #                               tuple([d.left() - ww, d.top() - hh]),
    #                               tuple([d.right() + ww, d.bottom() + hh]),
    #                               color_rectangle, 2)

    #                 # 7. Create blank image according to the size of face detected
    #                 img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

    #                 if save_flag:
    #                     # 8.Press 's' to save faces into local images
    #                     if kk == ord('s'):
    #                         # Check if you have pressed 'n'
    #                         if self.press_n_flag:
    #                             self.ss_cnt += 1
    #                             for ii in range(height*2):
    #                                 for jj in range(width*2):
    #                                     img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
    #                             cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
    #                             print("写入本地 / Save into：", str(current_face_dir) + "/img_face_" + str(self.ss_cnt) + ".jpg")
    #                         else:
    #                             print("请先按 'N' 来建文件夹, 按 'S' / Please press 'N' and press 'S'")

    #         self.current_frame_faces_cnt = len(faces)

    #         # 9.Add note on cv2 window
    #         self.draw_note(img_rd)

    #         # 10. Press 'q' to exit
    #         if kk == ord('q'):
    #             break

    #         # 11. Update FPS
    #         self.update_fps()

    #         cv2.namedWindow("camera", 1)
    #         cv2.imshow("camera", img_rd)








    def process(self, stream):
        # 1.Get faces known from "features.all.csv"
        self.pre_work_mkdir()
        test=False
        self.check_existing_faces_cnt()

        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                print(">>> Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # 2. 检测人脸 / Detect faces for frame X
                faces = detector(img_rd, 0)


                if kk == ord('n'):
                    self.existing_faces_cnt += 1
                    current_face_dir = path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
                    os.makedirs(current_face_dir)
                    print('\n')
                    print("Create folders: ", current_face_dir)

                    self.ss_cnt = 0                 # Clear the cnt of screen shots
                    self.press_n_flag = 1           # Pressed 'n' already

                if kk ==ord('p'):
                    test=True
                if kk == ord("l"):
                    test=False
                if len(faces) != 0 and test:
                # Show the ROI of faces
                    for k, d in enumerate(faces):
                        #Compute the size of rectangle box
                        height = (d.bottom() - d.top())
                        width = (d.right() - d.left())
                        hh = int(height/2)
                        ww = int(width/2)

                        # 6. If the size of ROI > 480x640
                        if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                            cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                            color_rectangle = (0, 0, 255)
                            save_flag = 0
                            if kk == ord('s'):
                                print("Please adjust your position")
                        else:
                            color_rectangle = (255, 255, 255)
                            save_flag = 1

                        cv2.rectangle(img_rd,
                                    tuple([d.left() - ww, d.top() - hh]),
                                    tuple([d.right() + ww, d.bottom() + hh]),
                                    color_rectangle, 2)

                        # 7. Create blank image according to the size of face detected
                        img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                        if save_flag:
                            # 8.Press 's' to save faces into local images
                            if kk == ord('s'):
                                # Check if you have pressed 'n'
                                if self.press_n_flag:
                                    self.ss_cnt += 1
                                    for ii in range(height*2):
                                        for jj in range(width*2):
                                            img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                                    cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                                    print("写入本地 / Save into：", str(current_face_dir) + "/img_face_" + str(self.ss_cnt) + ".jpg")
                                else:
                                    print("请先按 'N' 来建文件夹, 按 'S' / Please press 'N' and press 'S'")

                self.update_fps()



                self.current_frame_faces_cnt = len(faces)

                # 9.Add note on cv2 window
                self.draw_note(img_rd)


                if self.current_frame_face_name_list == ['Person_2', 'Person_2']:
                    break

                # Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)
                # Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                # update frame centroid list
                self.last_frame_centroid_list = self.current_frame_centroid_list
                self.current_frame_centroid_list = []
                print("   >>> current_frame_face_cnt: ", self.current_frame_face_cnt)

                # 2.1. if cnt not changes
                if self.current_frame_face_cnt == self.last_frame_face_cnt:
                    print("   >>> scene 1: 当前帧和上一帧相比没有发生人脸数变化 / no faces cnt changes in this frame!!!")
                    self.current_frame_face_position_list = []
                    if self.current_frame_face_cnt != 0:
                        # 2.1.1 Get ROI positions
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            # 计算矩形框大小 / Compute the size of rectangle box
                            height = (d.bottom() - d.top())
                            width = (d.right() - d.left())
                            hh = int(height / 2)
                            ww = int(width / 2)
                            cv2.rectangle(img_rd,
                                          tuple([d.left() - ww, d.top() - hh]),
                                          tuple([d.right() + ww, d.bottom() + hh]),
                                          (255, 255, 255), 2)

                    # multi-faces in current frames, use centroid tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 write names under ROI
                        cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                    self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                    cv2.LINE_AA)

                # 2.2 if cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    print("   >>> scene 2: 当前帧和上一帧相比人脸数发生变化 / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []

                    # 2.2.1 face cnt decrease: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        print("   >>> scene 2.1 人脸消失, 当前帧中没有人脸 / No guy in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                        self.current_frame_face_features_list = []

                    # 2.2.2 face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        print("   >>> scene 2.2 出现人脸，进行人脸识别 / Do face recognition for people detected in this frame")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_features_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 2.2.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                        for k in range(len(faces)):
                            print("      >>> For face " + str(k+1) + " in current frame:")
                            self.current_frame_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 2.2.2.2 每个捕获人脸的名字坐标 / Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 2.2.2.3 对于某张人脸，遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.features_known_list)):
                                # 如果 person_X 数据不为空
                                if str(self.features_known_list[i][0]) != '0.0':
                                    print("            >>> with person", str(i + 1), "the e distance: ", end='')
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_features_list[k],
                                        self.features_known_list[i])
                                    print(e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 2.2.2.4 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.name_known_list[similar_person_num]
                                print("            >>> recognition result for face " + str(k+1) +": "+ self.name_known_list[similar_person_num])
                            else:
                                print("            >>> recognition result for face " + str(k + 1) + ": " + "unknown")
                # 3. 生成的窗口添加说明文字 / Add note on cv2 window
                self.draw_note(img_rd)

                # 4. 按下 'q' 键退出 / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)
                print(">>> Frame ends\n\n")









    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_centroid_list[i], self.last_frame_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # 生成的 cv2 window 上面添加说明文字 / putText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some statements
        cv2.putText(img_rd, "Face recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            cv2.putText(img_rd, "Face " + str(i + 1), tuple(
                [int(self.current_frame_centroid_list[i][0]), int(self.current_frame_centroid_list[i][1])]), self.font,
                        0.8, (255, 190, 0),
                        1,
                        cv2.LINE_AA)

    def run(self):
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("head-pose-face-detection-female-and-male.mp4")
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
