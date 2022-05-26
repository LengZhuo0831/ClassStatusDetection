import warnings
import numpy as np
import torch
import math
from torchvision import transforms
import cv2
from models.dectect import AntiSpoofPredict
from pfld.pfld import PFLDInference
from PIL import Image

warnings.filterwarnings('ignore')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor,Normalize,Compose
from lz_model import LZ_CNN, EmotionResNet, BasicBlock

def get_num(point_dict,name,axis):
    num = point_dict.get(f'{name}')[axis]
    num = float(num)
    return num

def cross_point(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1)
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return distance



class Model:
    def __init__(self):
        model_path = "./checkpoint/snapshot/checkpoint.pth.tar"
        checkpoint = torch.load(model_path, map_location=device)
        plfd_backbone = PFLDInference().to(device)
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        plfd_backbone.eval()
        self.plfd_backbone = plfd_backbone.to(device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.model_test = AntiSpoofPredict(device_id=0)

        self.mouth_model = LZ_CNN(2).to(device)
        self.eye_model = LZ_CNN(2).to(device)
        self.mouth_model.load_state_dict(torch.load("checkpoint/mouth_best.pth",map_location=device))
        self.eye_model.load_state_dict(torch.load("checkpoint/eye_best.pth",map_location=device))
        self.mouth_model.eval()
        self.eye_model.eval()

        mu, st = 0, 255
        self.emotion_model_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.TenCrop(40),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
        ])
        self.emotion_model = EmotionResNet(BasicBlock, [2, 2, 2, 2]).to(device)
        self.emotion_model.load_state_dict(torch.load('./checkpoint/best_checkpoint.tar',map_location=device)['model_state_dict'])
        self.emotion_model.eval()
        self.emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    def get_landMarks(self, face):
        input = cv2.resize(face, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = self.transform(input).unsqueeze(0).to(device)
        _, landmarks = self.plfd_backbone(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
        point_dict = {}
        i = 0
        for (x, y) in pre_landmark.astype(np.float32):
            point_dict[f'{i}'] = [x, y]
            i += 1

        return point_dict

    def cropImage(self,img,return_box=False):
        height, width = img.shape[:2]
        model_test = self.model_test
        image_bbox = model_test.get_bbox(img)
        x1 = image_bbox[0]
        y1 = image_bbox[1]
        x2 = image_bbox[0] + image_bbox[2]
        y2 = image_bbox[1] + image_bbox[3]
        w = x2 - x1
        h = y2 - y1

        size = int(max([w, h]))
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped = img[int(y1):int(y2), int(x1):int(x2)]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        cropped = cv2.resize(cropped, (112, 112))
        if return_box:
            return cropped,(int(y1), int(y2), int(x1), int(x2))
        return cropped

    def cal_roll_pitch_roll(self,point_dict):
        point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
        point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
        point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
        crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
        yaw_mean = point_point(point1, point31) / 2
        yaw_right = point_point(point1, crossover51)
        yaw = (yaw_mean - yaw_right) / yaw_mean
        yaw = int(yaw * 71.58 + 0.7037)

        # pitch
        pitch_dis = point_point(point51, crossover51)
        if point51[1] < crossover51[1]:
            pitch_dis = -pitch_dis
        pitch = int(1.497 * pitch_dis + 18.97)

        # roll
        roll_tan = abs(get_num(point_dict, 60, 1) - get_num(point_dict, 72, 1)) / abs(
            get_num(point_dict, 60, 0) - get_num(point_dict, 72, 0))
        roll = math.atan(roll_tan)
        roll = math.degrees(roll)
        if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
            roll = -roll
        roll = int(roll)

        return yaw, pitch, roll


    def drawMarksOnImage(self,ori_img,box,landmarks,with_idx=False,upscale=1):
        scale_y, scale_x = (box[1]-box[0])/112,(box[3]-box[2])/112
        for key,pos in landmarks.items():
            # print(item)
            x = (int)(pos[0]*scale_x+box[2])
            y = (int)(pos[1]*scale_y+box[0])
            cv2.circle(ori_img,(x,y),2,(240,32,160))
        if upscale!=1:
            H,W = ori_img.shape[:2]
            ori_img = cv2.resize(ori_img,(upscale*W,upscale*H))
        if with_idx:
            for key, pos in landmarks.items():
                x = (int)(pos[0] * scale_x + box[2])*upscale
                y = (int)(pos[1] * scale_y + box[0])*upscale
                cv2.putText(ori_img,str(key),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

        return ori_img

    def CropEyeAndMouth(self,ori_img,box,landmarks,mode = 'all'):

        landmarks_arr = []
        for k, coord in landmarks.items():
            landmarks_arr.append(coord)
        landmarks_arr = np.array(landmarks_arr)
        # mouth: 76-95
        # left eye: 60-67, 96
        # right eye: 68-75, 97
        def get_box_by_coords(coors,boders=(0,0,0,5)):
            coors_x = coors[:,0]
            coors_y = coors[:,1]
            # 横向
            min_x,max_x = min(coors_x)-boders[0],max(coors_x)+boders[1]
            # 纵向
            min_y,max_y = min(coors_y)-boders[2],max(coors_y)+boders[3]
            min_x,min_y = max(min_x,0),max(min_y,0)
            max_x,max_y = min(112,max_x),min(112,max_y)
            return (min_x, min_y, max_x, max_y)

        def trans_to_ori_coord(box, pos):
            scale_y, scale_x = (box[1] - box[0]) / 112, (box[3] - box[2]) / 112
            x = (int)(pos[0] * scale_x + box[2])
            y = (int)(pos[1] * scale_y + box[0])
            return (x,y)

        if mode=='mouth' or 'all':
            mouth_box = get_box_by_coords(landmarks_arr[76:96])
            x1,y1 = trans_to_ori_coord(box,mouth_box[:2])
            x2,y2 = trans_to_ori_coord(box,mouth_box[2:])
            mouth = ori_img[y1:y2,x1:x2]
            if mouth.shape[0]>1 and mouth.shape[1]>1:
                mouth = cv2.resize(mouth,(40,20))
                mouth = cv2.cvtColor(mouth,cv2.COLOR_BGR2RGB)
            else:
                mouth=None


        if mode=='eye' or 'all':
            left_eye_box = get_box_by_coords(landmarks_arr[60:68],boders=(3,3,3,3))
            x1, y1 = trans_to_ori_coord(box, left_eye_box[:2])
            x2, y2 = trans_to_ori_coord(box, left_eye_box[2:])
            left_eye = ori_img[y1:y2, x1:x2]
            if left_eye.shape[0]>1 and left_eye.shape[1]>1:
                left_eye = cv2.resize(left_eye, (40, 20))
                left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
            else:
                left_eye=None

            right_eye_box = get_box_by_coords(landmarks_arr[68:75],boders=(3,3,3,3))
            x1, y1 = trans_to_ori_coord(box, right_eye_box[:2])
            x2, y2 = trans_to_ori_coord(box, right_eye_box[2:])
            right_eye = ori_img[y1:y2, x1:x2]
            if right_eye.shape[0]>1 and right_eye.shape[1]>1:
                right_eye = cv2.resize(right_eye, (40, 20))
                right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
            else:
                right_eye=None
            # plt.imsave('../data/eye_clip/' + name[:-4] +"_l.png", left_eye)
            # plt.imsave('../data/eye_clip/' + name[:-4] +"_r.png", right_eye)

        if mode=='all':
            return mouth, left_eye, right_eye

    def ImagePipeline3(self,ori_img,show=False,draw_marks=False,save=False,return_landmarks=True, draw_facebox = True, just_return_status=False):
        crop, box = self.cropImage(ori_img, return_box=True)
        landmarks = self.get_landMarks(crop)

        # yaw, pitch, roll
        yaw, pitch, roll = self.cal_roll_pitch_roll(landmarks)
        yaw_comment = f"turn right {yaw} degree" if yaw > 0 else f"turn left {-yaw} degree"
        pitch_comment = f"down {pitch} degree" if pitch > 0 else f"up {-pitch} degree"
        roll_comment = f"roll left {roll} degree" if roll > 0 else f"roll right {-roll} degree"

        # mouth and eye
        mouth,left_eye,right_eye = self.CropEyeAndMouth(ori_img,box,landmarks,"all")

        transform = Compose([
            ToTensor(),
            Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                      std=[58.395 / 255, 57.12 / 255, 57.375 / 255])])
        mouth_open = False
        eye_close = False
        if mouth!=None:
            mouth = transform(mouth)[None]
            out_mouth = self.mouth_model(mouth).argmax(1).cpu().numpy()[0]
            mouth_open = False if out_mouth == 0 else True
        if left_eye!=None and right_eye!=None:
            left_eye = transform(left_eye)[None]
            right_eye = transform(right_eye)[None]
            left_eye_out = self.eye_model(left_eye).argmax(1).cpu().numpy()[0]
            right_eye_out = self.eye_model(right_eye).argmax(1).cpu().numpy()[0]
            eye_close = False if left_eye_out==0 and right_eye_out==0 else True
            mouth_coment = "mouth open" if mouth_open else "mouth close"
            eye_coment = "eye open" if not eye_close else "eye close"

        # emotion
        gray_crop = cv2.cvtColor(cv2.resize(crop,(48,48)),cv2.COLOR_BGR2GRAY)
        gray_crop = Image.fromarray(gray_crop)
        gray_crop = self.emotion_model_transform(gray_crop)
        emotion_out = self.emotion_model(gray_crop).argmax(1).cpu().numpy()[0]
        emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        emo = 'eomotion: '+emotion_mapping[emotion_out]

        # return all status
        if just_return_status:
            return yaw,pitch,roll,eye_close,mouth_open,emotion_out

        # put text on image
        cv2.putText(ori_img, yaw_comment, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(ori_img, pitch_comment, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(ori_img, roll_comment, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(ori_img, mouth_coment, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(ori_img, eye_coment, (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(ori_img, emo, (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if draw_facebox:
            cv2.rectangle(ori_img,(box[2],box[0]),(box[3],box[1]),(255,255,0),2)

        if draw_marks:
            ori_img = self.drawMarksOnImage(ori_img, box, landmarks)

        rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        if show:
            plt.imshow(rgb)
            plt.show()
            plt.pause(0.1)

        if save:
            plt.imsave('landmark_ids.png', rgb)

        if return_landmarks:
            return ori_img, landmarks

        return ori_img

    def VideoPipeline3(self, video_path, video_out_path="./video/result.avi", draw_marks=False, frame_out_path=None):
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        # fps = 30
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames_count = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        print("fps:", fps, "size:", size, "frames count: ", frames_count)
        if video_out_path != None:
            videoWriter = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
        idx = 0
        frame2landmarks = dict()
        while videoCapture.isOpened():
            idx += 1
            success, img = videoCapture.read()
            if not success:
                break
            out, marks = self.ImagePipeline3(img, show=False, draw_marks=draw_marks, return_landmarks=True)
            if video_out_path != None:
                videoWriter.write(out)
            frame2landmarks[idx] = marks
            print(idx, '/', frames_count)
            if frame_out_path != None:
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                plt.imsave(frame_out_path + str(idx - 1) + '.png', out)
            # if idx>20:
            #     break
        return frame2landmarks

    def visualize(self,ori_img,draw_marks=False,drawmarkid=False,draw_facebox = True,write_emo=True,write_mouth_and_eye=True,draw_face_dir=True,save_path=None,show=True):

        crop, box = self.cropImage(ori_img, return_box=True)
        landmarks = self.get_landMarks(crop)

        # yaw, pitch, roll
        yaw, pitch, roll = self.cal_roll_pitch_roll(landmarks)

        # mouth and eye
        mouth,left_eye,right_eye = self.CropEyeAndMouth(ori_img,box,landmarks,"all")
        transform = Compose([
            ToTensor(),
            Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                      std=[58.395 / 255, 57.12 / 255, 57.375 / 255])])
        mouth_open = False
        eye_close = False
        left_eye_out = 0
        right_eye_out = 0
        if isinstance(mouth,np.ndarray) and mouth.shape[:2]==(20,40):
            mouth = transform(mouth)[None].to(device)
            out_mouth = self.mouth_model(mouth).argmax(1).cpu().numpy()[0]
            mouth_open = False if out_mouth == 0 else True
        if isinstance(left_eye,np.ndarray) and left_eye.shape[:2]==(20,40) and isinstance(right_eye,np.ndarray) and right_eye.shape[:2] ==(20,40):
            left_eye = transform(left_eye)[None].to(device)
            right_eye = transform(right_eye)[None].to(device)
            left_eye_out = self.eye_model(left_eye).argmax(1).cpu().numpy()[0]
            right_eye_out = self.eye_model(right_eye).argmax(1).cpu().numpy()[0]
            eye_close = False if left_eye_out == 0 and right_eye_out == 0 else True

        # emotion
        gray_crop = cv2.cvtColor(cv2.resize(crop,(48,48)),cv2.COLOR_BGR2GRAY)
        gray_crop = Image.fromarray(gray_crop)
        gray_crop = self.emotion_model_transform(gray_crop).to(device)
        emotion_out = self.emotion_model(gray_crop).argmax(1).cpu().numpy()[0]
        emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

        # draw face box
        if draw_facebox:
            cv2.rectangle(ori_img, (box[2], box[0]), (box[3], box[1]), (255, 255, 0), 2)

        # draw landmarks
        if draw_marks:
            ori_img = self.drawMarksOnImage(ori_img, box, landmarks, with_idx=drawmarkid)

        # draw face dir
        if draw_face_dir:
            # 鼻子尖，54
            pos = landmarks['54']
            scale_y, scale_x = (box[1] - box[0]) / 112, (box[3] - box[2]) / 112
            x = (int)(pos[0] * scale_x + box[2])
            y = (int)(pos[1] * scale_y + box[0])
            # 绘制鼻尖
            if not draw_marks:
                cv2.circle(ori_img, (x, y), 4, (255, 255, 0))
            yaw1, pitch1, roll1 = yaw*math.pi/180, pitch*math.pi/180, roll*math.pi/180     # 偏航 俯仰 翻滚
            arrow_length=100
            dy, dx, dz = arrow_length*math.sin(yaw1),arrow_length*math.sin(pitch1),arrow_length*math.sin(roll1)
            dy, dx, dz = (int)(dy), (int)(dx), (int)(dz)
            cv2.arrowedLine(ori_img,(x,y),(x-dy,y+dx),(36,55,135),2)

        # put emo text on image
        if write_emo:
            emocolors = ((0,0,255),(0,97,255),(135,138,128),(0,255,127),(192,192,192),(0,215,255),(15,94,56))
            cv2.putText(ori_img, emotion_mapping[emotion_out], (box[2]+10,box[0]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, emocolors[emotion_out], 2)

        if write_mouth_and_eye:
            if mouth_open:
                pos = landmarks['76']
                scale_y, scale_x = (box[1] - box[0]) / 112, (box[3] - box[2]) / 112
                x = (int)(pos[0] * scale_x + box[2])
                y = (int)(pos[1] * scale_y + box[0])
                cv2.putText(ori_img,"open",(x-10,y),cv2.FONT_HERSHEY_SIMPLEX,1,(15,54,138),2)
            if left_eye_out==1:
                pos = landmarks['60']
                scale_y, scale_x = (box[1] - box[0]) / 112, (box[3] - box[2]) / 112
                x = (int)(pos[0] * scale_x + box[2])
                y = (int)(pos[1] * scale_y + box[0])
                cv2.putText(ori_img, "close", (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (15, 54, 138), 2)
            if right_eye_out==1:
                pos = landmarks['68']
                scale_y, scale_x = (box[1] - box[0]) / 112, (box[3] - box[2]) / 112
                x = (int)(pos[0] * scale_x + box[2])
                y = (int)(pos[1] * scale_y + box[0])
                cv2.putText(ori_img, "close", (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (15, 54, 138), 2)


        rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        if show:
            plt.imshow(rgb)
            plt.show()
            plt.pause(0.1)

        if save_path != None:
            plt.imsave(save_path, rgb)

        return yaw,pitch,roll,eye_close,mouth_open,emotion_out,ori_img

    def save_mouth_eye(self,ori_img,prefix,_eye='eye_open',_mouth='mouth_close'):
        crop, box = self.cropImage(ori_img, return_box=True)
        landmarks = self.get_landMarks(crop)

        # mouth and eye
        mouth, left_eye, right_eye = self.CropEyeAndMouth(ori_img, box, landmarks, "all")

        # mouth = cv2.cvtColor(mouth,cv2.COLOR_BGR2RGB)
        # left_eye = cv2.cvtColor(left_eye,cv2.COLOR_BGR2RGB)
        # right_eye = cv2.cvtColor(right_eye,cv2.COLOR_BGR2RGB)

        # save
        if isinstance(mouth, np.ndarray) and mouth.shape[:2] == (20, 40):
            plt.imsave('data/'+_mouth+'/'+prefix+'_mouth.jpg',mouth)
        if isinstance(left_eye, np.ndarray) and left_eye.shape[:2] == (20, 40):
            plt.imsave('data/'+_eye+'/'+prefix+'_leye.jpg', left_eye)
        if isinstance(right_eye, np.ndarray) and right_eye.shape[:2] == (20, 40):
            plt.imsave('data/'+_eye+'/'+prefix+'_reye.jpg', right_eye)











