
import time
import tkinter.ttk
from tkinter import *
from tkinter import filedialog, messagebox
import cv2
import sys
sys.path.append("../..")
import torch
from PIL import Image, ImageTk

from lz_tests import Model
import random


global emo
global grade
global head_
global eye_
global mouse
global final
emo = "正常"
grade = 100
head_ = "端正"
eye_ = "正常"
mouse = "正常"
final = 100



class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        global emo
        global grade
        global head_
        global eye_
        global mouse
        global final

        self.num = 0       #当前的帧数
        self.size = (48,48)  #裁剪的大小
        self.fre = 5  #采样率
        self.refresh_rate = 300

        self.video = None

        self.model = Model()
        self.emotion_mapping = {0: '生气', 1: '恶心', 2: '害怕', 3: '高兴', 4: '悲伤', 5: '惊讶', 6: '正常'}
        self.emotion_scores = {0: '0', 1: '10', 2: '20', 3: '90', 4: '50', 5: '70', 6: '100'}

        self.root = Tk()
        self.root.title('学生课堂专注度检测系统')
        self.root.geometry('1200x800')
        self.root.maxsize(1200,800)
        self.root.minsize(1200,800)

        title = Label(self.root, text="学生课堂专注度检测系统", font=('STZhongsong', 24), pady=2, bd=12, bg="#B0C4DE", fg="Black", relief=GROOVE)
        title.pack(fill=X)

        self.paint_tools = Frame(self.root,width=200,height=600,relief=RIDGE,borderwidth=2)
        self.paint_tools.place(x=0,y=100)

        self.p = Label(self.paint_tools, text="当前表情:",borderwidth=0,font=('STZhongsong',10))
        self.p.place(x=5,y=20)
        self.En1 = Entry(self.paint_tools,bd=0, width=6,font=('STZhongsong'))
        self.En1.place(x=100,y=20)


        self.b = Label(self.paint_tools,borderwidth=0,text='表情打分:',font=('STZhongsong',10))
        self.b.place(x=5,y=100)
        self.En2 = Entry(self.paint_tools,bd=0, width=6,font=('STZhongsong'))
        self.En2.place(x=100,y=100)

        self.cl = Label(self.paint_tools, text='头部朝向:',font=('STZhongsong',10))
        self.cl.place(x=5,y=180)
        self.En3 = Entry(self.paint_tools,bd=0, width=6,font=('STZhongsong'))
        self.En3.place(x=100,y=180)

        self.e = Label(self.paint_tools, text='眼睛状态:',font=('STZhongsong',10))
        self.e.place(x=5,y=260)
        self.En4 = Entry(self.paint_tools,bd=0, width=6,font=('STZhongsong'))
        self.En4.place(x=100,y=260)

        self.e = Label(self.paint_tools, text='嘴巴状态:',font=('STZhongsong',10))
        self.e.place(x=5,y=340)
        self.En5 = Entry(self.paint_tools,bd=0, width=6,font=('STZhongsong'))
        self.En5.place(x=100,y=340)

        self.e = Label(self.paint_tools, text='最终判断:',font=('STZhongsong',10,'bold'))
        self.e.place(x=5,y=450)
        self.En6 = Entry(self.paint_tools,bd=0, width=10,font=('STZhongsong'))
        self.En6.place(x=50,y=500)

        self.video_logo = ImageTk.PhotoImage(Image.open('videoAndImage/video.png'))
        self.e = Label(self.root, image = self.video_logo)
        self.e.place(x=450,y=720)
        self.video_button = Button(self.root,text='选择本地视频',font=('STZhongsong',10), borderwidth=2,command = self.getvideo)
        self.video_button.place(x=500,y=720)

        self.cam_logo = ImageTk.PhotoImage(Image.open('videoAndImage/cam.png'))
        self.e = Label(self.root, image = self.cam_logo)
        self.e.place(x=650,y=720)
        self.cam_button = Button(self.root,text='选择摄像头',font=('STZhongsong',10), borderwidth=2,command=self.getcam)
        self.cam_button.place(x=700,y=720)

        self.fig_logo = ImageTk.PhotoImage(Image.open('videoAndImage/fig.png'))
        self.e = Label(self.root, image = self.fig_logo)
        self.e.place(x=850,y=720)
        self.fig_button = Button(self.root,text='选择本地图片',font=('STZhongsong',10), borderwidth=2,command=self.getfig)
        self.fig_button.place(x=900,y=720)

        self.c = Canvas(self.root, bg='white', width=800, height=600,relief=RIDGE,borderwidth=0)
        self.c.place(x=300,y=100)


        global photo
        photo = None
        self.label03 = Label(self.root, image = photo)
        self.label03.place(x=300,y=100)
        # self.label03.grid(column=0, row=0)

        self.canv_size = (600,800)

        # self.setup()
        self.root.after(1, self.update_node)
        self.root.mainloop()

    def update_node(self):
        # print("刷新一次")
        global emo
        global grade
        global head_
        global eye_
        global mouse
        global final

        self.En1.delete(0, 10)
        self.En1.insert(0, str(emo))  # delete
        self.En1.update()

        self.En2.delete(0, 10)
        self.En2.insert(0, grade)  # delete
        self.En2.update()

        self.En3.delete(0, 10)
        self.En3.insert(0, str(head_))  # delete
        self.En3.update()

        self.En4.delete(0, 10)
        self.En4.insert(0, str(eye_))  # delete
        self.En4.update()

        self.En5.delete(0, 10)
        self.En5.insert(0, str(mouse))  # delete
        self.En5.update()

        self.En6.delete(0, 10)
        self.En6.insert(0, str(final)+"(认真)" if final>=60 else str(final)+"(不认真)")  # delete
        self.En6.update()

        self.root.after(self.refresh_rate, self.update_node)

    #从文件夹中选取相关的视频
    def getvideo(self):
        global emo
        global grade
        global head_
        global eye_
        global mouse
        global final

        if self.video != None:
            self.video.release()
        self.num=0

        file_path = filedialog.askopenfilename()
        self.video = cv2.VideoCapture(file_path)
        ret, frame = self.video.read()

        fps = self.video.get(cv2.CAP_PROP_FPS)
        size = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        videoWriter = cv2.VideoWriter('videoAndImage/tmp/tmp.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, size)

        progbar = tkinter.ttk.Progressbar(self.label03)
        progbar.pack(padx=10,pady=10)
        progbar['maximum']=frames_count
        progbar['value']=0

        res = []
        while ret:
            self.num += 1
            if ret==True:

                yaw, pitch, roll, eye_close, mouth_open, emotion_out, img = self.model.visualize(frame,show=False,write_emo=True,write_mouth_and_eye=True,draw_marks=True)

                videoWriter.write(img)
                progbar['value'] = self.num

                self.label03.update()
                _emo = self.emotion_mapping[emotion_out]
                _grade = self.emotion_scores[emotion_out]
                _final = (int)(_grade)
                if abs(yaw) < 20 and abs(pitch) < 20 and abs(roll) < 20:
                    _head_ = "正常"
                else:
                    _head_ = "不正"
                    _final -= 5
                _eye_ = "正常" if not eye_close else "闭眼"
                _mouse = "正常" if not mouth_open else "张嘴"
                if mouth_open:
                    _final -= 5
                if eye_close:
                    _final -= 45
                res.append((_emo,_grade,_final,_head_,_eye_,_mouse))

            if self.video.isOpened():
                ret, frame = self.video.read()
            else:
                ret = False

        progbar.destroy()
        self.num=0

        self.video = cv2.VideoCapture('videoAndImage/tmp/tmp.avi')
        ret, frame = self.video.read()
        while ret:
            global photo
            global emo
            global grade
            global head_
            global eye_
            global mouse
            global final
            emo, grade, final, head_, eye_, mouse = res[self.num]
            self.num+=1
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((800, 600))
            photo = ImageTk.PhotoImage(img)  # 实际上是把这个变成全局变量
            self.label03.configure(image = photo)
            self.label03.image = photo
            self.label03.update()
            time.sleep(0.02)
            if self.video.isOpened():
                ret, frame = self.video.read()
            else:
                ret = False


    #从摄像头中直接录取
    def getcam(self):
        global emo
        global grade
        global head_
        global eye_
        global mouse
        global final

        self.video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.video.open(0)
        ret, frame = self.video.read()
        while ret:
            self.num += 1
            h,w,_ = frame.shape
            # size = min(h,w)
            # frame = frame[h//2-size//2:h//2-size//2+size,w//2-size//2:w//2-size//2+size]
            frame = cv2.resize(frame,(800,600))
            if ret == True and (self.num%self.fre==1):
                yaw, pitch, roll, eye_close, mouth_open, emotion_out, img = self.model.visualize(frame,show=False,draw_marks=True,draw_facebox=True,write_emo=True,write_mouth_and_eye=True)
                emo = self.emotion_mapping[emotion_out]
                grade = self.emotion_scores[emotion_out]
                final = (int)(grade)
                if abs(yaw) < 20 and abs(pitch) < 20 and abs(roll) < 20:
                    head_ = "正常"
                else:
                    head_ = "不正"
                    final -= 5
                eye_ = "正常" if not eye_close else "闭眼"
                mouse = "正常" if not mouth_open else "张嘴"
                if mouth_open:
                    final -= 5
                if eye_close:
                    final -= 45

                img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).resize((800, 600))

                global photo
                photo = ImageTk.PhotoImage(img)  # 实际上是把这个变成全局变量
                self.label03.configure(image = photo)
                self.label03.image = photo
                self.label03.update()
                # time.sleep(0.02)

            if self.video.isOpened():
                ret, frame = self.video.read()
            else:
                ret = False

    #从图片中获取
    def getfig(self):
        global emo
        global grade
        global head_
        global eye_
        global mouse
        global final

        if self.video != None:
            self.video.release()

        file_path = filedialog.askopenfilename(title='选择文件', filetypes=[('All Files', '*')])
        if file_path=='':
            return
        frame = cv2.imread(file_path)


        yaw, pitch, roll, eye_close, mouth_open, emotion_out, img = self.model.visualize(frame, show=False)
        emo = self.emotion_mapping[emotion_out]
        grade = self.emotion_scores[emotion_out]
        final = (int)(grade)
        if abs(yaw) < 20 and abs(pitch) < 20 and abs(roll) < 20:
            head_ = "正常"
        else:
            head_ = "不正"
            final -= 5
        eye_ = "正常" if not eye_close else "闭眼"
        mouse = "正常" if not mouth_open else "张嘴"
        if eye_close or mouth_open:
            final -= 5

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((800, 600))

        global photo
        photo = ImageTk.PhotoImage(img)  # 实际上是把这个变成全局变量
        self.label03.configure(image=photo)
        self.label03.image = photo


    def getcam2(self):
        global emo

        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video.open(0)
        ret, frame = self.video.read()
        self.num = 0

        start_id = random.randint(0,1000000)

        while ret:
            self.num += 1
            if ret == True and (self.num%self.fre==1):

                # 采集脸和嘴的图片
                if self.num>60:
                    prefix = str(start_id+self.num)
                    self.model.save_mouth_eye(frame,prefix=prefix,_eye='eye_open',_mouth='mouth_open')

                # 摄像头
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((800, 600))

                global photo
                photo = ImageTk.PhotoImage(img)  # 实际上是把这个变成全局变量
                self.label03.configure(image=photo)
                self.label03.image = photo
                self.label03.update()
                if self.num>60:
                    emo = '采'+str(self.num)
                else:
                    emo = str(self.num)

            if self.video.isOpened():
                ret, frame = self.video.read()
            else:
                ret = False


        emo = '采集完毕'



if __name__ == '__main__':
    Paint()
