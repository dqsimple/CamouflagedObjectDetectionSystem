import tkinter as tk
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.trainer import eval_mae, numpy2tensor
import cv2
import tkinter.filedialog
from PIL import Image, ImageTk
import torchvision.transforms as transforms


def picture():
    global select_file
    select_file = tk.filedialog.askopenfilename(title='选择文件')
    global image_file
    img = Image.open(select_file)
    img = img.resize((250, 250), Image.ANTIALIAS)
    image_file = ImageTk.PhotoImage(img)
    canvas.create_image(10, 10, anchor='nw', image=image_file)
    location.set(select_file)




def process():
    img = Image.open(select_file)
    img = img.convert('RGB')
    img_width = img.width
    img_height = img.height
    img_transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = img_transform(img).unsqueeze(0)
    _, cam = model(img)
    cam = F.upsample(cam, size=(img_height, img_width), mode='bilinear', align_corners=True)
    cam = cam.sigmoid().data.cpu().numpy().squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    #misc.imsave('./1', cam)
    global cam_predict
    misc.imsave('./Result/1.png', cam)
    cam_1 = Image.open('./Result/1.png')
    cam_1 = cam_1.resize((250, 250), Image.ANTIALIAS)
    #cam = Image.fromarray(cam)
    cam_predict = ImageTk.PhotoImage(cam_1)
    canvas.create_image(310, 10, anchor='nw', image=cam_predict)

def img_save():
    global save_path
    save_path = tk.filedialog.asksaveasfilename(title='选择保存路径')
    cam = Image.open('./Result/1.png')
    misc.imsave(save_path, cam)
model_path ='./Snapshot/2020-CVPR-SINet/SINet_40.pth'
model = SINet_ResNet50()
model.load_state_dict(torch.load(model_path))
model.eval()
window = tk.Tk()


window.title('Camouflaged Object Detection')

window.geometry('600x400')
location = tk.StringVar()
l_0 = tk.Label(window, text='图片路径', fg='black', font=('Arial', 12), width=8, height=1)
l_0.grid()
l_1 = tk.Label(window, textvariable=location, bg='white', fg='black', font=('Arial', 12), width=70, height=1)
l_1.grid(row=0, column=1)

canvas = tk.Canvas(window, bg='white', height=300, width=600)

load_button = tk.Button(window, text='上传图片', font=('Arial', 12), width=10, height=1, command=picture)
load_button.place(x=50, y=350, anchor='nw')
canvas.place(x=0, y=30)
#canvas.grid(row=1,column=2)
predict_button = tk.Button(window, text='显示结果', font=('Arial', 12), width=10, height=1, command=process)
predict_button.place(x=250, y=350, anchor='nw')
save_button = tk.Button(window, text='保存图片', font=('Arial', 12), width=10, height=1, command=img_save)
save_button.place(x=450, y=350, anchor='nw')
window.mainloop()