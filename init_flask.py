from flask import Flask, jsonify, request
from torchvision import transforms
from PIL import Image
import os
import io

from trainer import Continual
#from streaming.trainer import *

app = Flask(__name__)


RB_PATH = "./swap_data"

transform = transforms.Compose([#transforms.Resize(img_size),
                                        transforms.RandomCrop((32,32),padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ColorJitter(brightness=0.24705882352941178),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

#transform = transforms.Compose(
#               [transforms.ToTensor(),
#               transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#            ])

continual = Continual(batch_size=128, epochs=1, rb_size=2000, num_workers=0, swap=True,
                    opt_name="sgd", lr=0.1, lr_schedule=None, lr_decay=None,
                    sampling="ringbuffer", train_transform=transform, rb_path=RB_PATH, model="resnet18", 
                    agent_name="stream_er", mode="online")


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST': 
        try:
            img_file = request.files['file']
            
            img = Image.open(io.BytesIO(img_file.read()))
            img = img.convert('RGB')

            print(type(img))

            label = request.form['label']
            
            #label = request.form.getlist('label')
            
            #print(img, label)
        except:
            return "You should send both image file and label together.\n \
                Try 'curl -X POST -F label=(label_name) -F file=@(file_path) (server_adderess:port)\n"
        
        try:
            task_id = request.form['task_id']
            
            
            task_id = int(task_id)
        except:
            task_id = None
            pass

        
        return continual.send_stream_data(img, label, task_id)
        
app.run(port=5005)


#if __name__ == "__main__":
