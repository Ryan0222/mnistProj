import sys
import tensorflow as tf
from flask import Flask, request
from PIL import Image, ImageFilter
from redis import Redis, RedisError
import os
import socket
from cassandra.cluster import Cluster
import time

redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, "model4/model")



cluster = Cluster(contact_points=["172.17.0.1"],port=9042)
session = cluster.connect()

session.execute("create KEYSPACE if not exists mnist_database WITH replication = {'class':'SimpleStrategy', 'replication_factor': 2};")
session.execute("use mnist_database")
session.execute("create table if not exists mnist(id uuid, digits int, image_name text, upload_time timestamp, primary key(id));")


@app.route("/prediction", methods=['GET','POST'])
def predictint():
    imname = request.files["file"]  
    file_name = request.files["file"].filename
    imvalu = prepareImage(imname)
    prediction = tf.argmax(y,1)
    pred = prediction.eval(feed_dict={x: [imvalu]}, session=sess)
  
    uploadtime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
   
    session.execute("INSERT INTO mnist(id, digits, image_name, upload_time) values(uuid(), %s, %s, %s)",[int(str(pred[0])), file_name, uploadtime])
    return "The number is: %s" % str(pred[0])

def prepareImage(i):
    im = Image.open(i).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))
    if width > height: 
        
        nheight = int(round((20.0/width*height),0)) 
        if (nheight == 0): 
            nheight = 1
        
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) 
        newImage.paste(img, (4, wtop)) 
    else:
       
        nwidth = int(round((20.0/height*width),0)) 
        if (nwidth == 0): 
            nwidth = 1
         
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) 
        newImage.paste(img, (wleft, 4)) 
    
    tv = list(newImage.getdata()) 
    
  
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva 

@app.route('/')
def index():
    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"

    html = '''
	<!doctype html>
	<html>
	<body>
	<form action='/prediction' method='post' enctype='multipart/form-data'>
  		<input type='file' name='file'>
	<input type='submit' value='Upload'>
	</form>
	'''   
    return html.format(name=os.getenv("NAME", "MNIST"), hostname=socket.gethostname(), visits=visits) 

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)

