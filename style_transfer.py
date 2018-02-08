import argparse, sys, pickle
import theano, lasagne
import numpy as np
import cv2

p = argparse.ArgumentParser()
p.add_argument("--choice", required=True, type=str, choices=set(('image', 'video')))
p.add_argument("--vgg19-path", required=False, type=str, default='vgg19_normalized.pkl')
p.add_argument("--content-img-path", required=False, type=str)
p.add_argument("--style-img-path", required=False, type=str)
p.add_argument("--target-img-path", required=False, type=str)
p.add_argument("--input-video-path", required=False, type=str)
p.add_argument("--target-video-path", required=False, type=str)
p.add_argument("---video-start-sec", required=False, type=int, default=0)
p.add_argument("--n-iters", required=False, type=int, default=1000)
p.add_argument("--learning-rate", required=False, type=float, default=2.0)
p.add_argument("--alpha", required=False, type=float, default=2e3)
p.add_argument("--beta", required=False, type=float, default=20e11)
p.add_argument("--resize-ratio", required=False, type=float, default=1.0)
args = p.parse_args()

choice = args.choice
n_iters = args.n_iters
learning_rate = args.learning_rate
alpha = args.alpha 
beta = args.beta
ratio = args.resize_ratio
content_img_path = args.content_img_path
style_img_path = args.style_img_path
target_img_path = args.target_img_path
input_video_path = args.input_video_path
target_video_path = args.target_video_path
video_start_sec = args.video_start_sec
style_img_path = args.style_img_path
vgg19_path = args.vgg19_path

if choice == 'image':
	if None in [content_img_path, style_img_path, target_img_path]:
		print('Not enough parameters.')
		sys.exit()
		
else:
	if None in [input_video_path, target_video_path, style_img_path]:
		print('Not enough parameters.')
		sys.exit()

vgg_layers = \
['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4']

content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

class loss_vgg_data(object):
	
	def __init__ (self, input, network, height, width):
		
		self.height = height
		self.width = width
		self.input = input
		
		random_data = np.ones((1, 3, height, width), dtype = theano.config.floatX)
		
		self.content_representation = []
		for content_layer in content_layers:
			content_data = data_network[content_layer].eval({self.input: random_data})
			self.content_representation.append(theano.shared(content_data))
			
		self.style_representation = []
		for style_layer in style_layers:
			style_data = data_network[style_layer].eval({self.input: random_data})
			self.style_representation.append(theano.shared(style_data))

	def update_style(self, style_img):

		style_img = style_img.transpose(2, 0, 1).reshape(1, 3, self.height, self.width)
		for style_rep, style_layer in zip(self.style_representation, style_layers):
			style_data = data_network[style_layer].eval({self.input: style_img})
			style_rep.set_value(style_data)

	def update_content(self, content_img):
	
		content_img = content_img.transpose(2, 0, 1).reshape(1, 3, self.height, self.width)
		for content_rep, content_layer in zip(self.content_representation, content_layers):
			content_data = data_network[content_layer].eval({self.input: content_img})
			content_rep.set_value(content_data)

def read_vgg19_net(path):
	
	try:
		values = pickle.load(open(path, 'rb'))['param values']
	except:
		values = pickle.load(open(path, 'rb'), encoding='latin1')['param values']
		
	bgr_mean= np.array([103.939, 116.779, 123.68], dtype=theano.config.floatX).reshape((1, 3, 1, 1))

	shared_vars = []
	for val in values:		
		data = np.array(val)
		shared_vars.append(theano.shared(data))	
		
	return shared_vars, bgr_mean
			
def make_vgg19(x, vgg_vars, vgg_mean):
	
	network = {}
	x = x - vgg_mean
	
	data_pos =0	
	for i, layer in enumerate(vgg_layers):
		
		if 'conv' in layer:
			w = vgg_vars[data_pos]
			data_pos = data_pos + 1
			b = vgg_vars[data_pos]
			data_pos = data_pos + 1
			x = theano.tensor.nnet.conv2d(x, w, border_mode='half')
			x = x + b.dimshuffle('x', 0, 'x', 'x')
			
		if 'relu' in layer:
			x = theano.tensor.nnet.relu(x)

		if 'pool' in layer:
			x = theano.tensor.signal.pool.pool_2d(x, (2,2), ignore_border=False, mode='average_exc_pad')
		
		network[layer] = x

	return network
	
def gram_matrix(x):
    
	x = x[0].flatten(2)
	x_t = x.dimshuffle(1, 0)
	g = theano.tensor.dot(x, x_t)

	return g
    
def style_loss(x, a):
    
    A = gram_matrix(a)
    G = gram_matrix(x)
    N = a.shape[1]*a.shape[2]*a.shape[3]
    loss = 1.0/(4.0 * N**2)*theano.tensor.sum((G - A)**2)
   
    return loss    
    
def content_loss(x, p):
    
    loss = 0.5 * theano.tensor.sum((x - p)**2)
    
    return loss

def optimizer_step(vggdata, target_network):

	target_style_representation = []
	for style_layer in style_layers:
		target_style_representation.append(target_network[style_layer])
		
	target_content_representation = []
	for content_layer in content_layers:
		target_content_representation.append(target_network[content_layer])

	losses = []
	for s, t in zip(vggdata.style_representation, target_style_representation):
		losses.append(beta*style_loss(s, t))
	for s, t in zip(vggdata.content_representation, target_content_representation):
		losses.append(alpha*content_loss(s, t))
		
	loss = sum(losses)

	updates = lasagne.updates.adam(loss, [target_image], learning_rate=learning_rate)
	train_step = theano.function([], [loss], updates=updates)

	return train_step

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

vgg_vars, vgg_mean = read_vgg19_net(vgg19_path)
input_placeholder = theano.tensor.tensor4(name='input')
data_network = make_vgg19(input_placeholder, vgg_vars, vgg_mean)

if choice == 'image':

	content_img = cv2.imread(content_img_path)
	content_img = cv2.resize(content_img, (0,0), fx=ratio, fy=ratio)
	height, width = content_img.shape[:2]

if choice == 'video':

	cap = cv2.VideoCapture(input_video_path)
	cap.set(cv2.CAP_PROP_POS_MSEC, video_start_sec*1000)
	ret, content_img = cap.read()
	content_img = cv2.resize(content_img, (0,0), fx=ratio, fy=ratio)
	
	if ret == False:
			print ('can not open video')
			sys.exit()
	height, width = content_img.shape[:2]
	
style_img = cv2.imread(style_img_path)
style_img = cv2.resize(style_img, (width, height))

vggdata = loss_vgg_data(input_placeholder, data_network, height, width)
vggdata.update_content(content_img)
vggdata.update_style(style_img)	

target_image = theano.shared(content_img.transpose(2, 0, 1).reshape(1, 3, height, width).astype(np.float32))
target_network = make_vgg19(target_image, vgg_vars, vgg_mean)
train_step = optimizer_step(vggdata, target_network)

if choice == 'image':

	for i in range(n_iters):
		data_loss = train_step()
		print ('train iter:', i, 'loss:', float(data_loss[0]))
		
	out = target_image.get_value()
	out = np.clip(out, 0, 255)
	out = out.transpose(0, 2, 3, 1).reshape(height, width, 3).astype(np.uint8)
	cv2.imwrite(target_img_path, out)

if choice == 'video':
	
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	videowriter = cv2.VideoWriter(target_video_path, fourcc, 30.0, (width,height))

	frame_iter = 0
	while(cap.isOpened()):
		
		ret, content_img = cap.read()
		content_img = cv2.resize(content_img, (0,0), fx=ratio, fy=ratio)
		vggdata.update_content(content_img)

		for i in range(n_iters):
			data_loss = train_step()
			print ('frame_iter:', frame_iter, 'train iter:', i, 'loss:', float(data_loss[0])) 
			
		out = target_image.get_value()
		out = np.clip(out, 0, 255)
		out = out.transpose(0, 2, 3, 1).reshape(height, width, 3).astype(np.uint8)
		videowriter.write(out)		
		frame_iter = frame_iter + 1
