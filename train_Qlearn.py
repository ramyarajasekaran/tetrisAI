from random import randrange as rand
import pygame, sys
import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.

style.use("ggplot")  # setting our style!

SIZE = 10 # q size

EPISODES = 25000 # number of games to train on
REWARD = 10 # for completing one line
epsilon = 0.5 # for randomness
EPS_DECAY = 0.999 # decay per episode
SHOW_EVERY = 1000 # to visualize

start_q_table = None  # if we have a pickled Q table, we'll put the filename of it here.

LEARNING_RATE = 0.1
DISCOUNT = 0.5
outputAction=["LEFT","RIGHT","DOWN"]
# The configuration
config = {
	'cell_size':	20,
	'cols':		10,
	'rows':		16,
	'delay':	750,
	'maxfps':	120,
	'speedupPeriod':3000,
	'speedupFracNum':98,
	'speedupFracDenom':100,
}

colors = [
(0,   0,   0  ),
(255, 0,   0  ),
(0,   150, 0  ),
(0,   0,   255),
(255, 120, 0  ),
(255, 255, 0  ),
(180, 0,   255),
(0,   220, 220)
]

# Define the shapes of the single parts
tetris_shapes = [
	[[1, 1, 1],
	 [0, 1, 0]],
	
	[[0, 2, 2],
	 [2, 2, 0]],
	
	[[3, 3, 0],
	 [0, 3, 3]],
	
	[[4, 0, 0],
	 [4, 4, 4]],
	
	[[0, 0, 5],
	 [5, 5, 5]],
	
	[[6, 6, 6, 6]],
	
	[[7, 7],
	 [7, 7]]
]

def rotate_clockwise(shape):
	return [ [ shape[y][x]
			for y in 
range(len(shape)) ]
		for x in 
range(len(shape[0]) - 1, -1, -1) ]

def check_collision(board, shape, offset):
	off_x, off_y = offset
	for cy, row in enumerate(shape):
		for cx, cell in enumerate(row):
			try:
				if cell and board[ cy + off_y ][ cx + off_x ]:
					return True
			except IndexError:
				return True
	return False

def remove_row(board, row):
	del board[row]
	return [[0 for i in 
range(config['cols'])]] + board
	
def join_matrixes(mat1, mat2, mat2_off):
	off_x, off_y = mat2_off
	for cy, row in enumerate(mat2):
		for cx, val in enumerate(row):
			mat1[cy+off_y-1	][cx+off_x] += val
	return mat1

def new_board():
	board = [ [ 0 for x in 
range(config['cols']) ]
			for y in 
range(config['rows']) ]
	board += [[ 1 for x in 
range(config['cols'])]]
	return board

class TetrisApp(object):
	def __init__(self):
		pygame.init()
		pygame.key.set_repeat(250,25)
		self.width = config['cell_size']*config['cols']
		self.height = config['cell_size']*config['rows']
		
		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
		                                             # mouse movement
		                                             # events, so we
		                                             # block them.
		self.init_game()
	
	def new_stone(self):
		self.stone = tetris_shapes[rand(len(tetris_shapes))]
		self.stone_x = int(config['cols'] / 2 - len(self.stone[0])/2)
		self.stone_y = 0
		
		if check_collision(self.board,
		                   self.stone,
		                   (self.stone_x, self.stone_y)):
			self.gameover = True
	
	def init_game(self):
		self.board = new_board()
		self.new_stone()
		self.score=0
	
	def center_msg(self, msg):
		for i, line in enumerate(msg.splitlines()):
			msg_image =  pygame.font.Font(
				pygame.font.get_default_font(), 12).render(
					line, False, (255,255,255), (0,0,0))
		
			msgim_center_x, msgim_center_y = msg_image.get_size()
			msgim_center_x //= 2
			msgim_center_y //= 2
		
			self.screen.blit(msg_image, (
			  self.width // 2-msgim_center_x,
			  self.height // 2-msgim_center_y+i*22))
	def display_score(self):
		for i, line in enumerate(("Score: "+str(self.score)).split(".")[0].splitlines()):
			msg_image =  pygame.font.Font(
				pygame.font.get_default_font(), 15).render(
					line, False, (255,255,255), (0,0,0))
		
			msgim_center_x, msgim_center_y = msg_image.get_size()
			msgim_center_x //= 2
			msgim_center_y //= 2
		
			self.screen.blit(msg_image, (0,i*22))
	def draw_matrix(self, matrix, offset):
		off_x, off_y  = offset
		for y, row in enumerate(matrix):
			for x, val in enumerate(row):
				if val:
					pygame.draw.rect(
						self.screen,
						colors[val],
						pygame.Rect(
							(off_x+x) *
							  config['cell_size'],
							(off_y+y) *
							  config['cell_size'], 
							config['cell_size'],
							config['cell_size']),0)
	
	def move(self, delta_x):
		if not self.gameover and not self.paused:
			new_x = self.stone_x + delta_x
			if new_x < 0:
				new_x = 0
			if new_x > config['cols'] - len(self.stone[0]):
				new_x = config['cols'] - len(self.stone[0])
			if not check_collision(self.board,
			                       self.stone,
			                       (new_x, self.stone_y)):
				self.stone_x = new_x
	def quit(self):
		self.center_msg("Exiting...")
		pygame.display.update()
		sys.exit()
	
	def drop(self):
		if not self.gameover and not self.paused:
			self.stone_y += 1
			if check_collision(self.board,
			                   self.stone,
			                   (self.stone_x, self.stone_y)):
				self.board = join_matrixes(
				  self.board,
				  self.stone,
				  (self.stone_x, self.stone_y))
				self.new_stone()
				newScore=0
				while True:
					for i, row in enumerate(self.board[:-1]):
						if 0 not in row:
							self.board = remove_row(
							  self.board, i)
							newScore+=1
							break
					else:
						break
				self.score+=(1+newScore**2)//1.5
	
	def rotate_stone(self):
		if not self.gameover and not self.paused:
			new_stone = rotate_clockwise(self.stone)
			if not check_collision(self.board,
			                       new_stone,
			                       (self.stone_x, self.stone_y)):
				self.stone = new_stone
	
	def toggle_pause(self):
		self.paused = not self.paused
	
	def start_game(self):
		if self.gameover:
			self.init_game()
			self.gameover = False
	
	def run(self):
		key_actions = {
			'ESCAPE':	self.quit,
			'LEFT':		lambda:self.move(-1),
			'RIGHT':	lambda:self.move(+1),
			'DOWN':		self.drop,
			'UP':		self.rotate_stone,
			'p':		self.toggle_pause,
			'SPACE':	self.start_game
		}
		
		self.gameover = False
		self.paused = False
		
		pygame.time.set_timer(pygame.USEREVENT+1, config['delay'])
		dont_burn_my_cpu = pygame.time.Clock()
		i=0
		while 1:
			# print(self.score)
			i+=1
			if i%config['speedupPeriod']==0:
				config['delay']=(config['delay']*config['speedupFracNum'])//config['speedupFracDenom']
				pygame.time.set_timer(pygame.USEREVENT+1, config['delay'])
			self.screen.fill((0,0,0))
			self.display_score()
			if self.gameover:
				self.center_msg("""Game Over!
Press space to continue""")
			else:
				if self.paused:
					self.center_msg("Paused")
				else:
					self.draw_matrix(self.board, (0,0))
					self.draw_matrix(self.stone,
					                 (self.stone_x,
					                  self.stone_y))
			pygame.display.update()
			
			for event in pygame.event.get():
				if event.type == pygame.USEREVENT+1:
					self.drop()
				elif event.type == pygame.QUIT:
					self.quit()
				elif event.type == pygame.KEYDOWN:
					# for key in key_actions: # reads keys from keyboard
						# if event.key == eval("pygame.K_"+key):
						#	key_actions[key]()
					## WE WANT SW VERSION

						# use  key_actions[key]() to do different actions
						obs = (,) # ???? should be in terms of keys or other heuristics?

						#INPUTS
						boardState=self.board
						tileState=(self.stone,(self.stone_x, self.stone_y))
						
						if np.random.random() > epsilon:
						# GET THE ACTION
							action = np.argmax(q_table[obs])
						else:
							action = np.random.randint(0, 3)

						# OUTPUT
						out = action

						# run action 
						key_actions[outputAction[out]]()

						# rewarding is handled by tetris code 
						# needs to initialize reward var

						# updating q table
						new_obs = (,) #?? idk new observation
						max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
						current_q = q_table[obs][action]  # current Q for our chosen action
						
						if reward == 10: #completed line??
							new_q = reward;
						else:
							new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

						episode_reward += reward

			return episode_reward

			dont_burn_my_cpu.tick(config['maxfps'])

if __name__ == '__main__':
	main()

def main():
	App = TetrisApp()
	# predicting outputAction
	# outputAction=["LEFT","RIGHT","DOWN"]
	# predicting  out = 0 || 1 ||2


	## i know this looks really bad but we can fix it later plis
	if start_q_table is None:
		# initialize the q-table #
		q_table = {}
		for i in range(-SIZE+1, SIZE):
			for ii in range(-SIZE+1, SIZE):
				for iii in range(-SIZE+1, SIZE):
 					for iiii in range(-SIZE+1, SIZE):
						q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]
	else:
		with open(start_q_table, "rb") as f:
			q_table = pickle.load(f)

	episode_rewards = []
	for episode in range(EPISODES):
		
		if episode % SHOW_EVERY == 0:
			print(f"on #{episode}, epsilon is {epsilon}")
			print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
			show = True
		else:
			show = False

		episode_reward = App.run() # runs one iteration of the game and updates q table
		
		episode_rewards.append(episode_reward)
		epsilon *= EPS_DECAY

	print ('Training done!')



