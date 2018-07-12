import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QFileDialog, \
	QMessageBox, QInputDialog
from PyQt5.QtGui import QPainter, QPixmap, QColor, QIcon, QPolygon
from PyQt5.QtCore import Qt, QCoreApplication, QPoint
from GUI.ui.ui_mainwindow import Ui_MainWindow
from game_of_life import rle
from game_of_life.game import GameContainer
import numpy as np
import time
import pprofile
#from numba import jit

class MainWindow(QMainWindow):
	def __init__(self, game):
		super(QMainWindow, self).__init__()
		if game is not None:
			self.g = game
			self.gamesize = self.g.size()
		else:
			self.gamesize = (512, 512)
			self.g = GameContainer(self.gamesize[0], self.gamesize[1])

		self.playersize = (64, 64)
		self.max_steps = -1

		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

		self.ui.canvas_label.setMinimumSize(self.gamesize[0], self.gamesize[1])

		def load_icon(button, filename):
			picon = QIcon(QPixmap(filename))
			button.setIcon(picon)
		
		load_icon(self.ui.play_button, "GUI/icons/play.png")
		load_icon(self.ui.pause_button, "GUI/icons/pause.png")
		load_icon(self.ui.step_button, "GUI/icons/step.png")
		load_icon(self.ui.reset_button, "GUI/icons/reset.png")
		load_icon(self.ui.multistep_button, "GUI/icons/multi.png")

		self.ui.play_button.clicked.connect(lambda: self.run(self.max_steps))
		self.ui.pause_button.clicked.connect(self.pause)

		self.ui.step_button.clicked.connect(self.step)
		self.ui.reset_button.clicked.connect(self.reset)
		self.ui.multistep_button.clicked.connect(self.run_custom)

		self.ui.p1_open.clicked.connect(lambda: self.set_player(1))
		self.ui.p2_open.clicked.connect(lambda: self.set_player(2))
		self.ui.mistake.triggered.connect(self.not_a_feature)
		self.running = False
		
	def set_player(self, n):
		self.running = False
		try:
			filename, _ = \
				QFileDialog.getOpenFileName(self, "Select player {}".format(n))
			if filename == '':
				return
			p = rle.Pattern(filename)
			player = rle.pad_pattern(p.data, self.playersize)

			if n == 1:
				self.g.add_players(player)
				self.ui.p1_path.setText(filename)
			elif n == 2:
				self.g.add_players(None, player)
				self.ui.p2_path.setText(filename)
			self.draw_arena()
			self.ui.statusbar.showMessage(\
				"{} loaded!".format(filename[filename.rfind("/") + 1:]))
		except Exception as e:
			self.ui.statusbar.showMessage(str(e))

	def resizeEvent(self, event):
		pass

	#@jit
	def draw_arena(self):
		p1_color = QColor("blue")
		p2_color = QColor("yellow")

		def draw_bar(left, right):
			leftcol = p1_color if left > right else p1_color.darker()
			rightcol = p2_color if left < right else p2_color.darker()
			barsize = self.ui.percent_bar.size()
			pmap = QPixmap(barsize.width(), barsize.height())
			pmap.fill(QColor("white"))
			painter = QPainter(pmap)
			xdiv = barsize.width() * left / (left + right)
			painter.fillRect(0, 0, xdiv, barsize.height(), leftcol)
			painter.fillRect(xdiv, 0, \
				barsize.width() - xdiv, barsize.height(), rightcol)
			painter.end()
			self.ui.percent_bar.setPixmap(pmap)

		pmap = QPixmap(self.gamesize[0], self.gamesize[1])
		pmap.fill(QColor("gray"))

		if self.g.engine is not None:
			p1 = QPolygon()
			p2 = QPolygon()

			grid = self.g.grid()
			for x, y in self.g.engine._points:
				pt = QPoint(x, y)
				if (grid[y][x] & 1):
					p1.append(pt)
				else:
					p2.append(pt)

			draw_bar(p1.size(), p2.size())

			painter = QPainter(pmap)
			painter.setPen(p1_color)
			painter.drawPoints(p1)
			painter.setPen(p2_color)
			painter.drawPoints(p2)

			painter.end()

		self.ui.canvas_label.setPixmap(pmap)

	def run(self, steps = -1):
		if self.running:
			return
		self.running = True
		while steps != 0 and self.running:
			try:
				self.g.run_steps(1)
			except Exception as e:
				self.ui.statusbar.showMessage(str(e))
				return
			self.draw_arena()
			self.ui.statusbar.showMessage("step: {}".format(self.g.steps))
			QCoreApplication.processEvents()
			steps -= 1
		self.running = False

	def run_custom(self):
		try:
			steps, valid = QInputDialog.getInt(self, "Run N steps", \
				"Select the number of steps to run", 0, 0)
			if valid:
				self.run(steps)
		except Exception as e:
			self.ui.statusbar.showMessage(str(e))

	def pause(self):
		self.running = False

	def step(self):
		self.pause()
		self.run(1)

	def reset(self):
		self.pause()
		self.g.setup()
		self.draw_arena()
		self.ui.statusbar.showMessage("Game reset!")

	def not_a_feature(self):
		d = QMessageBox(self)
		d.setWindowTitle("Wrong. Wrong. Wrong.")
		d.setText("I can't believe you've done this...")
		d.exec_()

def start(game = None):
	app = QApplication(sys.argv)
	window = MainWindow(game)

	window.show()
	window.draw_arena()

	app.lastWindowClosed.connect(window.pause)
	app.aboutToQuit.connect(app.deleteLater)
	app.exec_()

if __name__ == "__main__":
	start()
