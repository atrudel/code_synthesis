import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QFileDialog, \
	QMessageBox, QInputDialog, QFileSystemModel, QTreeView
from PyQt5.QtGui import QPainter, QPixmap, QColor, QIcon, QPolygon
from PyQt5.QtCore import Qt, QCoreApplication, QPoint, QDir
from .GUI.ui.ui_mainwindow import Ui_MainWindow
from .fileLoader import Pattern, pad_pattern
from .arena import Arena
import numpy as np
import os
import multiprocessing
#from numba import jit

class MainWindow(QMainWindow):
	def __init__(self, game):
		super(QMainWindow, self).__init__()
		if game is not None:
			self.g = game
			self.gamesize = self.g.size()
		else:
			self.gamesize = (512, 512)
			self.g = Arena(*self.gamesize)

		self.playersize = (64, 64)
		self.max_steps = -1
		self.scale = 1
		self.offset = (0, 0)

		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

		self.ui.canvas_label.setMinimumSize(*self.gamesize)

		path = os.path.dirname(os.path.realpath(__file__))

		def load_icon(button, filename):
			picon = QIcon(QPixmap(os.path.join(path, filename)))
			button.setIcon(picon)

		load_icon(self.ui.play_button, "GUI/icons/play.png")
		load_icon(self.ui.pause_button, "GUI/icons/pause.png")
		load_icon(self.ui.step_button, "GUI/icons/step.png")
		load_icon(self.ui.reset_button, "GUI/icons/reset.png")
		load_icon(self.ui.multistep_button, "GUI/icons/multi.png")

		def init_tree_view(tree):
			model = QFileSystemModel()
			model.setFilter(model.filter() & ~QDir.NoDotDot)
			model.setRootPath(QDir.currentPath())
			model.sort(0)
			tree.setModel(model);
			tree.setRootIndex(model.index(QDir.currentPath()));
			for i in range(model.columnCount() - 1, 0, -1):
				tree.hideColumn(i)
			return model

		self.filemodel1 = init_tree_view(self.ui.path1_tree)
		self.filemodel2 = init_tree_view(self.ui.path2_tree)

		self.ui.path1_tree.activated.connect(lambda i: self.tree_open(i, 1))
		self.ui.path2_tree.activated.connect(lambda i: self.tree_open(i, 2))

		self.ui.play_button.clicked.connect(lambda: self.run(self.max_steps))
		self.ui.pause_button.clicked.connect(self.pause)
		self.ui.step_button.clicked.connect(self.step)
		self.ui.reset_button.clicked.connect(self.reset)
		self.ui.multistep_button.clicked.connect(self.run_custom)

		self.ui.p1_open.clicked.connect(lambda: self.button_open(1))
		self.ui.p2_open.clicked.connect(lambda: self.button_open(2))

		self.ui.canvas_label.mousePressEvent = self.view_clicked

		self.ui.mistake.triggered.connect(self.not_a_feature)
		self.running = False
	
	def view_clicked(self, event):
		pos = event.localPos()
		pos = (pos.x(), pos.y())
		if event.button() == Qt.LeftButton:
			off = self.offset
			s = self.scale * 1.5
			self.offset =\
				(off[0] - pos[0] / self.scale + (self.gamesize[0] / (s)) / 2,\
				off[1] - pos[1] / self.scale + (self.gamesize[1] / (s)) / 2)
			self.scale = s

		elif event.button() == Qt.RightButton:
			off = self.offset
			s = self.scale / 1.5 if self.scale / 1.5 > 1 else 1
			if s == 1:
				self.offset = (0, 0)
			else:
				self.offset =\
				(off[0] - pos[0] / self.scale + (self.gamesize[0] / (s)) / 2,\
				off[1] - pos[1] / self.scale + (self.gamesize[1] / (s)) / 2)
			self.scale = s
		self.draw_arena()

	def button_open(self, n):
		self.running = False
		try:
			filename, _ = \
				QFileDialog.getOpenFileName(self, "Select player {}".format(n))
			if filename == '':
				return
			self.set_player(filename, n)
		except Exception as e:
			self.ui.statusbar.showMessage(str(e))

	def tree_open(self, index, n):
		self.running = False
		try:
			if n == 1:
				model = self.filemodel1
				tree = self.ui.path1_tree
			else:
				model = self.filemodel2
				tree = self.ui.path2_tree
			filename = model.filePath(index)
			if model.isDir(index):
				model.setRootPath(filename)
				model.sort(0)
				tree.setRootIndex(model.index(filename))
				return
			self.set_player(filename, n)
		except Exception as e:
			self.ui.statusbar.showMessage(str(e))

	def set_player(self, filename, n):
		self.running = False
		if filename is None or (n != 1 and n != 2):
			return
		try:
			p = Pattern(filename)
			player = pad_pattern(p.data, self.playersize)

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
			if left + right == 0:
				return
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

		pmap = QPixmap(*self.gamesize)
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
			painter.scale(self.scale, self.scale)
			painter.translate(*self.offset)
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

def _run(game, player1_file, player2_file):
	app = QApplication(sys.argv)
	window = MainWindow(game)
	window.show()
	window.set_player(player1_file, 1)
	window.set_player(player2_file, 2)
	window.draw_arena()

	app.lastWindowClosed.connect(window.pause)
	app.aboutToQuit.connect(app.deleteLater)
	app.exec_()



def start(game = None, p1 = None, p2 = None):
	if __name__ == "__main__":
		_run(game, p1, p2)
		return None
	else:
		ctx = multiprocessing.get_context('spawn')
		proc = ctx.Process(target=_run, args=(game, p1, p2))
		proc.start()
		return proc

if __name__ == "__main__":
	args = len(sys.argv)
	p = []
	if args > 1 and args < 4:
		p += sys.argv[1:]
	start(None, *p)
