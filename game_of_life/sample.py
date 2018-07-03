import curses
import sys
from time import sleep

import gol
import rle

def draw_border(stdscr, width, height):
	stdscr.move(0, 0)
	stdscr.addstr(u"╔")
	stdscr.addstr(u"═" * (width - 2))
	stdscr.addstr(u"╗")
	for y in range(1, height):
		stdscr.move(y, 0)
		stdscr.addstr(u"║")
		stdscr.move(y, width - 1)
		stdscr.addstr(u"║")
	stdscr.move(height - 1, 0)
	stdscr.addstr(u"╚")
	stdscr.addstr(u"═" * (width - 2))
	stdscr.addstr(u"╝")


def update_grid(stdscr, game):
	size = game.size()
	cells = game.grid()
	for y in range(size[1]):
		stdscr.move(y + 1, 1)
		for x in range(size[0]):
			c = cells[y * size[0] + x]
			if c == 1:
				stdscr.addch(ord("1"))
			elif c == 2:
				stdscr.addch(ord("2"))
			else:
				stdscr.addch(ord("."))
	stdscr.refresh()

def start(stdscr):
	#curses.curs_set(0)
	stdscr.clear()

	size = stdscr.getmaxyx()
	game = []
	if (len(sys.argv)) == 2:
		patt = rle.Pattern(sys.argv[1])
		game = gol.GameOfLife(patt.width, patt.height)
		game.set_grid(patt.data)
		size = [patt.height + 3, patt.width + 2]
	else:
		game = gol.GameOfLife(size[1] - 2, size[0] - 3)

	guide = "s: set mode, r: run, q: quit $ "

	draw_border(stdscr, size[1], size[0] - 1)
	update_grid(stdscr, game)

	def prompt(msg):
		stdscr.move(size[0] - 1, 0)
		stdscr.addstr(msg)
		stdscr.clrtoeol()
		stdscr.refresh()
		return str(stdscr.getstr(size[0] - 1, len(msg)), "utf-8")

	def setmode():
		curses.noecho()
		stdscr.move(1, 1)
		while True:
			cmd = stdscr.getch()
			pos = stdscr.getyx()

			if cmd == ord('q'):
				break
			elif cmd == curses.KEY_RIGHT:
				if pos[1] < size[1] - 2:
					stdscr.move(pos[0], pos[1] + 1)
			elif cmd == curses.KEY_LEFT:
				if pos[1] > 1:
					stdscr.move(pos[0], pos[1] - 1)
			elif cmd == curses.KEY_UP:
				if pos[0] > 1:
					stdscr.move(pos[0] - 1, pos[1])
			elif cmd == curses.KEY_DOWN:
				if pos[0] < size[0] - 3:
					stdscr.move(pos[0] + 1, pos[1])
			elif cmd == ord('0'):
				stdscr.addch(ord('.'))
				stdscr.move(pos[0], pos[1])
				game.set_point(pos[1] - 1, pos[0] - 1, 0)
			elif cmd == ord('1'):
				stdscr.addch(cmd)
				stdscr.move(pos[0], pos[1])
				game.set_point(pos[1] - 1, pos[0] - 1, 1)
			elif cmd == ord('2'):
				stdscr.addch(cmd)
				stdscr.move(pos[0], pos[1])
				game.set_point(pos[1] - 1, pos[0] - 1, 2)

	while True:
		curses.echo()
		cmd = prompt(guide)
		stdscr.refresh()
		
		if cmd == "q":
			break
		elif cmd == "r":
			try:
				while True:
					steps = int(prompt("number of steps: "))
					for i in range(int(steps)):
						game.run_steps(1)
						update_grid(stdscr, game)
						stdscr.move(size[0] - 1, 0)
						stdscr.clrtoeol()
						stdscr.move(size[0] - 1, 0)
						stdscr.addstr("{}/{}".format(i, steps))
						sleep(0.1)
			except:
				pass

		elif cmd == "s":
			setmode()

if __name__ == '__main__':
	curses.wrapper(start)
