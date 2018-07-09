UI=ui/ui_mainwindow.ui

RES=$(UI:.ui=.py)

all: $(RES)

ui/%.py: ui/%.ui
	(python3 -m PyQt5.uic.pyuic $< -o $@ -x)
