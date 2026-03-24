PYTHON=python

extract:
	$(PYTHON) -m src.main --stage extract

clean:
	$(PYTHON) -m src.main --stage clean

labels:
	$(PYTHON) -m src.main --stage labels

features:
	$(PYTHON) -m src.main --stage features

train:
	$(PYTHON) -m src.main --stage train

score:
	$(PYTHON) -m src.main --stage score

monitor:
	$(PYTHON) -m src.main --stage monitor

full:
	$(PYTHON) -m src.main --stage full

test:
	pytest -q
