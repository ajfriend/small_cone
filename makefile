.PHONY: env lab clean purge tunnel test

env:
	python3 -m venv env
	env/bin/pip install --upgrade pip
	env/bin/pip install -r requirements.txt
	bash -l -c 'nvm install 12; nvm exec 12 env/bin/jupyter labextension install @jupyter-widgets/jupyterlab-manager @deck.gl/jupyter-widget nbdime-jupyterlab jupyterlab-plotly;'

lab:
	bash -l -c 'nvm install 12; nvm exec 12 env/bin/jupyter lab;'

clean:
	find . -type f -name '*.pyc' | xargs rm -r
	find . -type d -name '*.ipynb_checkpoints' | xargs rm -r
	find . -type d -name '*.egg-info' | xargs rm -r
	find . -type d -name '__pycache__' | xargs rm -r
	find . -type d -name '.pytest_cache' | xargs rm -r

purge: clean
	-@rm -rf env
