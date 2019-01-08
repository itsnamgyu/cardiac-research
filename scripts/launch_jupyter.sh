# deploy script for Jupyter
source venv/bin/activate
tmux new -s jup -d 'jupyter notebook --ip=0.0.0.0 --port=8010 2>&1 | tee jupyter.log'
