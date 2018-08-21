# deploy script for Jupyter
tmux new -s jup-cr -d 'jupyter notebook --ip=0.0.0.0 --port=8010'
