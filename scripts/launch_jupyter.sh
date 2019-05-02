# deploy script for Jupyter
source venv/bin/activate
tmux kill-session -t cr_jup
tmux new -s cr_jup -d 'jupyter notebook --ip=0.0.0.0 --port=8010 --NotebookApp.allow_remote_access=True 2>&1 | tee jupyter.log'
