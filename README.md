# gpbandits_for_algotrading

## Create virtual env & install dependencies
```
conda create -n exec python=3.7
conda install numpy pandas matplotlib seaborn jupyterlab
conda install botorch -c pytorch -c conda-forge
conda install tensorflow=1.15.0
pip install gpytorch==1.8.1
pip install tqdm
```

## Run
Use the following code block to run a single trial:
```
python scripts/sim_synthetic.py
```