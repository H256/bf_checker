# Binance Futures Overview
## A dashboard that shows all current positions and some stats

Requires Python 3.x to run.

To install: 
1. clone this repo
2. create and activate a virtual environment like mentioned [here](https://docs.python.org/3/library/venv.html).
3. install `requirements.txt` using `pip3 install -r requirements.txt` on linux or `pip install -r requirements.txt` on windows.
4. rename `.env.example` to `.env`
5. edit `.env` and fill in you Binance Futures API key and secret.
6. optionally set the port to something different than `8051` and edit the host to make it available outside the local machine.

To start, from within the activated venv run `python3 app.py` on linux or `python app.py` on windows.

Open a browser and navigate to `http://localhost:8051` (default). 
You should see the Dashboard appear and refresh every 10 seconds.

