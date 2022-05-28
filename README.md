# Garch-Double-Gaussian-Distribution
This project builds a Garch model with two sigmas (both follow Gaussian-distribution).

You can copy this folder into your Jupyter notebook. Then you can create a new ipynb file, and import functions as the following:

```py
from Main import runModel, runModelSingleGuassian
model = runModel("BTC") #Double Gaussian
runModelSingle("BTC", "t")

model.sigma1
model.sigma2
model.resids1
model.resids2
```

If you have no data in loca, run data scraper as the following:
```py
from DataSraper import get_crypto_data
get_crypto_data("BTC")
```

Then you can rerun the model.
