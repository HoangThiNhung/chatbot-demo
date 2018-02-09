### Chatbot with Machine Learning

#### Training Model
```
python3 training.py
```
#### Predict
```
from predict import response, classify

>>> response('is your shop open today?')
We're open every day from 9am-9pm

>>> classify('is your shop open today?')
[('opentoday', 0.93861651)]
```
