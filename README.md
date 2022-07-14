# Suspicious-Activity-Detection

In today's world we can see the rate of doing crimes is increasing on a daily basis. To overcome this surveillance cameras have been installed but a surveillance cameras needs a person to have a constant watch to activities and to do this is a very tiring job. We, Tranqulizer, propose a model which helps us to detect the unusual activities and rules out the need of a person to have a constant watch and report the unusual activities.

Dataset link: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset

The order to follow for preprocessing is:

The folder named preprocessing contains three files for performing the task of preprocessing

(i) first run the file preprocess1.py

(ii) then run the file preprocess2.py

Before running the preprocess3.py make a copy of videos you get inside the folder after running the preprocess2.py and name one folder as "DCSASS_Dataset_vid" and let the folder remain as it is.

(iii) atlast run the file preprocess3.py

Training:

For Training purpose we have created a ipynb file named Tranquilizer.ipynb. Along with ipynb file we have used 2 py files named essential.py and slowfast.py which are enclosed inside the folder named training.

References:

SlowFast: https://github.com/facebookresearch/SlowFast

Keras-SlowFast: https://github.com/xuzheyuan624/slowfast-keras
