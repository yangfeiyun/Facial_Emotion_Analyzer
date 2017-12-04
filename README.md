# Facial_Emotion_Analyzer

1. Set up your tensorflow under a virtual environment (preferred, easier to manage, Anaconda is recommended).

2. Run "python fer2013DataGen.py" to split the fer2013.csv.

3. Run "FE.ipynb" under jupyter notebook to train the CNN mode.

4. Run "FE_test.ipynb" under jupyter notebook for data visualization.

5. Run "python memecam.py haarcascade_frontalface_default.xml" to activate the real-time facial emotion analyzer. The emotions will be saved to emotion.txt.
