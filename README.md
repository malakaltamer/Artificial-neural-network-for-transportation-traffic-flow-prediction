# Transportation's traffic flow prediction using neural network algorithm

This is a neural network algorithm that predict traffic flow using a .csv file (containing Time Period Ending, Time Interval, Avg mph and Total Volume) to train. The algorithm can also predict traffic flow based on user inputted features.

## Installation

Download the files from here on GitHub.

for the executable version, you can use it straight away if you have Microsoft Visual C++ which you can get from (https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

for the source code please follow the following installation requirements in order:

1) Open terminal in the source code folder.
2) Paste the following string in the terminal.
```bash
pip install -r requirements.txt
```
3) Close the terminal and enjoy the program.

The code was developed on python 3.12 and should work best on it.

## How to use
Open the app then select the dataset to unlock all the input fields.

From here you can either train the model or use the already trained model.

To train the model click on train and wait (approximately 4 hours to finish), then two buttons will appear instead of training which are Results and Graph. the results option will open a window with R², RMSE and MAE values that the model achieved based on the training data. The graph button will show the actual vs predicted traffic flow.

To predict inputted values, please input your features numbers in their perspective box under time of day, interval and speed, then click on predict. A prediction should be appear in the model prediction box.

## Known issues and fixes
On some computers the text might be too large or too small and/or when the fullscreen feature is used the HUD becomes a box on the top left of the screen.

To fix these issues, right click on the program (MAIN.exe) then 'Properties', followed by 'Compatibility', then 'Change high DPI settings', check the 'Override high DPI scaling behavior.' box, and choose 'System (Enhanced)' from 'the scaling performed by:' menu.

or you can set the screen scale to 100% and display resolution to 1920x1080 from the display settings.
