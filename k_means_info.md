the classes are divided in the following form

Class   delay
C1      <398ps
C2      398 - 482ps
C3      483 - 585ps
C4      586-708ps
C5      >708ps


clusters into 4096 clusters

There are 2 files
1. k_means_Cell_delay.py
2. k_means_net_delay.py


File names are self explanatory

The file : k_means_Cell_delay.py is importing file paths.py
paths.py is a file, that keeps the file paths, so that, if any file error, python can point to that import statement

Hence, any changes in file path, is just to be done in paths.py, and the program handles the rest


results file is not uploaded to git
the f1 scores is 81% for test (Net delay)
the f1 score is 88% for test (Cell delay)

