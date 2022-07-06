docker run --mount type=bind,source=/radraid,target=/radraid --mount type=bind,source=/cvib2,target=/cvib2 --mount type=bind,source=/scratch,target=/scratch --mount type=bind,source=/scratch2,target=/scratch2 --mount type=bind,source=/dingo_data,target=/dingo_data --mount type=bind,source=/cvib,target=/cvib -it registry.rip.ucla.edu/deep_med /bin/bash


# need to pip install
#   > ipython
#   > nibabel
#   > nilearn

pip install ipython
pip install nibabel
pip install nilearn