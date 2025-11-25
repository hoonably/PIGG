import kagglehub
import shutil
import os

# Download
path = kagglehub.dataset_download("qinglongyang/fingertip-20k")
print("Downloaded to:", path)

# Move to current directory
dst = "./fingertip-20k"
shutil.move(path, dst)
print("Moved to:", dst)
