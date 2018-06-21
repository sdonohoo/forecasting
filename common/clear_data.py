import shutil
import os
shutil.rmtree('data')

os.makedirs('data')
os.makedirs(os.path.join('data', 'raw'))
