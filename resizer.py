#!/usr/bin/python
__author__ = 'Aaron Woodhouse'

"""
Resize images for prediciton models.
"""

# Imports #
import os
import sys
import threading
from PIL import Image

# Methods #
def thread_task(path, dirs, lock):
    """Resize images (multithreading).

    Parameters
    ----------
    path : str
        Image folder path.
    dirs : str
        File names.
    lock
        Multithreading lock.

    """
    lock.acquire()
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((36,36), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)
    lock.release()

def resize(path: str):
    """Resize images in folder.

    Parameters
    ----------
    path : str
        Path to image folder.

    """
    
    lock = threading.Lock()
    
    path = path + '/'
    dirs = os.listdir( path )
    num_dirs = (len(dirs) // 4)
    
    # split dirs
    d1 = dirs[:num_dirs]
    d2 = dirs[num_dirs:num_dirs*2]
    d3 = dirs[num_dirs*2:num_dirs*3]
    d4 = dirs[num_dirs*3:]
    
    # creating threads
    t1 = threading.Thread(target=thread_task, args=(path, d1, lock))
    t2 = threading.Thread(target=thread_task, args=(path, d2, lock))
    t3 = threading.Thread(target=thread_task, args=(path, d3, lock))
    t4 = threading.Thread(target=thread_task, args=(path, d4, lock))
    
    threads = [t1,t2,t3,t4]
  
    # start threads
    for thread in threads:
        thread.start()
  
    # wait until threads finish their job
    for thread in threads:
        thread.join()