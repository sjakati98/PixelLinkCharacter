from glob import glob
import os
import sys
import shutil

def move_matricies(source, destination):
    """
        Inputs:
            - source: Where the npy files live currently
            - destination: Where the npy files need to be moved
        Outputs:
            None
    """
    for filename in glob(os.path.join(source, "*.npy")):
        shutil.move(os.path.join(source, filename), destination)


if __name__ == "__main__":
    ## source
    rotation_images_dir = sys.argv[1]
    ## destination
    rotation_matricies_dir = sys.argv[2]
    ## complete move
    move_matricies(rotation_images_dir, rotation_matricies_dir)
    print("Done Moving!")