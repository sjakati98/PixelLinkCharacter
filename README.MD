# Character Detection with PixelLink

Goal of this project is to use PixelLink to detect individual characters within our map recognition dataset. 

## To Do:
- [X] Clean and crop dataset to match PixelLink record format
    - [X] Add functionality to handle cropping for each letter
    - [X] Remember to add system arguments for directories to crop scripts
    - [X] Could combine the crop scripts
    - [ ] Need to add '###' label to all cropped annotations
- [ ] Create TFRecord format of cropped images and annotations
- [ ] Train on character annotations
- [ ] Output training performance, and testing IoU