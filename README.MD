# Character Detection with PixelLink

Goal of this project is to use PixelLink to detect individual characters within our map recognition dataset. 

## To Do:
- [X] ~~Clean and crop dataset to match PixelLink record format~~
    - [X] ~~Add functionality to handle cropping for each letter~~
    - [X] ~~Remember to add system arguments for directories to crop scripts~~
    - [X] ~~Could combine the crop scripts~~
- [X] Create TFRecord format of cropped images and annotations
    - [ ] Need to make sure TFRecords are created for all characters
- [X] Train on character annotations
    - [ ] Train on all characters and organize models
- [ ] Plot loss graph
- [X] Get testing output points
    - [ ] Plot large image out of cropped
- [ ] Train all the other characters
- [ ] Train on rotated images 
- [ ] Output training performance, and testing IoU