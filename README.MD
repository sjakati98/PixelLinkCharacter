# Character Detection with PixelLink

Goal of this project is to use PixelLink to detect individual characters within our map recognition dataset. 

## To Do:
- [X] ~~Clean and crop dataset to match PixelLink record format~~
    - [X] ~~Add functionality to handle cropping for each letter~~
    - [X] ~~Remember to add system arguments for directories to crop scripts~~
    - [X] ~~Could combine the crop scripts~~
- [X] ~~Create TFRecord format of cropped images and annotations~~
    - [X] ~~Need to make sure TFRecords are created for all characters~~
- [X] ~~Scripts to rotate cropped images and annotations accordingly~~
    - [X] ~~Include the rotated crops into the test-time routine (the images which are rotated such that a proportion of the characters are now horizontal)~~
- [ ] Train on character annotations
    - [ ] Scripts to utilize all the character annotations
    - [ ] Train on all characters and organize models
- [ ] Plot loss graphs
- [X] Get testing output points
    - [ ] Plot large image out of cropped
- [ ] Train all the other characters
- [ ] Output training performance, and testing IoU