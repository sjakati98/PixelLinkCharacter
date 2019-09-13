# Character Detection with PixelLink

[PixelLink](https://github.com/ZJULearning/pixel_link) is word detection model. Here, we detail the performance of PixelLink after training it specifically on our character annotations. 

## Annotations
The character annotations themselves are approximate annotations, calculated using the overall word annotations found [here](https://people.cs.umass.edu/~ray/maps_project.html). 

*insert example images of ground truth character annotations*

## Results
To train our PixelLink detectors, the character annotations were segmented by letter. Thus, we have separate datasets for 'A', 'a', 'B', 'b', and so on. We also have datasets for digits 0 through 9. These segmented datasets were then split into a training and testing split, 80% and 20% respectively, to be used for the detection procedure. Models were then trained for `1000000` steps with a batch size of `2`, giving us an epoch number of `95`. 

