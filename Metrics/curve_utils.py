import os
import numpy as np
import matplotlib.pyplot as plt
import statistics

def marshal_thresholded_dictionary(thresholded_dictionary):
  """

  DEPRECATED: not used

  Inverts the provided dictionary for use with precision recall plotting
  Inputs:
    - thresholded_dictionary: {threshold: {image: (pre, rec)}}
  Outputs:
    - marshaled_dictionary: {image: [(threshold, (pre, rec)))]}
  """
  ## marshal the dictionary for easier access
  marshaled_dictionary = {}
  for threshold in thresholded_dictionary:
    for image in thresholded_dictionary[threshold]:
      ## if the image is already instantiated, then append the current precision recall
      if image in marshaled_dictionary:
        marshaled_dictionary[image].append((threshold, thresholded_dictionary[threshold][image]))
      ## else instantiate the list for the precision recall
      else:
        marshaled_dictionary[image] = [(threshold, thresholded_dictionary[threshold][image])]
  return marshaled_dictionary

def subplot_image(marshaled_dictionary, detector, filepath):
  """

  DEPRECATED: flawed logic

  Plots the marshaled dictionary, plots the values, and saves the image to the specified path
  Inputs:
    - marshaled_dictionary: {image: [(threshold, (pre, rec)))]}
    - detector: character detector letter
    - filepath: String specifying the location of the saved image
  Outputs:
    None
  """

  ## for each image
  for image in marshaled_dictionary:
    ## capture the precision values
    precision_vals = [pair[0] for thresh, pair in marshaled_dictionary[image]]
    ## capture the recall values
    recall_vals = [pair[1] for thresh, pair in marshaled_dictionary[image]]
    
    ## calculate the average precision value
    avg_pre = statistics.mean(precision_vals)

    ## create the plot
    plt.figure();
    plt.plot(recall_vals, precision_vals, '-o');
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.title("Average Precision: " + str(avg_pre));
    save_path = os.path.join(filepath, "{letter}_detector_on_{image}.jpg".format(letter=detector, image=image))
    print(save_path)
    plt.savefig(save_path);

def subplot_curve(precision, recall, ap, detector, image, filepath):
  """
  Plots curve of precision and recall for detector. Saves plot to specified filepath.

  Inputs:
    - precision: List of precision values
    - recall: List of recall values
    - ap: Average precision value
    - detector: Name of detector for report
    - image: Name of image for report
    - filepath: Out file path of plot image
  """
  ## create the plot
  plt.figure();
  plt.plot(recall, precision, '-o');
  plt.xlabel('Recall');
  plt.ylabel('Precision');
  plt.title("Average Precision: " + str(ap));
  save_path = os.path.join(filepath, "{letter}_detector_on_{image}.jpg".format(letter=detector, image=image))
  print(save_path)
  plt.savefig(save_path);