#include <iostream>
#include <stdio.h>

// OpenCV Includes
#include <object_tracker.h>
#include <cv.h>
#include <highgui.h>


int main(int argc, char** argv)
{
  // Setup the parameters to use OnlineBoosting as the underlying tracking algorithm
  cv::ObjectTrackerParams params;
  params.algorithm_ = cv::ObjectTrackerParams::CV_ONLINEBOOSTING;
  //params.algorithm_ = cv::ObjectTrackerParams::CV_SEMIONLINEBOOSTING;
  params.num_classifiers_ = 100;
  params.overlap_ = 0.99f;
  params.search_factor_ = 2;

  // Instantiate an object tracker
  cv::ObjectTracker tracker(params);

  // Read in a sequence of images from disk as the video source
  const char* directory = "data/David";
  const int start = 1;
  const int stop = 462;
  const int delta = 1;
  const char* prefix = "img";
  const char* suffix = "png";
  char filename[1024];

  // Some book-keeping
  bool is_tracker_initialized = false;
  CvRect init_bb = cvRect(122,58,75,97);  // the initial tracking bounding box
  CvRect theTrack;
  bool tracker_failed = false;

  // Read in images one-by-one and track them
  cvNamedWindow("Tracker Display");
  for (int frame = start; frame <= stop; frame += delta)
  {
    sprintf(filename, "%s/%s%05d.%s", directory, prefix, frame, suffix);
    IplImage* image = cvLoadImage(filename);
    if (image == NULL)
    {
      std::cerr << "Error loading image file: " << filename << "!\n" << std::endl;
      break;
    }

    // Initialize/Update the tracker
    if (!is_tracker_initialized)
    {
      // Initialize the tracker
      if (!tracker.initialize(image, init_bb))
      {
        // If it didn't work for some reason, exit now
        std::cerr << "\n\nCould not initialize the tracker!  Exiting early...\n" << std::endl;
        break;
      }

      // Store the track for display
      theTrack = init_bb;
      tracker_failed = false;

      // Now it's initialized
      is_tracker_initialized = true;
      std::cout << std::endl;
      continue;
    }
    else
    {
      // Update the tracker
      if (!tracker.update(image, &theTrack))
      {
        std::cerr << "\rCould not update tracker (" << frame << ")";
        tracker_failed = true;
      }
      else
      {
        tracker_failed = false;
      }
    }

    // Display the tracking box
    CvScalar box_color;
    if (tracker_failed)
    {
      box_color = CV_RGB(255,0,0);
    }
    else
    {
      box_color = CV_RGB(255,255,0);
    }
    cvRectangle(image, cvPoint(theTrack.x,theTrack.y), 
      cvPoint(theTrack.x+theTrack.width-1,theTrack.y+theTrack.height-1), box_color, 2);

    // Display the new image
    cvShowImage("Tracker Display", image);

    // Release the image memory
    cvReleaseImage(&image);

    // Check if the user wants to exit early
    int key = cvWaitKey(1);
    if (key == 'q' || key == 'Q')
    {
      break;
    }
  }
  cvDestroyWindow("Tracker Display");

  // Exit application
  std::cout << std::endl;
  return 0;
}

