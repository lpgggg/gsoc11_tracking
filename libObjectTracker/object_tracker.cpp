/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "object_tracker.h"
#include <iostream>
#include <cstdlib>

namespace cv
{

  //---------------------------------------------------------------------------
  ObjectTrackerParams::ObjectTrackerParams()
  {
    // By default, use online boosting
    algorithm_ = CV_ONLINEBOOSTING;
  }

  //---------------------------------------------------------------------------
  ObjectTrackerParams::ObjectTrackerParams(const int algorithm)
  {
    // Make sure a valid algorithm flag is used before storing it
    if ( (algorithm != CV_ONLINEBOOSTING) && (algorithm != CV_ONLINEMIL) && (algorithm != CV_LINEMOD) )
    {
      // Use CV_ERROR?
      std::cerr << "ObjectTrackerParams::ObjectTrackerParams(...) -- ERROR!  Invalid algorithm choice.\n";
      exit(-1);
    }
    // Store it
    algorithm_ = algorithm;
  }

  //
  //
  //

  //---------------------------------------------------------------------------
  TrackingAlgorithm::TrackingAlgorithm()
  {
  }

  //---------------------------------------------------------------------------
  TrackingAlgorithm::~TrackingAlgorithm()
  {
  }

  //
  //
  //

  //---------------------------------------------------------------------------
  OnlineBoostingAlgorithm::OnlineBoostingAlgorithm() 
    : TrackingAlgorithm()
  {
  }

  //---------------------------------------------------------------------------
  OnlineBoostingAlgorithm::~OnlineBoostingAlgorithm()
  {
  }

  //---------------------------------------------------------------------------
  bool OnlineBoostingAlgorithm::initialize(const IplImage* image, const ObjectTrackerParams& params, 
    const CvRect& init_bounding_box)
  {
    // Return success
    return true;
  }

  //---------------------------------------------------------------------------
  bool OnlineBoostingAlgorithm::update(const IplImage* image, const ObjectTrackerParams& params, 
      CvRect* track_box, IplImage* likelihood)
  {
    // Return success
    return true;
  }

  //
  //
  //

  //---------------------------------------------------------------------------
  OnlineMILAlgorithm::OnlineMILAlgorithm() 
    : TrackingAlgorithm()
  {
  }

  //---------------------------------------------------------------------------
  OnlineMILAlgorithm::~OnlineMILAlgorithm()
  {
  }

  //---------------------------------------------------------------------------
  bool OnlineMILAlgorithm::initialize(const IplImage* image, const ObjectTrackerParams& params, 
    const CvRect& init_bounding_box)
  {
    // Return success
    return true;
  }

  //---------------------------------------------------------------------------
  bool OnlineMILAlgorithm::update(const IplImage* image, const ObjectTrackerParams& params, 
      CvRect* track_box, IplImage* likelihood)
  {
    // Return success
    return true;
  }

  //
  //
  //

  //---------------------------------------------------------------------------
  LINEMODAlgorithm::LINEMODAlgorithm() 
    : TrackingAlgorithm()
  {
  }

  //---------------------------------------------------------------------------
  LINEMODAlgorithm::~LINEMODAlgorithm()
  {
  }

  //---------------------------------------------------------------------------
  bool LINEMODAlgorithm::initialize(const IplImage* image, const ObjectTrackerParams& params, 
    const CvRect& init_bounding_box)
  {
    // Return success
    return true;
  }

  //---------------------------------------------------------------------------
  bool LINEMODAlgorithm::update(const IplImage* image, const ObjectTrackerParams& params, 
      CvRect* track_box, IplImage* likelihood)
  {
    // Return success
    return true;
  }

  //
  //
  //

  //---------------------------------------------------------------------------
  ObjectTracker::ObjectTracker(const ObjectTrackerParams& params)
    : initialized_(false), tracker_(NULL)
  {
    // Store configurable parameters internally
    set_params(params);

    // Allocate the proper tracking algorithm (note: error-checking that a valid
    // tracking algorithm parameter is used is done in the ObjectTrackerParams
    // constructor, so at this point we are confident it's valid).
    switch(params.algorithm_)
    {
    case ObjectTrackerParams::CV_ONLINEBOOSTING:
      tracker_ = new OnlineBoostingAlgorithm();
      break;
    case ObjectTrackerParams::CV_ONLINEMIL:
      tracker_ = new OnlineMILAlgorithm();
      break;
    case ObjectTrackerParams::CV_LINEMOD:
      tracker_ = new LINEMODAlgorithm();
      break;
    default:
      // By default, if an invalid choice somehow gets through lets use online boosting?
      // Or throw an error and don't continue?
      tracker_ = new OnlineBoostingAlgorithm();
      break;
    }
  }

  //---------------------------------------------------------------------------
  ObjectTracker::~ObjectTracker()
  {
    // Delete the tracking algorithm object, if it was allocated properly
    if (tracker_ != NULL)
    {
      delete tracker_;
    }
  }

  //---------------------------------------------------------------------------
  bool ObjectTracker::initialize(const IplImage* image, const CvRect& bounding_box)
  {
    // Initialize the tracker and if it works, set the flag that we're now initialized
    // to true so that update() can work properly.
    bool success = tracker_->initialize(image, tracker_params_, bounding_box);
    if (success)
    {
      initialized_ = true;
    }
    else
    {
      initialized_ = false;
    }

    // Return success or failure
    return success;
  }

  //---------------------------------------------------------------------------
  bool ObjectTracker::update(const IplImage* image, CvRect* track_box, IplImage* likelihood)
  {
    // First make sure we have already initialized.  Otherwise we can't continue.
    if (!initialized_)
    {
      std::cerr << "ObjectTracker::update() -- ERROR! The ObjectTracker needs to be initialized before updating.\n";
      return false;
    }

    // Update the tracker and return whether or not it succeeded
    bool success = tracker_->update(image, tracker_params_, track_box, likelihood);

    // Return success or failure
    return success;
  }

  //---------------------------------------------------------------------------
  void ObjectTracker::reset()
  {
    //
    // Reset the internal state of the tracker: should we just delete and re-allocate
    // the tracking algorithm object, or do something different?  Do we even really
    // need this functionality?
    //
  }

  //---------------------------------------------------------------------------
  void ObjectTracker::set_params(const ObjectTrackerParams& params)
  {
    tracker_params_ = params;
  }

  //---------------------------------------------------------------------------
  void ObjectTracker::save( const char* filename) const
  {
  }

  //---------------------------------------------------------------------------
  bool ObjectTracker::load( const char* filename)
  {
    // Return success
    return true;
  }

}
