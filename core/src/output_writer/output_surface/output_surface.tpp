#include "output_writer/output_surface/output_surface.h"
#include <algorithm>
#include <limits>

namespace OutputWriter {

template <typename Solver>
OutputSurface<Solver>::OutputSurface(DihuContext context)
  : context_(context["OutputSurface"]), solver_(context_), data_(context_),
    ownRankInvolvedInOutput_(true), timeStepNo_(0), currentTime_(0.0),
    updatePointPositions_(false), enableCsvFile_(false),
    enableVtpFile_(false), enableGeometryInCsvFile_(false) {}

template <typename Solver>
void OutputSurface<Solver>::initialize()
{
  if (initialized_)
    return;

  // initialize solvers
  solver_.initialize();
  data_.setData(solver_.data());
  data_.initialize();
  ownRankInvolvedInOutput_ = data_.ownRankInvolvedInOutput();

  // initialize output writers
  PythonConfig specificSettings = context_.getPythonConfig();

  // initialize points
  PyObject *samplingPointsPy = specificSettings.getOptionPyObject("samplingPoints");
  sampledPointsRequestedPositions_ =
    PythonUtility::convertFromPython<std::vector<Vec3>>::get(samplingPointsPy);

  if (!sampledPointsRequestedPositions_.empty())
  {
    filename_ = specificSettings.getOptionString("filename", "out/sampledPoints.csv");
    updatePointPositions_ = specificSettings.getOptionBool("updatePointPositions", false);
    xiTolerance_ = specificSettings.getOptionDouble("xiTolerance", 0.3,
                                                    PythonUtility::NonNegative);
    enableCsvFile_ = specificSettings.getOptionBool("enableCsvFile", true);
    enableVtpFile_ = specificSettings.getOptionBool("enableVtpFile", true);
    enableGeometryInCsvFile_ = specificSettings.getOptionBool("enableGeometryInCsvFile", true);
    enableGeometryFiles_ = specificSettings.getOptionBool("enableGeometryFiles", true);
  }

  LOG(DEBUG) << "OutputSurface: initialize output writers";

  // initialize output writer to use smaller rank subset that only contains the
  // ranks that have parts of the surface. If the last argument is not given, by
  // default the common rank subset would be used.
  if (ownRankInvolvedInOutput_)
  {
    rankSubset_ = data_.functionSpace()->meshPartition()->rankSubset();
    outputWriterManager_.initialize(context_, specificSettings, rankSubset_);

    if (filename_ != "")
    {
      std::ofstream file;
      Generic::openFile(file, filename_);  // recreate and truncate file
      file.close();
    }

    initializeSampledPoints();

    // write positions of found sampling points
    if (!sampledPointsRequestedPositions_.empty() && enableGeometryFiles_)
      writeFoundAndNotFoundPointGeometry();
  }

  initialized_ = true;
}

template <typename Solver>
void OutputSurface<Solver>::initializeSampledPoints()
{
  if (sampledPointsRequestedPositions_.empty())
    return;

  LOG(DEBUG) << "initialize sampled points";

  foundSampledPoints_.clear();

  // get the 2D function spaces
  this->data_.getFunctionSpaces(functionSpaces_);

  for (int functionSpaceNo = 0; functionSpaceNo < functionSpaces_.size(); functionSpaceNo++)
  {
    LOG(DEBUG) << "functionSpace " << functionSpaceNo << "/"
               << functionSpaces_.size() << ": "
               << functionSpaces_[functionSpaceNo]->meshName();
  }

  const int nDofsPerElement =
    DataSurface::FunctionSpaceFirstFieldVariable::nDofsPerElement();

  // ---------------------------------------------------------------------------
  // Fixed electrode mapping logic
  // ---------------------------------------------------------------------------
  // The old implementation used functionSpace->findPosition(...) and accepted
  // the first element for which the local coordinates xi were inside the element.
  // On curved/deformed 2D surfaces, this can select a wrong element that yields
  // valid-looking xi values but whose interpolated physical point is far away
  // from the requested electrode location. That leads to distorted positions in
  // electrodes.csv and, more importantly, wrong EMG sampling locations.
  //
  // The replacement below explicitly checks all local elements of all candidate
  // surface function spaces, reconstructs the physical position for every valid
  // candidate, and chooses the one with the smallest physical distance to the
  // requested sampling point.
  // ---------------------------------------------------------------------------

  // Report suspicious matches. This is deliberately only a warning, not fatal,
  // because some geometries may have small round-off or interpolation offsets.
  // Units are the same as the mesh coordinates, usually cm in OpenDiHu examples.
  const double warningScoreThreshold = 1e-3;

  // now we have a 2D function space for each face in functionSpaces_
  // loop over sampling points and find them in the function spaces
  for (int samplingPointNo = 0;
       samplingPointNo < sampledPointsRequestedPositions_.size();
       samplingPointNo++)
  {
    Vec3 point = sampledPointsRequestedPositions_[samplingPointNo];

    bool bestPointFound = false;
    double bestScore = std::numeric_limits<double>::max();
    FoundSampledPoint bestSampledPoint;

    // loop over function spaces for the faces
    for (int functionSpaceNo = 0; functionSpaceNo < functionSpaces_.size(); functionSpaceNo++)
    {
      std::shared_ptr<
        typename ::Data::OutputSurface<Data>::FunctionSpaceFirstFieldVariable>
        functionSpace = functionSpaces_[functionSpaceNo];

      LOG(DEBUG) << "point no " << samplingPointNo << ", point " << point;

      const element_no_t nElementsLocal = functionSpace->nElementsLocal();

      // Brute-force check of all local elements. For HD-sEMG electrode arrays this
      // is cheap compared with the full simulation and avoids accepting a wrong
      // first-hit element.
      for (element_no_t candidateElementNoLocal = 0;
           candidateElementNoLocal < nElementsLocal;
           candidateElementNoLocal++)
      {
        Vec2 xi;
        double residual = 0.0;

        bool candidateFound = functionSpace->pointIsInElement(
          point, candidateElementNoLocal, xi, residual, xiTolerance_);

        if (!candidateFound)
          continue;

        // determine actual position on the mesh for this candidate element
        std::array<Vec3, nDofsPerElement> elementalGeometryValues;
        functionSpace->geometryField().getElementValues(
          candidateElementNoLocal, elementalGeometryValues);

        Vec3 candidatePosition =
          functionSpace->template interpolateValueInElement<3>(
            elementalGeometryValues, xi);

        // compute physical distance score, smaller is better
        const double candidateScore = MathUtility::distance<3>(candidatePosition, point);

        if (candidateScore < bestScore)
        {
          bestScore = candidateScore;

          bestSampledPoint.samplingPointNo = samplingPointNo;
          bestSampledPoint.requestedPosition = sampledPointsRequestedPositions_[samplingPointNo];
          bestSampledPoint.functionSpaceNo = functionSpaceNo;
          bestSampledPoint.elementNoLocal = candidateElementNoLocal;
          bestSampledPoint.xi = xi;
          bestSampledPoint.position = candidatePosition;
          bestSampledPoint.score = candidateScore;

          bestPointFound = true;
        }
      }
    }

    if (bestPointFound)
    {
      foundSampledPoints_[samplingPointNo] = bestSampledPoint;

      LOG(DEBUG) << "point found at el. " << bestSampledPoint.elementNoLocal
                 << ", xi: " << bestSampledPoint.xi
                 << ", functionSpaceNo: " << bestSampledPoint.functionSpaceNo
                 << ", score: " << bestSampledPoint.score;

      if (bestSampledPoint.score > warningScoreThreshold)
      {
        LOG(WARNING) << "OutputSurface: suspicious sampling point mapping, point "
                     << samplingPointNo
                     << ", requested=" << bestSampledPoint.requestedPosition
                     << ", mapped=" << bestSampledPoint.position
                     << ", distance=" << bestSampledPoint.score
                     << ", elementNoLocal=" << bestSampledPoint.elementNoLocal
                     << ", xi=" << bestSampledPoint.xi
                     << ", functionSpaceNo=" << bestSampledPoint.functionSpaceNo;
      }
    }
    else
    {
      LOG(WARNING) << "OutputSurface: sampling point " << samplingPointNo
                   << " was not found in any local surface element, requested="
                   << point;
    }
  }
}

template <typename Solver>
void OutputSurface<Solver>::advanceTimeSpan(bool withOutputWritersEnabled)
{
  solver_.advanceTimeSpan(withOutputWritersEnabled);

  LOG(DEBUG) << "OutputSurface: writeOutput, ownRankInvolvedInOutput_: "
             << ownRankInvolvedInOutput_;

  // if the own rank has a part of the surface that will be written
  if (ownRankInvolvedInOutput_)
  {
    // write 2D surface files
    if (withOutputWritersEnabled)
      outputWriterManager_.writeOutput(data_, timeStepNo_++, currentTime_);

    // write out values at points
    writeSampledPointValues();

    // write positions of found sampling points
    if (updatePointPositions_)
    {
      initializeSampledPoints();
      writeFoundAndNotFoundPointGeometry();
    }
  }
}

template <typename Solver>
void OutputSurface<Solver>::run()
{
  initialize();
  solver_.run();

  LOG(DEBUG) << "OutputSurface: writeOutput";

  if (ownRankInvolvedInOutput_)
  {
    outputWriterManager_.writeOutput(data_);
  }
}

template <typename Solver>
void OutputSurface<Solver>::reset()
{
  solver_.reset();
}

template <typename Solver>
void OutputSurface<Solver>::setTimeSpan(double startTime, double endTime)
{
  currentTime_ = startTime;
  solver_.setTimeSpan(startTime, endTime);
}

//! call the output writer on the data object, output files will contain
//! currentTime, with callCountIncrement != 1 output timesteps can be skipped
template <typename Solver>
void OutputSurface<Solver>::callOutputWriter(int timeStepNo, double currentTime,
                                             int callCountIncrement)
{
  // call output writers of nested solvers
  solver_.callOutputWriter(timeStepNo, currentTime, callCountIncrement);

  // call own output writers
  if (ownRankInvolvedInOutput_)
  {
    outputWriterManager_.writeOutput(data_, timeStepNo, currentTime,
                                     callCountIncrement);
  }
}

template <typename Solver>
typename OutputSurface<Solver>::Data &OutputSurface<Solver>::data()
{
  return solver_.data();
}

template <typename Solver>
std::shared_ptr<typename OutputSurface<Solver>::SlotConnectorDataType>
OutputSurface<Solver>::getSlotConnectorData()
{
  return solver_.getSlotConnectorData();
}

}  // namespace OutputWriter
