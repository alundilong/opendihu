#include "output_writer/output_surface/output_surface.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
  // Fast and robust electrode mapping
  // ---------------------------------------------------------------------------
  // The old OpenDiHu implementation accepted the first element for which the
  // local element coordinates xi looked valid. On curved 2D surfaces this may
  // be the wrong element. The previous robust fix scanned all elements for
  // every electrode, which is correct but slow.
  //
  // This implementation keeps the robustness but accelerates the search:
  //   1. Build a local spatial hash of surface-element axis-aligned bounding
  //      boxes on every MPI rank that owns a part of the OutputSurface.
  //   2. For every requested sampling point, test only nearby candidate
  //      elements from the hash.
  //   3. Compute the physical reconstructed point for every valid candidate
  //      and keep the candidate with the smallest physical distance.
  //   4. If the local hash search finds no good candidate, fall back to a full
  //      local scan. This preserves robustness and parallel compatibility.
  //
  // Parallel behavior:
  //   - Each MPI rank determines its best local candidate.
  //   - The existing OutputSurface write path gathers candidates from ranks and
  //     sorts/removes duplicates by score, so the globally best candidate wins.
  //   - Therefore this code must not LOG(FATAL) on a bad local candidate before
  //     the global selection step. It only prints warnings.
  // ---------------------------------------------------------------------------

  // Distance threshold for triggering a full fallback search and warning.
  // Units are mesh coordinate units, usually cm in the OpenDiHu examples.
  const double suspiciousScoreThreshold = 1e-3;

  struct SearchElement
  {
    int functionSpaceNo;
    element_no_t elementNoLocal;
    std::array<double,3> minCorner;
    std::array<double,3> maxCorner;
  };

  struct SpatialKey
  {
    long long i;
    long long j;
    long long k;

    bool operator==(SpatialKey const &other) const
    {
      return i == other.i && j == other.j && k == other.k;
    }
  };

  struct SpatialKeyHash
  {
    std::size_t operator()(SpatialKey const &key) const
    {
      // Mix three signed integer coordinates. The constants are common large
      // primes used for spatial hashing.
      const long long h = key.i * 73856093LL ^ key.j * 19349663LL ^ key.k * 83492791LL;
      return std::hash<long long>()(h);
    }
  };

  std::vector<SearchElement> searchElements;
  searchElements.reserve(1024);

  double sumElementDiagonal = 0.0;
  long long nElementDiagonals = 0;

  // ---------------------------------------------------------------------------
  // Build local element AABBs and estimate a characteristic element size.
  // ---------------------------------------------------------------------------
  for (int functionSpaceNo = 0; functionSpaceNo < functionSpaces_.size(); functionSpaceNo++)
  {
    std::shared_ptr<
      typename ::Data::OutputSurface<Data>::FunctionSpaceFirstFieldVariable>
      functionSpace = functionSpaces_[functionSpaceNo];

    const element_no_t nElementsLocal = functionSpace->nElementsLocal();

    for (element_no_t elementNoLocal = 0; elementNoLocal < nElementsLocal; elementNoLocal++)
    {
      std::array<Vec3, nDofsPerElement> elementalGeometryValues;
      functionSpace->geometryField().getElementValues(elementNoLocal, elementalGeometryValues);

      SearchElement element;
      element.functionSpaceNo = functionSpaceNo;
      element.elementNoLocal = elementNoLocal;

      for (int component = 0; component < 3; component++)
      {
        element.minCorner[component] = elementalGeometryValues[0][component];
        element.maxCorner[component] = elementalGeometryValues[0][component];
      }

      for (int dofNo = 1; dofNo < nDofsPerElement; dofNo++)
      {
        for (int component = 0; component < 3; component++)
        {
          element.minCorner[component] = std::min(element.minCorner[component], elementalGeometryValues[dofNo][component]);
          element.maxCorner[component] = std::max(element.maxCorner[component], elementalGeometryValues[dofNo][component]);
        }
      }

      const double dx = element.maxCorner[0] - element.minCorner[0];
      const double dy = element.maxCorner[1] - element.minCorner[1];
      const double dz = element.maxCorner[2] - element.minCorner[2];
      const double diagonal = std::sqrt(dx*dx + dy*dy + dz*dz);

      if (diagonal > 0.0 && std::isfinite(diagonal))
      {
        sumElementDiagonal += diagonal;
        nElementDiagonals++;
      }

      searchElements.push_back(element);
    }
  }

  double meanElementDiagonal = 1.0;
  if (nElementDiagonals > 0)
    meanElementDiagonal = sumElementDiagonal / double(nElementDiagonals);

  if (!(meanElementDiagonal > 0.0) || !std::isfinite(meanElementDiagonal))
    meanElementDiagonal = 1.0;

  // Spatial hash cell size. A value larger than one element diagonal reduces
  // hash overhead while still keeping candidate lists small.
  const double hashCellSize = std::max(2.0 * meanElementDiagonal, 1e-12);

  // Expand AABBs slightly before inserting into the spatial hash. This avoids
  // missing candidates for points near element boundaries or slightly off the
  // curved surface due to round-off.
  const double aabbPadding = std::max(0.10 * meanElementDiagonal, 1e-12);

  auto cellCoordinate = [&](double x) -> long long
  {
    return static_cast<long long>(std::floor(x / hashCellSize));
  };

  auto makeKey = [&](long long i, long long j, long long k) -> SpatialKey
  {
    SpatialKey key;
    key.i = i;
    key.j = j;
    key.k = k;
    return key;
  };

  std::unordered_map<SpatialKey, std::vector<std::size_t>, SpatialKeyHash> spatialHash;
  spatialHash.reserve(searchElements.size() * 2 + 1);

  for (std::size_t searchElementNo = 0; searchElementNo < searchElements.size(); searchElementNo++)
  {
    SearchElement const &element = searchElements[searchElementNo];

    const long long iMin = cellCoordinate(element.minCorner[0] - aabbPadding);
    const long long jMin = cellCoordinate(element.minCorner[1] - aabbPadding);
    const long long kMin = cellCoordinate(element.minCorner[2] - aabbPadding);

    const long long iMax = cellCoordinate(element.maxCorner[0] + aabbPadding);
    const long long jMax = cellCoordinate(element.maxCorner[1] + aabbPadding);
    const long long kMax = cellCoordinate(element.maxCorner[2] + aabbPadding);

    for (long long k = kMin; k <= kMax; k++)
      for (long long j = jMin; j <= jMax; j++)
        for (long long i = iMin; i <= iMax; i++)
          spatialHash[makeKey(i,j,k)].push_back(searchElementNo);
  }

  LOG(DEBUG) << "OutputSurface spatial search index: " << searchElements.size()
             << " local surface elements, " << spatialHash.size()
             << " occupied hash cells, mean element diagonal " << meanElementDiagonal
             << ", hash cell size " << hashCellSize;

  auto evaluateCandidate = [&](std::size_t searchElementNo, Vec3 const &point,
                               int samplingPointNo, FoundSampledPoint &bestSampledPoint,
                               double &bestScore, bool &bestPointFound) -> bool
  {
    SearchElement const &searchElement = searchElements[searchElementNo];

    std::shared_ptr<
      typename ::Data::OutputSurface<Data>::FunctionSpaceFirstFieldVariable>
      functionSpace = functionSpaces_[searchElement.functionSpaceNo];

    Vec2 xi;
    double residual = 0.0;

    bool candidateFound = functionSpace->pointIsInElement(
      point, searchElement.elementNoLocal, xi, residual, xiTolerance_);

    if (!candidateFound)
      return false;

    std::array<Vec3, nDofsPerElement> elementalGeometryValues;
    functionSpace->geometryField().getElementValues(searchElement.elementNoLocal, elementalGeometryValues);

    Vec3 candidatePosition =
      functionSpace->template interpolateValueInElement<3>(elementalGeometryValues, xi);

    const double candidateScore = MathUtility::distance<3>(candidatePosition, point);

    if (candidateScore < bestScore)
    {
      bestScore = candidateScore;

      bestSampledPoint.samplingPointNo = samplingPointNo;
      bestSampledPoint.requestedPosition = sampledPointsRequestedPositions_[samplingPointNo];
      bestSampledPoint.functionSpaceNo = searchElement.functionSpaceNo;
      bestSampledPoint.elementNoLocal = searchElement.elementNoLocal;
      bestSampledPoint.xi = xi;
      bestSampledPoint.position = candidatePosition;
      bestSampledPoint.score = candidateScore;

      bestPointFound = true;
    }

    return true;
  };

  auto gatherNearbyCandidates = [&](Vec3 const &point, int radius,
                                    std::vector<std::size_t> &nearbyCandidates)
  {
    nearbyCandidates.clear();
    std::unordered_set<std::size_t> seen;

    const long long i0 = cellCoordinate(point[0]);
    const long long j0 = cellCoordinate(point[1]);
    const long long k0 = cellCoordinate(point[2]);

    for (long long dk = -radius; dk <= radius; dk++)
    {
      for (long long dj = -radius; dj <= radius; dj++)
      {
        for (long long di = -radius; di <= radius; di++)
        {
          typename std::unordered_map<SpatialKey, std::vector<std::size_t>, SpatialKeyHash>::const_iterator iter =
            spatialHash.find(makeKey(i0 + di, j0 + dj, k0 + dk));

          if (iter == spatialHash.end())
            continue;

          for (std::size_t candidateNo : iter->second)
          {
            if (seen.insert(candidateNo).second)
              nearbyCandidates.push_back(candidateNo);
          }
        }
      }
    }
  };

  // ---------------------------------------------------------------------------
  // Locate all requested sampling points.
  // ---------------------------------------------------------------------------
  std::vector<std::size_t> nearbyCandidates;
  std::unordered_set<std::size_t> checkedCandidates;

  for (int samplingPointNo = 0;
       samplingPointNo < sampledPointsRequestedPositions_.size();
       samplingPointNo++)
  {
    Vec3 point = sampledPointsRequestedPositions_[samplingPointNo];

    bool bestPointFound = false;
    double bestScore = std::numeric_limits<double>::max();
    FoundSampledPoint bestSampledPoint;

    checkedCandidates.clear();

    // First attempt: local hash search. Increase search radius gradually. In
    // most cases radius 1 is enough; radius 2/3 covers points close to cell
    // boundaries or very curved surfaces.
    const int maxHashRadius = 3;
    for (int radius = 1; radius <= maxHashRadius; radius++)
    {
      gatherNearbyCandidates(point, radius, nearbyCandidates);

      bool foundInThisRadius = false;
      for (std::size_t candidateNo : nearbyCandidates)
      {
        checkedCandidates.insert(candidateNo);
        bool candidateAccepted = evaluateCandidate(candidateNo, point, samplingPointNo,
                                                   bestSampledPoint, bestScore, bestPointFound);
        foundInThisRadius = foundInThisRadius || candidateAccepted;
      }

      // If we already found a very good candidate, stop early. Otherwise keep
      // expanding before falling back to a full scan.
      if (foundInThisRadius && bestScore <= suspiciousScoreThreshold)
        break;
    }

    // Fallback full local scan. This keeps the method as robust as the brute
    // force version, but only triggers for hard cases or suspicious mappings.
    if (!bestPointFound || bestScore > suspiciousScoreThreshold)
    {
      const double scoreBeforeFallback = bestScore;
      const bool foundBeforeFallback = bestPointFound;

      for (std::size_t candidateNo = 0; candidateNo < searchElements.size(); candidateNo++)
      {
        if (checkedCandidates.find(candidateNo) != checkedCandidates.end())
          continue;

        evaluateCandidate(candidateNo, point, samplingPointNo,
                          bestSampledPoint, bestScore, bestPointFound);
      }

      if (!foundBeforeFallback && bestPointFound)
      {
        LOG(DEBUG) << "OutputSurface: sampling point " << samplingPointNo
                   << " was found only by fallback full scan, score=" << bestScore;
      }
      else if (foundBeforeFallback && bestScore < scoreBeforeFallback)
      {
        LOG(DEBUG) << "OutputSurface: fallback improved sampling point "
                   << samplingPointNo << " score from " << scoreBeforeFallback
                   << " to " << bestScore;
      }
    }

    if (bestPointFound)
    {
      foundSampledPoints_[samplingPointNo] = bestSampledPoint;

      LOG(DEBUG) << "point " << samplingPointNo
                 << " found at el. " << bestSampledPoint.elementNoLocal
                 << ", xi: " << bestSampledPoint.xi
                 << ", functionSpaceNo: " << bestSampledPoint.functionSpaceNo
                 << ", score: " << bestSampledPoint.score;

      if (bestSampledPoint.score > suspiciousScoreThreshold)
      {
        // Warning only. In parallel, another rank may still have a better global
        // candidate. The existing gather/sort-by-score path will choose it.
        LOG(WARNING) << "OutputSurface: suspicious local sampling point mapping, point "
                     << samplingPointNo
                     << ", requested=" << bestSampledPoint.requestedPosition
                     << ", mapped=" << bestSampledPoint.position
                     << ", local distance=" << bestSampledPoint.score
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
