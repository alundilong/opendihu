#include "output_writer/paraview/paraview.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>

#include "easylogging++.h"

#include "base64.h"
#include "control/diagnostic_tool/performance_measurement.h"
#include "output_writer/paraview/loop_collect_mesh_properties.h"
#include "output_writer/paraview/loop_get_geometry_field_nodal_values.h"
#include "output_writer/paraview/loop_get_nodal_values.h"
#include "output_writer/paraview/loop_output.h"
#include "output_writer/paraview/poly_data_properties_for_mesh.h"

namespace OutputWriter
{

template<typename FieldVariablesForOutputWriterType>
void Paraview::writePolyDataFile(
    const FieldVariablesForOutputWriterType &fieldVariables,
    std::set<std::string> &meshNames)
{
  // Output a single *.vtp file containing all compatible 1D meshes.
  //
  // Modified behavior:
  //   Each individual OpenDiHu 1D mesh is written as one VTK POLY_LINE.
  //   The original implementation wrote every 1D finite element as a
  //   separate two-point VTK line cell. For large fiber simulations this
  //   produced millions of line cells and required an expensive vtkStripper
  //   postprocessing step.
  //
  // The point-data arrays and geometry retain exactly the same ordering as
  // before. Only the <Lines> connectivity and NumberOfLines are changed.

  bool meshPropertiesInitialized = !meshPropertiesPolyDataFile_.empty();
  std::vector<std::string> meshNamesVector;

  if (!meshPropertiesInitialized)
  {
    Control::PerformanceMeasurement::start("durationParaview1DInit");

    // Collect mesh sizes and preserve the traversal order of the meshes.
    // The order is important because loopGetNodalValues and
    // loopGetGeometryFieldNodalValues append values in this traversal order.
    ParaviewLoopOverTuple::loopCollectMeshProperties<FieldVariablesForOutputWriterType>(
        fieldVariables,
        meshPropertiesPolyDataFile_,
        meshNamesVector);

    Control::PerformanceMeasurement::stop("durationParaview1DInit");
  }

  VLOG(1) << "writePolyDataFile on rankSubset_: " << *this->rankSubset_;
  assert(this->rankSubset_);
  VLOG(1) << "meshPropertiesPolyDataFile_: " << meshPropertiesPolyDataFile_;

  if (!meshPropertiesInitialized)
  {
    Control::PerformanceMeasurement::start("durationParaview1DInit");

    // Parse the collected mesh properties and determine which 1D meshes can
    // be combined into the same VTK Piece.
    for (std::map<std::string, PolyDataPropertiesForMesh>::iterator iter =
             meshPropertiesPolyDataFile_.begin();
         iter != meshPropertiesPolyDataFile_.end();
         ++iter)
    {
      const std::string meshName = iter->first;

      // Do not combine meshes other than 1D meshes.
      if (iter->second.dimensionality != 1)
        continue;

      bool combineMesh = true;

      // Check whether this mesh has the same point-data layout as the
      // previously accepted 1D meshes.
      if (!vtkPiece1D_.properties.pointDataArrays.empty())
      {
        if (vtkPiece1D_.properties.pointDataArrays.size() !=
            iter->second.pointDataArrays.size())
        {
          LOG(DEBUG)
              << "Mesh " << meshName << " cannot be combined with "
              << vtkPiece1D_.meshNamesCombinedMeshes
              << ". Number of field variables mismatches for " << meshName
              << " (is " << iter->second.pointDataArrays.size()
              << " instead of "
              << vtkPiece1D_.properties.pointDataArrays.size() << ")";

          combineMesh = false;
        }
        else
        {
          for (std::size_t j = 0; j < iter->second.pointDataArrays.size(); ++j)
          {
            if (vtkPiece1D_.properties.pointDataArrays[j].name !=
                iter->second.pointDataArrays[j].name)
            {
              LOG(DEBUG)
                  << "Mesh " << meshName << " cannot be combined with "
                  << vtkPiece1D_.meshNamesCombinedMeshes
                  << ". Field variable names mismatch for " << meshName
                  << " (there is \""
                  << vtkPiece1D_.properties.pointDataArrays[j].name
                  << "\" instead of \""
                  << iter->second.pointDataArrays[j].name << "\")";

              combineMesh = false;
              break;
            }

            if (vtkPiece1D_.properties.pointDataArrays[j].nComponents !=
                iter->second.pointDataArrays[j].nComponents)
            {
              LOG(DEBUG)
                  << "Mesh " << meshName << " cannot be combined with "
                  << vtkPiece1D_.meshNamesCombinedMeshes
                  << ". Number of components mismatches for field variable \""
                  << iter->second.pointDataArrays[j].name << "\".";

              combineMesh = false;
              break;
            }
          }
        }

        if (combineMesh)
        {
          VLOG(1)
              << "Combine mesh " << meshName << " with "
              << vtkPiece1D_.meshNamesCombinedMeshes << ", add "
              << iter->second.nPointsLocal << " points, "
              << iter->second.nCellsLocal << " elements to "
              << vtkPiece1D_.properties.nPointsLocal << " points, "
              << vtkPiece1D_.properties.nCellsLocal << " elements";

          vtkPiece1D_.properties.nPointsLocal += iter->second.nPointsLocal;
          vtkPiece1D_.properties.nCellsLocal += iter->second.nCellsLocal;
          vtkPiece1D_.properties.nPointsGlobal += iter->second.nPointsGlobal;
          vtkPiece1D_.properties.nCellsGlobal += iter->second.nCellsGlobal;
          vtkPiece1D_.setVTKValues();
        }
      }
      else
      {
        VLOG(1) << "this is the first 1D mesh";

        // Properties are not yet assigned.
        vtkPiece1D_.properties = iter->second;
        vtkPiece1D_.setVTKValues();
      }

      VLOG(1) << "combineMesh: " << combineMesh;

      if (combineMesh)
        vtkPiece1D_.meshNamesCombinedMeshes.insert(meshName);
    }

    // Preserve the same mesh order used by the tuple traversal that appends
    // geometry and solution values. Do not generate connectivity by iterating
    // directly over std::set, because its lexical order can differ from the
    // field-variable traversal order.
    vtkPiece1D_.meshNamesCombinedMeshesVector.clear();

    for (const std::string &meshName : meshNamesVector)
    {
      if (vtkPiece1D_.meshNamesCombinedMeshes.count(meshName) > 0)
      {
        vtkPiece1D_.meshNamesCombinedMeshesVector.push_back(meshName);
      }
    }

    // Defensive fallback: append any accepted mesh that was not encountered
    // in meshNamesVector. This should normally never be needed.
    for (const std::string &meshName : vtkPiece1D_.meshNamesCombinedMeshes)
    {
      if (std::find(
              vtkPiece1D_.meshNamesCombinedMeshesVector.begin(),
              vtkPiece1D_.meshNamesCombinedMeshesVector.end(),
              meshName) ==
          vtkPiece1D_.meshNamesCombinedMeshesVector.end())
      {
        LOG(WARNING)
            << "The 1D mesh \"" << meshName
            << "\" was not present in meshNamesVector. Appending it to the "
               "combined-mesh order.";

        vtkPiece1D_.meshNamesCombinedMeshesVector.push_back(meshName);
      }
    }

    LOG(DEBUG)
        << "vtkPiece1D_: meshNamesCombinedMeshes: "
        << vtkPiece1D_.meshNamesCombinedMeshes
        << ", meshNamesCombinedMeshesVector size: "
        << vtkPiece1D_.meshNamesCombinedMeshesVector.size()
        << ", properties: " << vtkPiece1D_.properties
        << ", firstScalarName: " << vtkPiece1D_.firstScalarName
        << ", firstVectorName: " << vtkPiece1D_.firstVectorName;

    Control::PerformanceMeasurement::stop("durationParaview1DInit");
  }

  meshNames = vtkPiece1D_.meshNamesCombinedMeshes;

  // If there are no compatible 1D meshes, return.
  if (meshNames.empty())
    return;

  if (vtkPiece1D_.meshNamesCombinedMeshesVector.empty())
  {
    LOG(FATAL)
        << "The ordered list of combined 1D meshes is empty although "
        << vtkPiece1D_.meshNamesCombinedMeshes.size()
        << " meshes were selected.";
  }

  if (!meshPropertiesInitialized)
  {
    // Add field variable "partitioning" with one component.
    PolyDataPropertiesForMesh::DataArrayName dataArrayName;
    dataArrayName.name = "partitioning";
    dataArrayName.nComponents = 1;
    dataArrayName.componentNames = std::vector<std::string>(1, "rankNo");
    vtkPiece1D_.properties.pointDataArrays.push_back(dataArrayName);
  }

  // Determine the local polyline count and validate that the ordered list
  // covers the exact point range written by the point-data/geometry loops.
  int nLinesLocal1D = 0;
  global_no_t nPointsFromOrderedMeshesLocal = 0;
  int nConnectivityValuesLocal = 0;

  for (const std::string &meshName :
       vtkPiece1D_.meshNamesCombinedMeshesVector)
  {
    const std::map<std::string, PolyDataPropertiesForMesh>::const_iterator iter =
        meshPropertiesPolyDataFile_.find(meshName);

    if (iter == meshPropertiesPolyDataFile_.end())
    {
      LOG(FATAL)
          << "No PolyDataPropertiesForMesh entry exists for combined 1D mesh \""
          << meshName << "\".";
    }

    const PolyDataPropertiesForMesh &properties = iter->second;

    if (properties.dimensionality != 1)
    {
      LOG(FATAL)
          << "Mesh \"" << meshName
          << "\" is in the combined 1D mesh list but has dimensionality "
          << properties.dimensionality << ".";
    }

    nPointsFromOrderedMeshesLocal += properties.nPointsLocal;

    // A VTK line/polyline needs at least two points.
    if (properties.nPointsLocal >= 2)
    {
      ++nLinesLocal1D;

      if (properties.nPointsLocal >
          static_cast<global_no_t>(std::numeric_limits<int>::max()))
      {
        LOG(FATAL)
            << "Mesh \"" << meshName << "\" contains "
            << properties.nPointsLocal
            << " local points, which exceeds the Int32 VTK connectivity limit.";
      }

      if (nConnectivityValuesLocal >
          std::numeric_limits<int>::max() -
              static_cast<int>(properties.nPointsLocal))
      {
        LOG(FATAL)
            << "The local VTK connectivity array exceeds the Int32 limit.";
      }

      nConnectivityValuesLocal += static_cast<int>(properties.nPointsLocal);
    }
  }

  if (nPointsFromOrderedMeshesLocal !=
      vtkPiece1D_.properties.nPointsLocal)
  {
    LOG(FATAL)
        << "The ordered 1D mesh list covers "
        << nPointsFromOrderedMeshesLocal << " local points, but vtkPiece1D_ "
        << "contains " << vtkPiece1D_.properties.nPointsLocal
        << " local points. The connectivity order would not match the "
           "geometry/solution order.";
  }

  // Determine filename and broadcast it from rank 0.
  std::stringstream filename;
  filename << this->filenameBaseWithNo_ << ".vtp";

  int filenameLength = filename.str().length();

  MPIUtility::handleReturnValue(
      MPI_Bcast(
          &filenameLength,
          1,
          MPI_INT,
          0,
          this->rankSubset_->mpiCommunicator()),
      "MPI_Bcast (3)");

  std::vector<char> receiveBuffer(filenameLength + 1, char(0));
  std::strcpy(receiveBuffer.data(), filename.str().c_str());

  MPIUtility::handleReturnValue(
      MPI_Bcast(
          receiveBuffer.data(),
          filenameLength,
          MPI_CHAR,
          0,
          this->rankSubset_->mpiCommunicator()),
      "MPI_Bcast (4)");

  std::string filenameStr(receiveBuffer.begin(), receiveBuffer.end());

  // Remove an existing file on rank 0. Synchronization is provided by the
  // following MPI operations.
  assert(this->rankSubset_);
  const int ownRankNo = this->rankSubset_->ownRankNo();

  if (ownRankNo == 0)
  {
    std::ofstream file;
    Generic::openFile(file, filenameStr);
    file.close();
    std::remove(filenameStr.c_str());
  }

  if (!meshPropertiesInitialized)
  {
    // Exchange point offsets and global point/polyline counts.
    //
    // nCellsPreviousRanks1D_ is retained as a class member for compatibility
    // with the original writer but is no longer used here, because one output
    // VTK cell now represents one complete local 1D mesh rather than one FE
    // element.
    nCellsPreviousRanks1D_ = 0;
    nPointsPreviousRanks1D_ = 0;
    nPointsGlobal1D_ = 0;
    nLinesGlobal1D_ = 0;

    Control::PerformanceMeasurement::start("durationParaview1DInit");
    Control::PerformanceMeasurement::start("durationParaview1DReduction");

    MPIUtility::handleReturnValue(
        MPI_Exscan(
            &vtkPiece1D_.properties.nPointsLocal,
            &nPointsPreviousRanks1D_,
            1,
            MPI_INT,
            MPI_SUM,
            this->rankSubset_->mpiCommunicator()),
        "MPI_Exscan");

    // MPI_Exscan leaves the receive buffer undefined on rank 0.
    if (ownRankNo == 0)
      nPointsPreviousRanks1D_ = 0;

    MPIUtility::handleReturnValue(
        MPI_Reduce(
            &vtkPiece1D_.properties.nPointsLocal,
            &nPointsGlobal1D_,
            1,
            MPI_INT,
            MPI_SUM,
            0,
            this->rankSubset_->mpiCommunicator()),
        "MPI_Reduce");

    MPIUtility::handleReturnValue(
        MPI_Reduce(
            &nLinesLocal1D,
            &nLinesGlobal1D_,
            1,
            MPI_INT,
            MPI_SUM,
            0,
            this->rankSubset_->mpiCommunicator()),
        "MPI_Reduce");

    Control::PerformanceMeasurement::stop("durationParaview1DReduction");
    Control::PerformanceMeasurement::stop("durationParaview1DInit");
  }

  // Offsets in a VTK XML CellArray refer to positions in the concatenated
  // connectivity array, not to point IDs. The number of connectivity entries
  // can differ from the number of points if a rank owns a one-point mesh
  // portion, so determine this prefix independently.
  int nConnectivityValuesPreviousRanks = 0;

  MPIUtility::handleReturnValue(
      MPI_Exscan(
          &nConnectivityValuesLocal,
          &nConnectivityValuesPreviousRanks,
          1,
          MPI_INT,
          MPI_SUM,
          this->rankSubset_->mpiCommunicator()),
      "MPI_Exscan");

  if (ownRankNo == 0)
    nConnectivityValuesPreviousRanks = 0;

  // -------------------------------------------------------------------------
  // Construct one VTK POLY_LINE for every individual local OpenDiHu 1D mesh.
  // -------------------------------------------------------------------------
  std::vector<int> connectivityValues;
  std::vector<int> offsetValues;

  connectivityValues.reserve(
      static_cast<std::size_t>(nConnectivityValuesLocal));
  offsetValues.reserve(static_cast<std::size_t>(nLinesLocal1D));

  int localPointOffset = 0;
  int cumulativeConnectivityOffset = nConnectivityValuesPreviousRanks;

  for (const std::string &meshName :
       vtkPiece1D_.meshNamesCombinedMeshesVector)
  {
    const PolyDataPropertiesForMesh &properties =
        meshPropertiesPolyDataFile_.at(meshName);

    const int nPointsForMesh =
        static_cast<int>(properties.nPointsLocal);

    if (nPointsForMesh >= 2)
    {
      // Geometry and point-data arrays are concatenated mesh-by-mesh in the
      // same traversal order. Therefore, the global point IDs for this mesh
      // form one contiguous range.
      for (int pointNo = 0; pointNo < nPointsForMesh; ++pointNo)
      {
        connectivityValues.push_back(
            nPointsPreviousRanks1D_ + localPointOffset + pointNo);
      }

      cumulativeConnectivityOffset += nPointsForMesh;
      offsetValues.push_back(cumulativeConnectivityOffset);
    }

    // Advance for every mesh, including a possible one-point local portion,
    // because its point still occupies a slot in the combined point arrays.
    localPointOffset += nPointsForMesh;
  }

  if (localPointOffset != vtkPiece1D_.properties.nPointsLocal)
  {
    LOG(FATAL)
        << "Incorrect local point offset while constructing 1D VTK "
           "polylines. Processed "
        << localPointOffset << " points, expected "
        << vtkPiece1D_.properties.nPointsLocal << ".";
  }

  if (static_cast<int>(connectivityValues.size()) !=
      nConnectivityValuesLocal)
  {
    LOG(FATAL)
        << "Incorrect 1D VTK connectivity size. Created "
        << connectivityValues.size() << " entries, expected "
        << nConnectivityValuesLocal << ".";
  }

  if (static_cast<int>(offsetValues.size()) != nLinesLocal1D)
  {
    LOG(FATAL)
        << "Incorrect number of local 1D VTK polylines. Created "
        << offsetValues.size() << ", expected " << nLinesLocal1D << ".";
  }

  LOG(DEBUG)
      << "Created " << offsetValues.size()
      << " local VTK polylines from " << connectivityValues.size()
      << " local point references.";

  // Collect all data for the field variables, organized by field-variable
  // name.
  std::map<std::string, std::vector<double>> fieldVariableValues;

  ParaviewLoopOverTuple::loopGetNodalValues<FieldVariablesForOutputWriterType>(
      fieldVariables,
      vtkPiece1D_.meshNamesCombinedMeshes,
      fieldVariableValues);

  assert(!fieldVariableValues.empty());

  fieldVariableValues["partitioning"].resize(
      vtkPiece1D_.properties.nPointsLocal,
      static_cast<double>(this->rankSubset_->ownRankNo()));

  if (fieldVariableValues.size() !=
      vtkPiece1D_.properties.pointDataArrays.size())
  {
    LOG(DEBUG)
        << "n field variable values: " << fieldVariableValues.size()
        << ", n point data arrays: "
        << vtkPiece1D_.properties.pointDataArrays.size();

    LOG(DEBUG)
        << "vtkPiece1D_.meshNamesCombinedMeshes: "
        << vtkPiece1D_.meshNamesCombinedMeshes;

    std::stringstream pointDataArraysNames;

    for (std::size_t i = 0;
         i < vtkPiece1D_.properties.pointDataArrays.size();
         ++i)
    {
      pointDataArraysNames
          << vtkPiece1D_.properties.pointDataArrays[i].name << " ";
    }

    LOG(DEBUG)
        << "pointDataArraysNames: " << pointDataArraysNames.str();
  }

  assert(
      fieldVariableValues.size() ==
      vtkPiece1D_.properties.pointDataArrays.size());

#ifndef NDEBUG
  LOG(DEBUG) << "fieldVariableValues: ";

  for (std::map<std::string, std::vector<double>>::iterator iter =
           fieldVariableValues.begin();
       iter != fieldVariableValues.end();
       ++iter)
  {
    LOG(DEBUG) << iter->first;
  }
#endif

  // Check whether field-variable names have changed since initialization.
  for (std::vector<PolyDataPropertiesForMesh::DataArrayName>::iterator
           pointDataArrayIter =
               vtkPiece1D_.properties.pointDataArrays.begin();
       pointDataArrayIter !=
       vtkPiece1D_.properties.pointDataArrays.end();
       ++pointDataArrayIter)
  {
    LOG(DEBUG)
        << " field variable \"" << pointDataArrayIter->name << "\".";

    if (fieldVariableValues.find(pointDataArrayIter->name) ==
        fieldVariableValues.end())
    {
      LOG(DEBUG)
          << "Field variable names have changed, reinitialize Paraview "
             "output writer.";

      meshPropertiesInitialized = false;
      meshPropertiesPolyDataFile_.clear();
      vtkPiece1D_ = VTKPiece();

      writePolyDataFile(fieldVariables, meshNames);
      return;
    }
  }

  // Collect geometry-field values in the same point order.
  std::vector<double> geometryFieldValues;

  ParaviewLoopOverTuple::loopGetGeometryFieldNodalValues<
      FieldVariablesForOutputWriterType>(
      fieldVariables,
      vtkPiece1D_.meshNamesCombinedMeshes,
      geometryFieldValues);

  if (vtkPiece1D_.meshNamesCombinedMeshes.empty())
  {
    LOG(ERROR)
        << "There are no 1D meshes that could be combined, but Paraview "
           "output with combineFiles=True was specified.\n"
        << "(This only works for 1D meshes.)";
  }

  LOG(DEBUG)
      << "Combined mesh from "
      << vtkPiece1D_.meshNamesCombinedMeshes;

  const int nOutputFileParts =
      4 + vtkPiece1D_.properties.pointDataArrays.size();

  // Transform current time to string.
  std::vector<double> time(1, this->currentTime_);
  std::string stringTime;

  if (binaryOutput_)
    stringTime = Paraview::encodeBase64Float(time.begin(), time.end());
  else
    stringTime = Paraview::convertToAscii(time, fixedFormat_);

  // Create the XML structure in separate parts. The numerical arrays are
  // inserted between these parts using MPI-IO below.
  std::vector<std::stringstream> outputFileParts(nOutputFileParts);
  int outputFilePartNo = 0;

  outputFileParts[outputFilePartNo]
      << "<?xml version=\"1.0\"?>" << std::endl
      << "<!-- " << DihuContext::versionText() << " "
      << DihuContext::metaText()
      << ", currentTime: " << this->currentTime_
      << ", timeStepNo: " << this->timeStepNo_ << " -->" << std::endl
      << "<VTKFile type=\"PolyData\" version=\"1.0\" "
         "byte_order=\"LittleEndian\">"
      << std::endl
      << std::string(1, '\t') << "<PolyData>" << std::endl
      << std::string(2, '\t') << "<FieldData>" << std::endl
      << std::string(3, '\t')
      << "<DataArray type=\"Float32\" Name=\"Time\" "
         "NumberOfTuples=\"1\" format=\""
      << (binaryOutput_ ? "binary" : "ascii") << "\" >" << std::endl
      << std::string(4, '\t') << stringTime << std::endl
      << std::string(3, '\t') << "</DataArray>" << std::endl
      << std::string(2, '\t') << "</FieldData>" << std::endl;

  outputFileParts[outputFilePartNo]
      << std::string(2, '\t')
      << "<Piece NumberOfPoints=\"" << nPointsGlobal1D_
      << "\" NumberOfVerts=\"0\" "
      << "NumberOfLines=\"" << nLinesGlobal1D_
      << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">"
      << std::endl
      << std::string(3, '\t') << "<PointData";

  if (!vtkPiece1D_.firstScalarName.empty())
  {
    outputFileParts[outputFilePartNo]
        << " Scalars=\"" << vtkPiece1D_.firstScalarName << "\"";
  }

  if (!vtkPiece1D_.firstVectorName.empty())
  {
    outputFileParts[outputFilePartNo]
        << " Vectors=\"" << vtkPiece1D_.firstVectorName << "\"";
  }

  outputFileParts[outputFilePartNo] << ">" << std::endl;

  // PointData arrays.
  for (std::vector<PolyDataPropertiesForMesh::DataArrayName>::iterator
           pointDataArrayIter =
               vtkPiece1D_.properties.pointDataArrays.begin();
       pointDataArrayIter !=
       vtkPiece1D_.properties.pointDataArrays.end();
       ++pointDataArrayIter)
  {
    std::stringstream componentNames;
    bool isComponentNamesSet = false;

    for (int componentNo = 0;
         componentNo < pointDataArrayIter->nComponents;
         ++componentNo)
    {
      const std::string componentName =
          pointDataArrayIter->componentNames[componentNo];

      componentNames
          << "ComponentName" << componentNo << "=\""
          << componentName << "\" ";

      std::stringstream trivialComponentName;
      trivialComponentName << componentNo;

      if (componentName != trivialComponentName.str())
        isComponentNamesSet = true;
    }

    if (!isComponentNamesSet)
      componentNames.str("");

    outputFileParts[outputFilePartNo]
        << std::string(4, '\t') << "<DataArray "
        << "Name=\"" << pointDataArrayIter->name << "\" "
        << "type=\""
        << (pointDataArrayIter->name == "partitioning"
                ? "Int32"
                : "Float32")
        << "\" "
        << "NumberOfComponents=\""
        << pointDataArrayIter->nComponents << "\" "
        << componentNames.str()
        << "format=\"" << (binaryOutput_ ? "binary" : "ascii")
        << "\" >" << std::endl
        << std::string(5, '\t');

    // Field-variable data are inserted here.
    ++outputFilePartNo;

    outputFileParts[outputFilePartNo]
        << std::endl
        << std::string(4, '\t') << "</DataArray>" << std::endl;
  }

  outputFileParts[outputFilePartNo]
      << std::string(3, '\t') << "</PointData>" << std::endl
      << std::string(3, '\t') << "<CellData>" << std::endl
      << std::string(3, '\t') << "</CellData>" << std::endl
      << std::string(3, '\t') << "<Points>" << std::endl
      << std::string(4, '\t')
      << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" "
         "format=\""
      << (binaryOutput_ ? "binary" : "ascii")
      << "\" >" << std::endl
      << std::string(5, '\t');

  // Geometry data are inserted here.
  ++outputFilePartNo;

  outputFileParts[outputFilePartNo]
      << std::endl
      << std::string(4, '\t') << "</DataArray>" << std::endl
      << std::string(3, '\t') << "</Points>" << std::endl
      << std::string(3, '\t') << "<Verts></Verts>" << std::endl
      << std::string(3, '\t') << "<Lines>" << std::endl
      << std::string(4, '\t')
      << "<DataArray Name=\"connectivity\" type=\"Int32\" "
      << (binaryOutput_ ? "format=\"binary\"" : "format=\"ascii\"")
      << ">" << std::endl
      << std::string(5, '\t');

  // Connectivity data are inserted here.
  ++outputFilePartNo;

  outputFileParts[outputFilePartNo]
      << std::endl
      << std::string(4, '\t') << "</DataArray>" << std::endl
      << std::string(4, '\t')
      << "<DataArray Name=\"offsets\" type=\"Int32\" "
      << (binaryOutput_ ? "format=\"binary\"" : "format=\"ascii\"")
      << ">" << std::endl
      << std::string(5, '\t');

  // Offset data are inserted here.
  ++outputFilePartNo;

  outputFileParts[outputFilePartNo]
      << std::endl
      << std::string(4, '\t') << "</DataArray>" << std::endl
      << std::string(3, '\t') << "</Lines>" << std::endl
      << std::string(3, '\t') << "<Strips></Strips>" << std::endl
      << std::string(3, '\t') << "<Polys></Polys>" << std::endl
      << std::string(2, '\t') << "</Piece>" << std::endl
      << std::string(1, '\t') << "</PolyData>" << std::endl
      << "</VTKFile>" << std::endl;

  assert(outputFilePartNo + 1 == nOutputFileParts);

  VLOG(1) << "outputFileParts:";

  for (std::vector<std::stringstream>::iterator iter =
           outputFileParts.begin();
       iter != outputFileParts.end();
       ++iter)
  {
    VLOG(1) << " " << iter->str();
  }

  LOG(DEBUG) << "open MPI file \"" << filenameStr << "\".";

  MPI_File fileHandle;

  MPIUtility::handleReturnValue(
      MPI_File_open(
          this->rankSubset_->mpiCommunicator(),
          filenameStr.c_str(),
          MPI_MODE_WRONLY | MPI_MODE_CREATE,
          MPI_INFO_NULL,
          &fileHandle),
      "MPI_File_open");

  Control::PerformanceMeasurement::start("durationParaview1DWrite");

  // Write the beginning of the file on rank 0.
  outputFilePartNo = 0;

  writeAsciiDataShared(
      fileHandle,
      ownRankNo,
      outputFileParts[outputFilePartNo].str());

  ++outputFilePartNo;

  VLOG(1) << "get current shared file position";

  MPI_Offset currentFilePosition = 0;

  MPIUtility::handleReturnValue(
      MPI_File_get_position_shared(
          fileHandle,
          &currentFilePosition),
      "MPI_File_get_position_shared");

  LOG(DEBUG)
      << "current shared file position: "
      << currentFilePosition;

  // Write PointData arrays.
  int fieldVariableNo = 0;

  for (std::vector<PolyDataPropertiesForMesh::DataArrayName>::iterator
           pointDataArrayIter =
               vtkPiece1D_.properties.pointDataArrays.begin();
       pointDataArrayIter !=
       vtkPiece1D_.properties.pointDataArrays.end();
       ++pointDataArrayIter, ++fieldVariableNo)
  {
    assert(
        fieldVariableValues.find(pointDataArrayIter->name) !=
        fieldVariableValues.end());

    const bool writeFloatsAsInt =
        pointDataArrayIter->name == "partitioning";

    writeCombinedValuesVector(
        fileHandle,
        ownRankNo,
        fieldVariableValues[pointDataArrayIter->name],
        fieldVariableNo,
        writeFloatsAsInt);

    writeAsciiDataShared(
        fileHandle,
        ownRankNo,
        outputFileParts[outputFilePartNo].str());

    ++outputFilePartNo;
  }

  // Write geometry.
  writeCombinedValuesVector(
      fileHandle,
      ownRankNo,
      geometryFieldValues,
      fieldVariableNo++);

  writeAsciiDataShared(
      fileHandle,
      ownRankNo,
      outputFileParts[outputFilePartNo].str());

  ++outputFilePartNo;

  // Write polyline connectivity.
  writeCombinedValuesVector(
      fileHandle,
      ownRankNo,
      connectivityValues,
      fieldVariableNo++);

  writeAsciiDataShared(
      fileHandle,
      ownRankNo,
      outputFileParts[outputFilePartNo].str());

  ++outputFilePartNo;

  // Write polyline offsets.
  writeCombinedValuesVector(
      fileHandle,
      ownRankNo,
      offsetValues,
      fieldVariableNo++);

  writeAsciiDataShared(
      fileHandle,
      ownRankNo,
      outputFileParts[outputFilePartNo].str());

  Control::PerformanceMeasurement::stop("durationParaview1DWrite");

  MPIUtility::handleReturnValue(
      MPI_File_close(&fileHandle),
      "MPI_File_close");

  // Register file in the *.vtk.series JSON file.
  if (ownRankNo == 0)
  {
    Paraview::seriesWriter().registerNewFile(
        filenameStr,
        this->currentTime_);
  }
}

} // namespace OutputWriter
