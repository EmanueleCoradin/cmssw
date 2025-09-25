#include "EventFilter/Utilities/interface/DAQSourceModelsScoutingPhase2.h"
#include <iostream>
using namespace edm::streamer;

void DataModeScoutingPhase2::makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                                                  std::vector<int> const& numSources,
                                                  std::vector<int> const& sourceIDs,
                                                  std::string const& sourceIdentifier,
                                                  std::string const& runDir) {
  std::cout << "makeDirectoryEntries called with runDir " << runDir << ",\n baseDirs = {\n";
  for (auto& b : baseDirs)
    std::cout << "   " << b << ",\n";
  std::cout << "}\n";
  std::filesystem::path runDirP(runDir);
  for (auto& baseDir : baseDirs) {
    std::filesystem::path baseDirP(baseDir);
    buPaths_.emplace_back(baseDirP / runDirP);
  }
  std::cout << ", numSources = { \n";
  for (auto& n : numSources)
    std::cout << "   " << n << ",\n";
  std::cout << "}\n";
  std::cout << std::endl;

  // store the number of sources in each BU
  buNumSources_ = numSources;
  totalNumSources_ = std::accumulate(numSources.begin(), numSources.begin() + baseDirs.size(), 0u);
}

std::pair<bool, std::vector<std::string>> DataModeScoutingPhase2::defineAdditionalFiles(std::string const& primaryName,
                                                                                        bool fileListMode) const {
  std::vector<std::string> additionalFiles;
  assert(!buNumSources_.empty());
  auto fullpath = std::filesystem::path(primaryName);
  auto fullstr = fullpath.filename().generic_string();
  auto pos = fullstr.rfind("_stream00");
  assert(pos != std::string::npos);
  std::string pre = fullstr.substr(0, pos + 7), post = fullstr.substr(pos + 9);
  char buff[3];
  unsigned int istream = 0;
  //std::string files = primaryName;
  for (unsigned int i = 0, n = buPaths_.size(); i < n; ++i) {
    for (unsigned int j = 0, nj = buNumSources_[i]; j < nj; ++j, ++istream) {
      if (istream == 0)
        continue;  // this is the main file
      snprintf(buff, sizeof(buff), "%02u", istream);
      auto path = buPaths_[i] / (pre + std::string(buff) + post);
      additionalFiles.push_back(path.generic_string());
      //files += ", " + path.generic_string();
    }
  }
  assert(istream == totalNumSources_);
  //std::cout << "Will open " << files << std::endl;
  return std::make_pair(true, additionalFiles);
}

void DataModeScoutingPhase2::readEvent(edm::EventPrincipal& eventPrincipal) {
  assert(!events_.empty());

  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  // set provenance helpers
  uint32_t hdrEventID = currOrbit_;
  edm::EventID eventID = edm::EventID(daqSource_->eventRunNumber(), daqSource_->currentLumiSection(), hdrEventID);
  edm::EventAuxiliary aux(
      eventID, daqSource_->processGUID(), tstamp, events_[0]->isRealData(), edm::EventAuxiliary::PhysicsTrigger);

  aux.setProcessHistoryID(daqSource_->processHistoryID());
  daqSource_->makeEventWrapper(eventPrincipal, aux);

  // create scouting raw data collection
  //std::cout << "Called here with " << events_.size() << " events" << std::endl;
  std::unique_ptr<SDSRawDataCollection> rawData(new SDSRawDataCollection);
  for (unsigned int id = 0, n = events_.size(); id < n; ++id) {
    const auto& e = *events_[id];
    auto size = e.eventSize();
    //std::cout << "Event " << id << " of size " << size << " payload at " << (void*)(e.payload()) << std::endl;
    auto& fedData = rawData->FEDData(id);
    fedData.resize(size);
    memcpy(fedData.data(), e.payload(), size);
  }

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<SDSRawDataCollection>(std::move(rawData)));
  eventPrincipal.put(
      daqProvenanceHelpers_[0]->productDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());

  eventCached_ = false;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeScoutingPhase2::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(
      std::make_shared<const edm::DaqProvenanceHelper>(edm::TypeID(typeid(SDSRawDataCollection)), "DAQSource"));
  return daqProvenanceHelpers_;
}

bool DataModeScoutingPhase2::nextEventView(RawInputFile*) {
  blockCompleted_ = false;
  if (eventCached_)
    return true;

  // move the data block address only for the sources processed
  // un the previous event by adding the last event size
  for (const auto& pair : sourceValidOrbitPair_) {
    dataBlockAddrs_[pair.first] += events_[pair.second]->size();
  }

  return makeEvents();
}

bool DataModeScoutingPhase2::makeEvents() {
  // clear events and reset current orbit
  events_.clear();
  sourceValidOrbitPair_.clear();
  currOrbit_ = 0xFFFFFFFF;  // max uint
  assert(!blockCompleted_);

  // create current "events" (= orbits) list from each data source,
  // check if one dataBlock terminated earlier than others.
  for (int i = 0; i < numFiles_; i++) {
    /*
    std::cout << " i = " << i << ", dataBlockAddrs_[i] = " << (void*)(dataBlockAddrs_[i])
              << ", dataBlockMaxAddrs_[i] = " << (void*)(dataBlockMaxAddrs_[i])
              << ", blockCompleted_ = " << blockCompleted_ << std::endl;
    */
    if (dataBlockAddrs_[i] >= dataBlockMaxAddrs_[i]) {
      completedBlocks_[i] = true;
      continue;
    }

    // event contains data, add it to the events list
    events_.emplace_back(std::make_unique<FRDEventMsgView>(dataBlockAddrs_[i]));
    /*
    std::cout << "Emplaced event " << i << " (check: " << (events_.size() - 1) << ") of size " << events_[i]->size()
              << " eventSize " << events_[i]->eventSize() << ", payload at " << (void*)(events_[i]->payload())
              << ", event id:" << events_[i]->event() << std::endl;
    */
    if (dataBlockAddrs_[i] + events_.back()->size() > dataBlockMaxAddrs_[i])
      throw cms::Exception("DAQSource::getNextEvent")
          << " event id:" << events_.back()->event() << " lumi:" << events_.back()->lumi()
          << " run:" << events_.back()->run() << " of size:" << events_.back()->size()
          << " bytes does not fit into the buffer or has corrupted header";

    // find the minimum orbit for the current event between all files
    if ((events_.back()->event() < currOrbit_) && (!completedBlocks_[i])) {
      currOrbit_ = events_.back()->event();
    }
  }

  // mark valid orbits from each data source
  // e.g. find when orbit is missing from one source
  bool allBlocksCompleted = true;
  int evt_idx = 0;
  for (int i = 0; i < numFiles_; i++) {
    if (completedBlocks_[i]) {
      continue;
    }

    if (events_[evt_idx]->event() != currOrbit_) {
      // current source (=i-th source) doesn't contain the expected orbit.
      // skip it, and move to the next orbit
    } else {
      // add a pair <current surce index, event index>
      // evt_idx can be different from variable i, as some data blocks can be
      // completed before others
      sourceValidOrbitPair_.emplace_back(std::make_pair(i, evt_idx));
      allBlocksCompleted = false;
    }

    evt_idx++;
  }

  if (allBlocksCompleted) {
    blockCompleted_ = true;
  }
  return !allBlocksCompleted;
}

bool DataModeScoutingPhase2::checksumValid() { return true; }

std::string DataModeScoutingPhase2::getChecksumError() const { return std::string(); }
