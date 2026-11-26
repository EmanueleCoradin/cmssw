#include <algorithm>

#include "HeavyIonsAnalysis/MuonAnalysis/interface/MuonAnalyzer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

using namespace std::placeholders;
using namespace std;
using namespace reco;
using namespace edm;
using namespace HepMC;

MuonAnalyzer::MuonAnalyzer(const edm::ParameterSet& ps) {
  doGen_ = ps.getParameter<bool>("doGen");
  doReco_ = ps.getUntrackedParameter<bool>("doReco");

  const auto outputBranches = ps.getUntrackedParameter<std::vector<std::string>>("outputBranches", {"*"});
  const bool saveAll = std::find(outputBranches.begin(), outputBranches.end(), "*") != outputBranches.end();

  // Lambda function to select the output branches
  auto bookBranch = [this, &outputBranches, saveAll](const std::string& name, auto* address) {
    if (saveAll || std::find(outputBranches.begin(), outputBranches.end(), name) != outputBranches.end()) {
      tree_->Branch(name.c_str(), address);
    }
  };

  vertexToken_ = consumes<std::vector<reco::Vertex>>(ps.getParameter<edm::InputTag>("vertexSrc"));
  genToken_ = consumes<edm::View<pat::PackedGenParticle>>(ps.getParameter<edm::InputTag>("genparticle"));
  muonToken_ = consumes<edm::View<pat::Muon>>(ps.getParameter<edm::InputTag>("muonSrc"));
  trackBuilderToken_ = esConsumes(edm::ESInputTag("", "TransientTrackBuilder"));

  // initialize output TTree
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("MuonTree", "muon tree");

  bookBranch("run", &run_);
  bookBranch("event", &event_);
  bookBranch("lumi", &lumi_);

  bookBranch("nGen", &nGen_);
  bookBranch("genVtx_x", &genVtx_x_);
  bookBranch("genVtx_y", &genVtx_y_);
  bookBranch("genVtx_z", &genVtx_z_);
  bookBranch("genPID", &genPID_);
  bookBranch("genStatus", &genStatus_);
  bookBranch("genP", &genP_);
  bookBranch("genPt", &genPt_);
  bookBranch("genEta", &genEta_);
  bookBranch("genPhi", &genPhi_);
  bookBranch("genMotherID", &genMotherID_);

  // reco muon candidates
  bookBranch("nReco", &nReco_);
  bookBranch("recoP", &recoP_);
  bookBranch("recoPt", &recoPt_);
  bookBranch("recoEta", &recoEta_);
  bookBranch("recoPhi", &recoPhi_);
  bookBranch("recoL1Eta", &recoL1Eta_);
  bookBranch("recoL1Phi", &recoL1Phi_);
  bookBranch("recoCharge", &recoCharge_);
  bookBranch("recoType", &recoType_);
  bookBranch("recoIsGood", &recoIsGood_);
  bookBranch("recoIsGlobal", &recoIsGlobal_);
  bookBranch("recoIsTracker", &recoIsTracker_);
  bookBranch("recoIsPF", &recoIsPF_);
  bookBranch("recoIsSTA", &recoIsSTA_);
  bookBranch("recoDxy", &recoDxy_);
  bookBranch("recoDz", &recoDz_);
  bookBranch("recoDxyErr", &recoDxyErr_);
  bookBranch("recoDzErr", &recoDzErr_);
  bookBranch("recoIP3D", &recoIP3D_);
  bookBranch("recoIP3DErr", &recoIP3DErr_);
  bookBranch("recoNMatchedStations", &recoNMatchedStations_);
  bookBranch("recoIsoTrk", &recoIsoTrk_);
  bookBranch("recoPFChIso", &recoPFChIso_);
  bookBranch("recoPFPhoIso", &recoPFPhoIso_);
  bookBranch("recoPFNeuIso", &recoPFNeuIso_);
  bookBranch("recoPFPUIso", &recoPFPUIso_);
  bookBranch("recoMVAIso", &recoMVAIso_);
  for (auto& w : recoMVAIsoWP_)
    bookBranch(("recoMVAIso"+w.first).c_str(), &(w.second));
  bookBranch("recoIDHybridSoft", &recoIDHybridSoft_);
  bookBranch("recoIDSoft", &recoIDSoft_);
  bookBranch("recoIDLoose", &recoIDLoose_);
  bookBranch("recoIDMedium", &recoIDMedium_);
  bookBranch("recoIDMediumPrompt", &recoIDMediumPrompt_);
  bookBranch("recoIDTight", &recoIDTight_);
  bookBranch("recoIDGlobalHighPt", &recoIDGlobalHighPt_);
  bookBranch("recoIDTrkHighPt", &recoIDTrkHighPt_);
  bookBranch("recoIDInTime", &recoIDInTime_);
  bookBranch("recoMVAIDSoft", &recoMVAIDSoft_);
  bookBranch("recoMVAIDLoose", &recoMVAIDLoose_);
  bookBranch("recoMVAIDMedium", &recoMVAIDMedium_);
  bookBranch("recoMVAIDTight", &recoMVAIDTight_);
  bookBranch("recoMVAIDLooseLowPt", &recoMVAIDLooseLowPt_);
  bookBranch("recoMVAIDMediumLowPt", &recoMVAIDMediumLowPt_);

  // inner tracks
  bookBranch("nInner", &nInner_);
  bookBranch("innerDxy", &innerDxy_);
  bookBranch("innerDz", &innerDz_);
  bookBranch("innerDxyErr", &innerDxyErr_);
  bookBranch("innerDzErr", &innerDzErr_);
  bookBranch("innerP", &innerP_);
  bookBranch("innerPt", &innerPt_);
  bookBranch("innerPtErr", &innerPtErr_);
  bookBranch("innerEta", &innerEta_);
  bookBranch("innerTrkLayers", &innerTrkLayers_);
  bookBranch("innerNTrkHits", &innerNTrkHits_);
  bookBranch("innerPixelLayers", &innerPixelLayers_);
  bookBranch("innerNPixelHits", &innerNPixelHits_);
  bookBranch("innerIsHighPurityTrack", &innerIsHighPurityTrack_);
  bookBranch("innerNormChi2", &innerNormChi2_);

  // global muons
  bookBranch("nGlobal", &nGlobal_);
  bookBranch("globalP", &globalP_);
  bookBranch("globalPt", &globalPt_);
  bookBranch("globalPtErr", &globalPtErr_);
  bookBranch("globalEta", &globalEta_);
  bookBranch("globalIsArbitrated", &globalIsArbitrated_);
  bookBranch("globalDxy", &globalDxy_);
  bookBranch("globalDz", &globalDz_);
  bookBranch("globalDxyErr", &globalDxyErr_);
  bookBranch("globalDzErr", &globalDzErr_);
  bookBranch("globalNormChi2", &globalNormChi2_);
  bookBranch("globalNMuonHits", &globalNMuonHits_);
}

MuonAnalyzer::~MuonAnalyzer() {}

void MuonAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& es) {
  // cleanup from previous event

  if (doGen_) {
    nGen_ = 0;
    genVtx_x_.clear();
    genVtx_y_.clear();
    genVtx_z_.clear();
    genPID_.clear();
    genStatus_.clear();
    genP_.clear();
    genPt_.clear();
    genEta_.clear();
    genPhi_.clear();
    genMotherID_.clear();
  }

  nReco_ = 0;
  recoP_.clear();
  recoPt_.clear();
  recoEta_.clear();
  recoPhi_.clear();
  recoL1Eta_.clear();
  recoL1Phi_.clear();
  recoCharge_.clear();
  recoType_.clear();
  recoIsGood_.clear();
  recoIsGlobal_.clear();
  recoIsTracker_.clear();
  recoIsPF_.clear();
  recoIsSTA_.clear();
  recoDxy_.clear();
  recoDz_.clear();
  recoDxyErr_.clear();
  recoDzErr_.clear();
  recoIP3D_.clear();
  recoIP3DErr_.clear();
  recoNMatchedStations_.clear();
  recoIsoTrk_.clear();
  recoPFChIso_.clear();
  recoPFPhoIso_.clear();
  recoPFNeuIso_.clear();
  recoPFPUIso_.clear();
  recoMVAIso_.clear();
  for (auto& w : recoMVAIsoWP_)
    w.second.clear();
  recoIDHybridSoft_.clear();
  recoIDSoft_.clear();
  recoIDLoose_.clear();
  recoIDMedium_.clear();
  recoIDMediumPrompt_.clear();
  recoIDTight_.clear();
  recoIDGlobalHighPt_.clear();
  recoIDTrkHighPt_.clear();
  recoIDInTime_.clear();

  recoMVAIDSoft_.clear();
  recoMVAIDLoose_.clear();
  recoMVAIDMedium_.clear();
  recoMVAIDTight_.clear();
  recoMVAIDLooseLowPt_.clear();
  recoMVAIDMediumLowPt_.clear();

  nInner_ = 0;
  innerDxy_.clear();
  innerDz_.clear();
  innerDxyErr_.clear();
  innerDzErr_.clear();
  innerP_.clear();
  innerPt_.clear();
  innerPtErr_.clear();
  innerEta_.clear();
  innerTrkLayers_.clear();
  innerNTrkHits_.clear();
  innerPixelLayers_.clear();
  innerNPixelHits_.clear();
  innerIsHighPurityTrack_.clear();
  innerNormChi2_.clear();

  nGlobal_ = 0;
  globalP_.clear();
  globalPt_.clear();
  globalPtErr_.clear();
  globalEta_.clear();
  globalIsArbitrated_.clear();
  globalDxy_.clear();
  globalDz_.clear();
  globalDxyErr_.clear();
  globalDzErr_.clear();
  globalNormChi2_.clear();
  globalNMuonHits_.clear();

  run_ = e.id().run();
  event_ = e.id().event();
  lumi_ = e.luminosityBlock();

  if (doGen_) {
    edm::Handle<edm::View<pat::PackedGenParticle>> genCollection;
    e.getByToken(genToken_, genCollection);

    if (genCollection.isValid()) {
      for (unsigned i = 0; i < genCollection->size(); ++i) {
        pat::PackedGenParticle const& genParticle = genCollection->at(i);
        if (!(abs(genParticle.pdgId()) == 13 && genParticle.status() == 1))
          continue;  // only muons

        nGen_++;

        genVtx_x_.push_back(genParticle.vx());
        genVtx_y_.push_back(genParticle.vy());
        genVtx_z_.push_back(genParticle.vz());

        genPID_.push_back(genParticle.pdgId());
        genStatus_.push_back(genParticle.status());
        genP_.push_back(genParticle.p());
        genPt_.push_back(genParticle.pt());
        genEta_.push_back(genParticle.eta());
        genPhi_.push_back(genParticle.phi());

        // search for the mother ID (if any)
        int motherID = genParticle.pdgId();  // initialize with the muon ID in case of single particle gun

        if (genParticle.numberOfMothers() > 0) {
          bool isQuarkoniumMother = false;

          for (unsigned int mom = 0; mom < genParticle.numberOfMothers(); mom++) {
            int momID = genParticle.mother(mom)->pdgId();
            if (momID == 443 || momID == 100443 || momID == 553 || momID == 100553 || momID == 200553) {
              isQuarkoniumMother = true;
              motherID = momID;
              break;  // mother found!
            }
          }

          // if the mother is not a quarkonium state, check the grandmother
          if (!isQuarkoniumMother) {
            const Candidate* motherCand = genParticle.mother(0);

            for (unsigned int gmom = 0; gmom < motherCand->numberOfMothers(); gmom++) {
              int gmomID = motherCand->mother(gmom)->pdgId();
              if (gmomID == 443 || gmomID == 100443 || gmomID == 553 || gmomID == 100553 || gmomID == 200553) {
                motherID = gmomID;
                break;  // mother found!
              }
            }
          }
        }

        genMotherID_.push_back(motherID);
      }  // end of gen particle loop
    }
  }  // end of doGen_

  if (doReco_) {
    edm::Handle<std::vector<reco::Vertex>> vertices;
    e.getByToken(vertexToken_, vertices);

    // best-known primary vertex coordinates
    reco::Vertex pv(math::XYZPoint(0, 0, -999), math::Error<3>::type());
    for (const auto& vertex : *vertices) {
      if (!vertex.isFake()) {
        pv = vertex;
        break;
      }
    }

    const auto& tb = es.getHandle(trackBuilderToken_);

    // fill tree branches with reconstructed muons.
    edm::Handle<edm::View<pat::Muon>> recoMuons;
    e.getByToken(muonToken_, recoMuons);

    for (const auto& mu : *recoMuons) {
      if (!(mu.isPFMuon() || mu.isGlobalMuon() || mu.isTrackerMuon()))
        continue;

      nReco_++;

      recoP_.push_back(mu.p());
      recoPt_.push_back(mu.pt());
      recoEta_.push_back(mu.eta());
      recoPhi_.push_back(mu.phi());
      recoL1Eta_.push_back(mu.hasUserFloat("l1Eta") ? mu.userFloat("l1Eta") : -99);
      recoL1Phi_.push_back(mu.hasUserFloat("l1Phi") ? mu.userFloat("l1Phi") : -99);
      recoCharge_.push_back(mu.charge());
      recoType_.push_back(mu.type());
      recoIsGood_.push_back(muon::isGoodMuon(mu, muon::selectionTypeFromString("TMOneStationTight")));

      recoIsGlobal_.push_back(mu.isGlobalMuon());
      recoIsTracker_.push_back(mu.isTrackerMuon());
      recoIsPF_.push_back(mu.isPFMuon());
      recoIsSTA_.push_back(mu.isStandAloneMuon());

      recoDxy_.push_back(mu.muonBestTrack()->dxy(pv.position()));
      recoDz_.push_back(mu.muonBestTrack()->dz(pv.position()));
      recoDxyErr_.push_back(mu.muonBestTrack()->dxyError());
      recoDzErr_.push_back(mu.muonBestTrack()->dzError());

      // initialize with unphysical values
      float recoIP3D = -999;
      float recoIP3DErr = -999;
      if (pv.isValid()) {
        // 3D impact parameter
        reco::TransientTrack tt = tb->build(mu.muonBestTrack().get());
        recoIP3D = IPTools::absoluteImpactParameter3D(tt, pv).second.value();
        recoIP3DErr = IPTools::absoluteImpactParameter3D(tt, pv).second.error();
      }
      recoIP3D_.push_back(recoIP3D);
      recoIP3DErr_.push_back(recoIP3DErr);

      // inner track info
      if (mu.innerTrack().isNonnull()) {
        const reco::TrackRef innMu = mu.innerTrack();

        const reco::HitPattern& hitPat = innMu->hitPattern();

        nInner_++;

        innerDxy_.push_back(innMu->dxy(pv.position()));
        innerDz_.push_back(innMu->dz(pv.position()));
        innerDxyErr_.push_back(innMu->dxyError());
        innerDzErr_.push_back(innMu->dzError());

        innerP_.push_back(innMu->p());
        innerPt_.push_back(innMu->pt());
        innerPtErr_.push_back(innMu->ptError());
        innerEta_.push_back(innMu->eta());

        innerTrkLayers_.push_back(hitPat.trackerLayersWithMeasurement());
        innerNTrkHits_.push_back(hitPat.numberOfValidTrackerHits());

        innerPixelLayers_.push_back(hitPat.pixelLayersWithMeasurement());
        innerNPixelHits_.push_back(hitPat.numberOfValidPixelHits());

        innerIsHighPurityTrack_.push_back(innMu->quality(reco::TrackBase::highPurity));

        innerNormChi2_.push_back(innMu->normalizedChi2());

      } else {
        innerDxy_.push_back(-99);
        innerDz_.push_back(-99);
        innerDxyErr_.push_back(-99);
        innerDzErr_.push_back(-99);

        innerP_.push_back(-99);
        innerPt_.push_back(-99);
        innerPtErr_.push_back(-99);
        innerEta_.push_back(-99);

        innerTrkLayers_.push_back(-99);
        innerNTrkHits_.push_back(-99);
        innerPixelLayers_.push_back(-99);
        innerNPixelHits_.push_back(-99);

        innerIsHighPurityTrack_.push_back(false);

        innerNormChi2_.push_back(-99);
      }

      // global muons, avoiding overlaps with inner track variables if possible
      if (mu.globalTrack().isNonnull() && mu.isGlobalMuon()) {
          const reco::TrackRef glbMu = mu.globalTrack();

          nGlobal_++;

          globalP_.push_back(glbMu->p());
          globalPt_.push_back(glbMu->pt());
          globalPtErr_.push_back(glbMu->ptError());
          globalEta_.push_back(glbMu->eta());

          globalIsArbitrated_.push_back(muon::isGoodMuon(mu, muon::selectionTypeFromString("TrackerMuonArbitrated")));

          globalDxy_.push_back(glbMu->dxy(pv.position()));
          globalDz_.push_back(glbMu->dz(pv.position()));
          globalDxyErr_.push_back(glbMu->dxyError());
          globalDzErr_.push_back(glbMu->dzError());

          globalNormChi2_.push_back(glbMu->normalizedChi2());
          globalNMuonHits_.push_back(glbMu->hitPattern().numberOfValidMuonHits());

        } else {
          globalP_.push_back(-99);
          globalPt_.push_back(-99);
          globalPtErr_.push_back(-99);
          globalEta_.push_back(-99);

          globalIsArbitrated_.push_back(false);

          globalDxy_.push_back(-99);
          globalDz_.push_back(-99);
          globalDxyErr_.push_back(-99);
          globalDzErr_.push_back(-99);

          globalNormChi2_.push_back(-99);
          globalNMuonHits_.push_back(-99);
        }

      recoNMatchedStations_.push_back(mu.numberOfMatchedStations());
      recoIsoTrk_.push_back(mu.isolationR03().sumPt);
      recoPFChIso_.push_back(mu.pfIsolationR04().sumChargedHadronPt);
      recoPFPhoIso_.push_back(mu.pfIsolationR04().sumPhotonEt);
      recoPFNeuIso_.push_back(mu.pfIsolationR04().sumNeutralHadronEt);
      recoPFPUIso_.push_back(mu.pfIsolationR04().sumPUPt);

      recoMVAIso_.push_back(mu.hasUserFloat("hiMVAIso") ? mu.userFloat("hiMVAIso") : -99);
      for (auto& w : recoMVAIsoWP_)
        w.second.push_back(mu.hasUserInt("hiMVAIso"+w.first) && mu.userInt("hiMVAIso"+w.first)>0);

      recoIDHybridSoft_.push_back(mu.isGlobalMuon() && mu.isTrackerMuon() && mu.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 && mu.innerTrack()->hitPattern().pixelLayersWithMeasurement() > 0 && fabs(mu.innerTrack()->dxy(pv.position()) < 0.3) && fabs(mu.innerTrack()->dz(pv.position()) < 20.));

      //  muon selectors available at https://github.com/cms-sw/cmssw/blob/4c9240b33ace61c92c6803f0c4eace9ba06e6c8d/DataFormats/MuonReco/interface/Muon.h#L202
      // Cut-based Ids
      recoIDSoft_.push_back(mu.passed(reco::Muon::SoftCutBasedId));
      recoIDLoose_.push_back(mu.passed(reco::Muon::CutBasedIdLoose));
      recoIDMedium_.push_back(mu.passed(reco::Muon::CutBasedIdMedium));
      recoIDMediumPrompt_.push_back(mu.passed(reco::Muon::CutBasedIdMediumPrompt));
      recoIDTight_.push_back(mu.passed(reco::Muon::CutBasedIdTight));
      recoIDGlobalHighPt_.push_back(mu.passed(reco::Muon::CutBasedIdGlobalHighPt));
      recoIDTrkHighPt_.push_back(mu.passed(reco::Muon::CutBasedIdTrkHighPt));
      recoIDInTime_.push_back(mu.passed(reco::Muon::InTimeMuon));

      // MVA-based Ids
      recoMVAIDSoft_.push_back(mu.passed(reco::Muon::SoftMvaId));
      recoMVAIDLoose_.push_back(mu.passed(reco::Muon::MvaLoose));
      recoMVAIDMedium_.push_back(mu.passed(reco::Muon::MvaMedium));
      recoMVAIDTight_.push_back(mu.passed(reco::Muon::MvaTight));
      recoMVAIDLooseLowPt_.push_back(mu.passed(reco::Muon::LowPtMvaLoose));
      recoMVAIDMediumLowPt_.push_back(mu.passed(reco::Muon::LowPtMvaMedium));

    }  // muons loop
  }    // end of doReco_

  tree_->Fill();
}

DEFINE_FWK_MODULE(MuonAnalyzer);
