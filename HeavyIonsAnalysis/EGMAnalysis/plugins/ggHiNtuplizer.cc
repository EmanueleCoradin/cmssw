#include <algorithm>

#include "HeavyIonsAnalysis/EGMAnalysis/plugins/ggHiNtuplizer.h"
#include "HeavyIonsAnalysis/EGMAnalysis/plugins/GenParticleParentage.h"
#include "HeavyIonsAnalysis/EGMAnalysis/interface/pfIsoCalculator.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
//#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "CommonTools/Egamma/interface/EffectiveAreas.h"

using namespace std::placeholders;

ggHiNtuplizer::ggHiNtuplizer(const edm::ParameterSet& ps)
    : effectiveAreas_((ps.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath()) {

  const auto outputBranches = ps.getUntrackedParameter<std::vector<std::string>>("outputBranches", {"*"});
  const bool saveAll = std::find(outputBranches.begin(), outputBranches.end(), "*") != outputBranches.end();
  
  // Lambda function to select the output branches
  auto bookBranch = [this, &outputBranches, saveAll](const std::string& name, auto* address) {
    if (saveAll || std::find(outputBranches.begin(), outputBranches.end(), name) != outputBranches.end()) {
      tree_->Branch(name.c_str(), address);
    }
  };

  doGenParticles_ = ps.getParameter<bool>("doGenParticles");
  doElectrons_ = ps.getParameter<bool>("doElectrons");
  doPhotons_ = ps.getParameter<bool>("doPhotons");
  doMuons_ = ps.getParameter<bool>("doMuons");

  doEffectiveAreas_ = ps.getParameter<bool>("doEffectiveAreas");

  isParticleGun_ = ps.getParameter<bool>("isParticleGun");
  useValMapIso_ = ps.getParameter<bool>("useValMapIso");
  doPhoEReg_ = ps.getParameter<bool>("doPhoERegression");
  doRecHitsEB_ = ps.getParameter<bool>("doRecHitsEB");
  doRecHitsEE_ = ps.getParameter<bool>("doRecHitsEE");

  if (doRecHitsEB_) {
    recHitsEB_ = consumes<EcalRecHitCollection>(
        ps.getUntrackedParameter<edm::InputTag>("recHitsEB", edm::InputTag("ecalRecHit", "EcalRecHitsEB")));
  }
  if (doRecHitsEE_) {
    recHitsEE_ = consumes<EcalRecHitCollection>(
        ps.getUntrackedParameter<edm::InputTag>("recHitsEE", edm::InputTag("ecalRecHit", "EcalRecHitsEE")));
  }

  doSuperClusters_ = ps.getParameter<bool>("doSuperClusters");
  if (doSuperClusters_) {
    scToken_ = consumes<std::vector<reco::SuperCluster>>(ps.getParameter<edm::InputTag>("superClusters"));
  }

  doPfIso_ = ps.getParameter<bool>("doPfIso");
  if (doPfIso_) {
    isPackedPFCandidate_ = ps.getParameter<bool>("isPackedPFCandidate");

    std::string strPFCandSrc = "particleFlowCollection";
    if (isPackedPFCandidate_) {
      packedPFCandToken_ =
          consumes<edm::View<pat::PackedCandidate>>(ps.getParameter<edm::InputTag>(strPFCandSrc.c_str()));
    } else {
      recoPFCandToken_ = consumes<edm::View<reco::PFCandidate>>(ps.getParameter<edm::InputTag>(strPFCandSrc.c_str()));
    }
  }

  vertexToken_ = consumes<std::vector<reco::Vertex>>(ps.getParameter<edm::InputTag>("vertexSrc"));
  rhoToken_ = consumes<double>(ps.getParameter<edm::InputTag>("rhoSrc"));

  if (doGenParticles_) {
    pileupToken_ = consumes<std::vector<PileupSummaryInfo>>(ps.getParameter<edm::InputTag>("pileupSrc"));

    doPackedGenParticle_ = ps.getParameter<bool>("doPackedGenParticle");
    std::string strGenParticleSrc = "genParticleSrc";
    if (doPackedGenParticle_) {
      packedGenParticlesToken_ =
          consumes<std::vector<pat::PackedGenParticle>>(ps.getParameter<edm::InputTag>(strGenParticleSrc.c_str()));
      signalPackedGenParticlesToken_ =
          consumes<edm::View<pat::PackedGenParticle>>(ps.getParameter<edm::InputTag>("signalGenParticleSrc"));
    } else {
      genParticlesToken_ =
          consumes<std::vector<reco::GenParticle>>(ps.getParameter<edm::InputTag>(strGenParticleSrc.c_str()));
    }
  }

  if (doElectrons_) {
    electronsToken_ = consumes<edm::View<pat::Electron>>(ps.getParameter<edm::InputTag>("electronSrc"));
    beamSpotToken_ = consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamSpotSrc"));
    conversionsToken_ = consumes<reco::ConversionCollection>(ps.getParameter<edm::InputTag>("conversionsSrc"));
  }

  if (doPhotons_) {
    photonsToken_ = consumes<edm::View<pat::Photon>>(ps.getParameter<edm::InputTag>("photonSrc"));
  }

  if (doMuons_) {
    muonsToken_ = consumes<edm::View<pat::Muon>>(ps.getParameter<edm::InputTag>("muonSrc"));
    muonPtMin_ = ps.getParameter<double>("muonPtMin");
  }

  ttbESToken_ = esConsumes(edm::ESInputTag("", "TransientTrackBuilder"));
  geometryToken_ = esConsumes();

  // initialize output TTree
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("EventTree", "electrons, photons, muons");

  bookBranch("run", &run_);
  bookBranch("event", &event_);
  bookBranch("lumis", &lumis_);

  bookBranch("rho", &rho_);

  if (doGenParticles_) {
    bookBranch("nPUInfo", &nPUInfo_);
    bookBranch("nPU", &nPU_);
    bookBranch("puBX", &puBX_);
    bookBranch("puTrue", &puTrue_);

    bookBranch("nMC", &nMC_);

    bookBranch("mcVtx_x", &mcVtx_x_);
    bookBranch("mcVtx_y", &mcVtx_y_);
    bookBranch("mcVtx_z", &mcVtx_z_);

    bookBranch("mcPID", &mcPID_);
    bookBranch("mcStatus", &mcStatus_);
    bookBranch("mcPt", &mcPt_);
    bookBranch("mcEta", &mcEta_);
    bookBranch("mcPhi", &mcPhi_);
    bookBranch("mcE", &mcE_);
    bookBranch("mcEt", &mcEt_);
    bookBranch("mcMass", &mcMass_);

    bookBranch("mcParentage", &mcParentage_);
    bookBranch("mcMomPID", &mcMomPID_);
    bookBranch("mcMomKey", &mcMomKey_);
    bookBranch("mcMomPt", &mcMomPt_);
    bookBranch("mcMomEta", &mcMomEta_);
    bookBranch("mcMomPhi", &mcMomPhi_);
    bookBranch("mcMomMass", &mcMomMass_);
    bookBranch("mcGMomPID", &mcGMomPID_);
    bookBranch("mcGMomKey", &mcGMomKey_);
    bookBranch("mcGGMomPID", &mcGGMomPID_);
    bookBranch("mcGGMomKey", &mcGGMomKey_);
    bookBranch("mcIndex", &mcIndex_);
    bookBranch("mcSube", &mcSube_);

    bookBranch("mcCalIsoDR03", &mcCalIsoDR03_);
    bookBranch("mcCalIsoDR04", &mcCalIsoDR04_);
    bookBranch("mcTrkIsoDR03", &mcTrkIsoDR03_);
    bookBranch("mcTrkIsoDR04", &mcTrkIsoDR04_);
  }

  if (doSuperClusters_) {
    bookBranch("nSC", &nSC_);
    bookBranch("scE", &scE_);
    bookBranch("scRawE", &scRawE_);
    bookBranch("scEta", &scEta_);
    bookBranch("scPhi", &scPhi_);
  }

  if (doElectrons_) {
    bookBranch("nEle", &nEle_);

    bookBranch("eleD0", &eleD0_);
    bookBranch("eleDz", &eleDz_);
    bookBranch("eleD0Err", &eleD0Err_);
    bookBranch("eleDzErr", &eleDzErr_);
    bookBranch("eleTrkPt", &eleTrkPt_);
    bookBranch("eleTrkEta", &eleTrkEta_);
    bookBranch("eleTrkPhi", &eleTrkPhi_);
    bookBranch("eleTrkCharge", &eleTrkCharge_);
    bookBranch("eleTrkPtErr", &eleTrkPtErr_);
    bookBranch("eleTrkChi2", &eleTrkChi2_);
    bookBranch("eleTrkNdof", &eleTrkNdof_);
    bookBranch("eleTrkNormalizedChi2", &eleTrkNormalizedChi2_);
    bookBranch("eleTrkValidHits", &eleTrkValidHits_);
    bookBranch("eleTrkLayers", &eleTrkLayers_);
    bookBranch("eleMissHits", &eleMissHits_);
    bookBranch("eleIP3D", &eleIP3D_);
    bookBranch("eleIP3DErr", &eleIP3DErr_);

    bookBranch("elePt", &elePt_);
    bookBranch("eleRawPt", &eleRawPt_);
    bookBranch("eleEta", &eleEta_);
    bookBranch("elePhi", &elePhi_);
    bookBranch("eleCharge", &eleCharge_);
    bookBranch("eleEn", &eleEn_);

    bookBranch("eleMVAIso", &eleMVAIso_);
    for (auto& w : eleMVAIsoWP_)
      bookBranch(("eleMVAIso"+w.first).c_str(), &(w.second));
    bookBranch("eleMVAId", &eleMVAId_);
    for (auto& w : eleMVAIdWP_)
      bookBranch(("eleMVAId"+w.first).c_str(), &(w.second));
    for (auto& w : eleCutIdWP_)
      bookBranch(("eleCutId"+w.first).c_str(), &(w.second));

    bookBranch("eleSCEn", &eleSCEn_);
    bookBranch("eleESEn", &eleESEn_);
    bookBranch("eleSCEta", &eleSCEta_);
    bookBranch("eleSCPhi", &eleSCPhi_);
    bookBranch("eleSCRawEn", &eleSCRawEn_);
    bookBranch("eleSCEtaWidth", &eleSCEtaWidth_);
    bookBranch("eleSCPhiWidth", &eleSCPhiWidth_);
    bookBranch("eleSCClustersSize", &eleSCClustersSize_);
    bookBranch("eleSeedEn", &eleSeedEn_);
    bookBranch("eleSeedEta", &eleSeedEta_);
    bookBranch("eleSeedPhi", &eleSeedPhi_);

    bookBranch("eleHoverE", &eleHoverE_);
    bookBranch("eleHoverEBc", &eleHoverEBc_);
    bookBranch("eleRawHoverE", &eleRawHoverE_);
    bookBranch("eleRawHoverEBc", &eleRawHoverEBc_);
    bookBranch("eleEoverP", &eleEoverP_);
    bookBranch("eleEoverPInv", &eleEoverPInv_);
    bookBranch("eleEcalE", &eleEcalE_);
    bookBranch("eleRawEcalE", &eleRawEcalE_);
    bookBranch("elePAtVtx", &elePAtVtx_);
    bookBranch("elePAtSC", &elePAtSC_);
    bookBranch("elePAtCluster", &elePAtCluster_);
    bookBranch("elePAtSeed", &elePAtSeed_);
    bookBranch("eledEtaAtVtx", &eledEtaAtVtx_);
    bookBranch("eledPhiAtVtx", &eledPhiAtVtx_);
    bookBranch("eledEtaSeedAtVtx", &eledEtaSeedAtVtx_);
    bookBranch("eleSigmaIEtaIEta", &eleSigmaIEtaIEta_);
    bookBranch("eleSigmaIPhiIPhi", &eleSigmaIPhiIPhi_);
    bookBranch("eleBrem", &eleBrem_);

    bookBranch("eleConvVeto", &eleConvVeto_);

    bookBranch("eleR9", &eleR9_);
    bookBranch("eleE3x3", &eleE3x3_);
    bookBranch("eleE5x5", &eleE5x5_);
    bookBranch("eleR9Full5x5", &eleR9Full5x5_);
    bookBranch("eleE3x3Full5x5", &eleE3x3Full5x5_);
    bookBranch("eleE5x5Full5x5", &eleE5x5Full5x5_);
    bookBranch("eleSigmaIEtaIEta_2012", &eleSigmaIEtaIEta_2012_);

    bookBranch("elePFChIso", &elePFChIso_);
    bookBranch("elePFPhoIso", &elePFPhoIso_);
    bookBranch("elePFNeuIso", &elePFNeuIso_);
    bookBranch("elePFPUIso", &elePFPUIso_);

    if (doEffectiveAreas_) {
      bookBranch("elePFRelIsoWithEA", &elePFRelIsoWithEA_);
      bookBranch("elePFRelIsoWithDBeta", &elePFRelIsoWithDBeta_);
      bookBranch("eleEffAreaTimesRho", &eleEffAreaTimesRho_);
    }

    if (doPfIso_) {
      bookBranch("elePFChIso03", &elePFChIso03_);
      bookBranch("elePFPhoIso03", &elePFPhoIso03_);
      bookBranch("elePFNeuIso03", &elePFNeuIso03_);
      bookBranch("elePFChIso04", &elePFChIso04_);
      bookBranch("elePFPhoIso04", &elePFPhoIso04_);
      bookBranch("elePFNeuIso04", &elePFNeuIso04_);
    }

    bookBranch("eleSeedCryEta", &eleSeedCryEta_);
    bookBranch("eleSeedCryPhi", &eleSeedCryPhi_);
    bookBranch("eleSeedCryIeta", &eleSeedCryIeta_);
    bookBranch("eleSeedCryIphi", &eleSeedCryIphi_);

    if (doGenParticles_)
      bookBranch("ele_genMatchedIndex", &ele_genMatchedIndex_);
  }

  if (doPhotons_) {
    bookBranch("nPho", &nPho_);

    bookBranch("phoE", &phoE_);
    bookBranch("phoEt", &phoEt_);
    bookBranch("phoEta", &phoEta_);
    bookBranch("phoPhi", &phoPhi_);

    bookBranch("phoEcorrStdEcal", &phoEcorrStdEcal_);
    bookBranch("phoEcorrPhoEcal", &phoEcorrPhoEcal_);
    bookBranch("phoEcorrRegr1", &phoEcorrRegr1_);
    bookBranch("phoEcorrRegr2", &phoEcorrRegr2_);
    bookBranch("phoEcorrErrStdEcal", &phoEcorrErrStdEcal_);
    bookBranch("phoEcorrErrPhoEcal", &phoEcorrErrPhoEcal_);
    bookBranch("phoEcorrErrRegr1", &phoEcorrErrRegr1_);
    bookBranch("phoEcorrErrRegr2", &phoEcorrErrRegr2_);

    bookBranch("phoSCE", &phoSCE_);
    bookBranch("phoSCRawE", &phoSCRawE_);
    bookBranch("phoSCEta", &phoSCEta_);
    bookBranch("phoSCPhi", &phoSCPhi_);
    bookBranch("phoSCEtaWidth", &phoSCEtaWidth_);
    bookBranch("phoSCPhiWidth", &phoSCPhiWidth_);
    bookBranch("phoSCBrem", &phoSCBrem_);
    bookBranch("phoSCnHits", &phoSCnHits_);
    bookBranch("phoSCflags", &phoSCflags_);
    bookBranch("phoSCinClean", &phoSCinClean_);
    bookBranch("phoSCinUnClean", &phoSCinUnClean_);
    bookBranch("phoSCnBC", &phoSCnBC_);
    bookBranch("phoESEn", &phoESEn_);

    bookBranch("phoIsPFPhoton", &phoIsPFPhoton_);
    bookBranch("phoIsStandardPhoton", &phoIsStandardPhoton_);
    bookBranch("phoHasPixelSeed", &phoHasPixelSeed_);
    bookBranch("phoHasConversionTracks", &phoHasConversionTracks_);
    bookBranch("phoHadTowerOverEm", &phoHadTowerOverEm_);
    bookBranch("phoHoverE", &phoHoverE_);
    bookBranch("phoHoverEValid", &phoHoverEValid_);
    bookBranch("phoSigmaIEtaIEta", &phoSigmaIEtaIEta_);
    bookBranch("phoR9", &phoR9_);
    bookBranch("phoE1x5", &phoE1x5_);
    bookBranch("phoE2x5", &phoE2x5_);
    bookBranch("phoE3x3", &phoE3x3_);
    bookBranch("phoE5x5", &phoE5x5_);
    bookBranch("phoMaxEnergyXtal", &phoMaxEnergyXtal_);
    bookBranch("phoSigmaEtaEta", &phoSigmaEtaEta_);
    bookBranch("phoSigmaIEtaIEta_2012", &phoSigmaIEtaIEta_2012_);
    bookBranch("phoR9_2012", &phoR9_2012_);
    bookBranch("phoE1x5_2012", &phoE1x5_2012_);
    bookBranch("phoE2x5_2012", &phoE2x5_2012_);
    bookBranch("phoE3x3_2012", &phoE3x3_2012_);
    bookBranch("phoE5x5_2012", &phoE5x5_2012_);
    bookBranch("phoMaxEnergyXtal_2012", &phoMaxEnergyXtal_2012_);
    bookBranch("phoSigmaEtaEta_2012", &phoSigmaEtaEta_2012_);

    if (doPhoEReg_) {
      bookBranch("phoHadTowerOverEm1", &phoHadTowerOverEm1_);
      bookBranch("phoHadTowerOverEm2", &phoHadTowerOverEm2_);
      bookBranch("phoHoverE1", &phoHoverE1_);
      bookBranch("phoHoverE2", &phoHoverE2_);

      bookBranch("phoSigmaIEtaIPhi", &phoSigmaIEtaIPhi_);
      bookBranch("phoSigmaIPhiIPhi", &phoSigmaIPhiIPhi_);
      bookBranch("phoR1x5", &phoR1x5_);
      bookBranch("phoR2x5", &phoR2x5_);
      bookBranch("phoE2nd", &phoE2nd_);
      bookBranch("phoETop", &phoETop_);
      bookBranch("phoEBottom", &phoEBottom_);
      bookBranch("phoELeft", &phoELeft_);
      bookBranch("phoERight", &phoERight_);
      bookBranch("phoE1x3", &phoE1x3_);
      bookBranch("phoE2x2", &phoE2x2_);
      bookBranch("phoE2x5Max", &phoE2x5Max_);
      bookBranch("phoE2x5Top", &phoE2x5Top_);
      bookBranch("phoE2x5Bottom", &phoE2x5Bottom_);
      bookBranch("phoE2x5Left", &phoE2x5Left_);
      bookBranch("phoE2x5Right", &phoE2x5Right_);
      //bookBranch("phoSMMajor",         &phoSMMajor_);
      //bookBranch("phoSMMinor",         &phoSMMinor_);
      //bookBranch("phoSMAlpha",         &phoSMAlpha_);

      bookBranch("phoSigmaIEtaIPhi_2012", &phoSigmaIEtaIPhi_2012_);
      bookBranch("phoSigmaIPhiIPhi_2012", &phoSigmaIPhiIPhi_2012_);
      bookBranch("phoR1x5_2012", &phoR1x5_2012_);
      bookBranch("phoR2x5_2012", &phoR2x5_2012_);
      bookBranch("phoE2nd_2012", &phoE2nd_2012_);
      bookBranch("phoETop_2012", &phoETop_2012_);
      bookBranch("phoEBottom_2012", &phoEBottom_2012_);
      bookBranch("phoELeft_2012", &phoELeft_2012_);
      bookBranch("phoERight_2012", &phoERight_2012_);
      bookBranch("phoE1x3_2012", &phoE1x3_2012_);
      bookBranch("phoE2x2_2012", &phoE2x2_2012_);
      bookBranch("phoE2x5Max_2012", &phoE2x5Max_2012_);
      bookBranch("phoE2x5Top_2012", &phoE2x5Top_2012_);
      bookBranch("phoE2x5Bottom_2012", &phoE2x5Bottom_2012_);
      bookBranch("phoE2x5Left_2012", &phoE2x5Left_2012_);
      bookBranch("phoE2x5Right_2012", &phoE2x5Right_2012_);
      //bookBranch("phoSMMajor_2012",       &phoSMMajor_2012_);
      //bookBranch("phoSMMinor_2012",       &phoSMMinor_2012_);
      //bookBranch("phoSMAlpha_2012",       &phoSMAlpha_2012_);
    }

    bookBranch("phoBC1E", &phoBC1E_);
    bookBranch("phoBC1Ecorr", &phoBC1Ecorr_);
    bookBranch("phoBC1Eta", &phoBC1Eta_);
    bookBranch("phoBC1Phi", &phoBC1Phi_);
    bookBranch("phoBC1size", &phoBC1size_);
    bookBranch("phoBC1flags", &phoBC1flags_);
    bookBranch("phoBC1inClean", &phoBC1inClean_);
    bookBranch("phoBC1inUnClean", &phoBC1inUnClean_);
    bookBranch("phoBC1rawID", &phoBC1rawID_);

    if (useValMapIso_) {
      bookBranch("pho_ecalClusterIsoR1", &pho_ecalClusterIsoR1_);
      bookBranch("pho_ecalClusterIsoR2", &pho_ecalClusterIsoR2_);
      bookBranch("pho_ecalClusterIsoR3", &pho_ecalClusterIsoR3_);
      bookBranch("pho_ecalClusterIsoR4", &pho_ecalClusterIsoR4_);
      bookBranch("pho_ecalClusterIsoR5", &pho_ecalClusterIsoR5_);
      bookBranch("pho_hcalRechitIsoR1", &pho_hcalRechitIsoR1_);
      bookBranch("pho_hcalRechitIsoR2", &pho_hcalRechitIsoR2_);
      bookBranch("pho_hcalRechitIsoR3", &pho_hcalRechitIsoR3_);
      bookBranch("pho_hcalRechitIsoR4", &pho_hcalRechitIsoR4_);
      bookBranch("pho_hcalRechitIsoR5", &pho_hcalRechitIsoR5_);
      bookBranch("pho_trackIsoR1PtCut20", &pho_trackIsoR1PtCut20_);
      bookBranch("pho_trackIsoR2PtCut20", &pho_trackIsoR2PtCut20_);
      bookBranch("pho_trackIsoR3PtCut20", &pho_trackIsoR3PtCut20_);
      bookBranch("pho_trackIsoR4PtCut20", &pho_trackIsoR4PtCut20_);
      bookBranch("pho_trackIsoR5PtCut20", &pho_trackIsoR5PtCut20_);
      bookBranch("pho_swissCrx", &pho_swissCrx_);
      bookBranch("pho_seedTime", &pho_seedTime_);
    }

    if (doGenParticles_) {
      bookBranch("pho_genMatchedIndex", &pho_genMatchedIndex_);
    }

    if (doRecHitsEB_ || doRecHitsEE_) {
      // rechit info
      bookBranch("nRH", &nRH_);
      bookBranch("rhRawId", &rhRawId_);
      bookBranch("rhieta", &rhieta_);
      bookBranch("rhiphi", &rhiphi_);
      bookBranch("rhix", &rhix_);
      bookBranch("rhiy", &rhiy_);
      bookBranch("rhE", &rhE_);
      bookBranch("rhEt", &rhEt_);
      bookBranch("rhEta", &rhEta_);
      bookBranch("rhPhi", &rhPhi_);
      bookBranch("rhChi2", &rhChi2_);
      bookBranch("rhEerror", &rhEerror_);
      bookBranch("rhFlags", &rhFlags_);
      bookBranch("rhPhoIdx", &rhPhoIdx_);
      bookBranch("rhBCIdx", &rhBCIdx_);
    }

    if (doPfIso_) {
      bookBranch("pfcIso1", &pfcIso1_);
      bookBranch("pfcIso2", &pfcIso2_);
      bookBranch("pfcIso3", &pfcIso3_);
      bookBranch("pfcIso4", &pfcIso4_);
      bookBranch("pfcIso5", &pfcIso5_);

      bookBranch("pfpIso1", &pfpIso1_);
      bookBranch("pfpIso2", &pfpIso2_);
      bookBranch("pfpIso3", &pfpIso3_);
      bookBranch("pfpIso4", &pfpIso4_);
      bookBranch("pfpIso5", &pfpIso5_);

      bookBranch("pfnIso1", &pfnIso1_);
      bookBranch("pfnIso2", &pfnIso2_);
      bookBranch("pfnIso3", &pfnIso3_);
      bookBranch("pfnIso4", &pfnIso4_);
      bookBranch("pfnIso5", &pfnIso5_);

      bookBranch("pfpIso1subSC", &pfpIso1subSC_);
      bookBranch("pfpIso2subSC", &pfpIso2subSC_);
      bookBranch("pfpIso3subSC", &pfpIso3subSC_);
      bookBranch("pfpIso4subSC", &pfpIso4subSC_);
      bookBranch("pfpIso5subSC", &pfpIso5subSC_);

      bookBranch("pfcIso1subUE", &pfcIso1subUE_);
      bookBranch("pfcIso2subUE", &pfcIso2subUE_);
      bookBranch("pfcIso3subUE", &pfcIso3subUE_);
      bookBranch("pfcIso4subUE", &pfcIso4subUE_);
      bookBranch("pfcIso5subUE", &pfcIso5subUE_);

      bookBranch("pfpIso1subUE", &pfpIso1subUE_);
      bookBranch("pfpIso2subUE", &pfpIso2subUE_);
      bookBranch("pfpIso3subUE", &pfpIso3subUE_);
      bookBranch("pfpIso4subUE", &pfpIso4subUE_);
      bookBranch("pfpIso5subUE", &pfpIso5subUE_);

      bookBranch("pfnIso1subUE", &pfnIso1subUE_);
      bookBranch("pfnIso2subUE", &pfnIso2subUE_);
      bookBranch("pfnIso3subUE", &pfnIso3subUE_);
      bookBranch("pfnIso4subUE", &pfnIso4subUE_);
      bookBranch("pfnIso5subUE", &pfnIso5subUE_);

      bookBranch("pfpIso1subSCsubUE", &pfpIso1subSCsubUE_);
      bookBranch("pfpIso2subSCsubUE", &pfpIso2subSCsubUE_);
      bookBranch("pfpIso3subSCsubUE", &pfpIso3subSCsubUE_);
      bookBranch("pfpIso4subSCsubUE", &pfpIso4subSCsubUE_);
      bookBranch("pfpIso5subSCsubUE", &pfpIso5subSCsubUE_);

      bookBranch("pfcIso1pTgt1p0subUE", &pfcIso1pTgt1p0subUE_);
      bookBranch("pfcIso2pTgt1p0subUE", &pfcIso2pTgt1p0subUE_);
      bookBranch("pfcIso3pTgt1p0subUE", &pfcIso3pTgt1p0subUE_);
      bookBranch("pfcIso4pTgt1p0subUE", &pfcIso4pTgt1p0subUE_);
      bookBranch("pfcIso5pTgt1p0subUE", &pfcIso5pTgt1p0subUE_);

      bookBranch("pfcIso1pTgt2p0subUE", &pfcIso1pTgt2p0subUE_);
      bookBranch("pfcIso2pTgt2p0subUE", &pfcIso2pTgt2p0subUE_);
      bookBranch("pfcIso3pTgt2p0subUE", &pfcIso3pTgt2p0subUE_);
      bookBranch("pfcIso4pTgt2p0subUE", &pfcIso4pTgt2p0subUE_);
      bookBranch("pfcIso5pTgt2p0subUE", &pfcIso5pTgt2p0subUE_);

      bookBranch("pfcIso1pTgt3p0subUE", &pfcIso1pTgt3p0subUE_);
      bookBranch("pfcIso2pTgt3p0subUE", &pfcIso2pTgt3p0subUE_);
      bookBranch("pfcIso3pTgt3p0subUE", &pfcIso3pTgt3p0subUE_);
      bookBranch("pfcIso4pTgt3p0subUE", &pfcIso4pTgt3p0subUE_);
      bookBranch("pfcIso5pTgt3p0subUE", &pfcIso5pTgt3p0subUE_);

      // photon pf isolation UE-subtracted and cone excluded
      bookBranch("pfcIso2subUEec", &pfcIso2subUEec_);
      bookBranch("pfcIso3subUEec", &pfcIso3subUEec_);
      bookBranch("pfcIso4subUEec", &pfcIso4subUEec_);

      bookBranch("pfpIso2subUEec", &pfpIso2subUEec_);
      bookBranch("pfpIso3subUEec", &pfpIso3subUEec_);
      bookBranch("pfpIso4subUEec", &pfpIso4subUEec_);

      bookBranch("pfnIso2subUEec", &pfnIso2subUEec_);
      bookBranch("pfnIso3subUEec", &pfnIso3subUEec_);
      bookBranch("pfnIso4subUEec", &pfnIso4subUEec_);

      bookBranch("pfcIso2pTgt2p0subUEec", &pfcIso2pTgt2p0subUEec_);
      bookBranch("pfcIso3pTgt2p0subUEec", &pfcIso3pTgt2p0subUEec_);
      bookBranch("pfcIso4pTgt2p0subUEec", &pfcIso4pTgt2p0subUEec_);
    }
  }

  if (doMuons_) {
    bookBranch("nMu", &nMu_);

    bookBranch("muPt", &muPt_);
    bookBranch("muEta", &muEta_);
    bookBranch("muPhi", &muPhi_);
    bookBranch("muL1Eta", &muL1Eta_);
    bookBranch("muL1Phi", &muL1Phi_);
    bookBranch("muMVAIso", &muMVAIso_);
    for (auto& w : muMVAIsoWP_)
      bookBranch(("muMVAIso"+w.first).c_str(), &(w.second));
    bookBranch("muCharge", &muCharge_);
    bookBranch("muType", &muType_);
    bookBranch("muIsGood", &muIsGood_);

    bookBranch("muIsGlobal", &muIsGlobal_);
    bookBranch("muIsTracker", &muIsTracker_);
    bookBranch("muIsPF", &muIsPF_);
    bookBranch("muIsSTA", &muIsSTA_);

    bookBranch("muD0", &muD0_);
    bookBranch("muDz", &muDz_);
    bookBranch("muIP3D", &muIP3D_);
    bookBranch("muD0Err", &muD0Err_);
    bookBranch("muDzErr", &muDzErr_);
    bookBranch("muIP3DErr", &muIP3DErr_);
    bookBranch("muChi2NDF", &muChi2NDF_);
    bookBranch("muInnerD0", &muInnerD0_);
    bookBranch("muInnerDz", &muInnerDz_);

    bookBranch("muInnerD0Err", &muInnerD0Err_);
    bookBranch("muInnerDzErr", &muInnerDzErr_);
    bookBranch("muInnerPt", &muInnerPt_);
    bookBranch("muInnerPtErr", &muInnerPtErr_);
    bookBranch("muInnerEta", &muInnerEta_);

    bookBranch("muTrkLayers", &muTrkLayers_);
    bookBranch("muPixelLayers", &muPixelLayers_);
    bookBranch("muPixelHits", &muPixelHits_);
    bookBranch("muMuonHits", &muMuonHits_);
    bookBranch("muTrkQuality", &muTrkQuality_);
    bookBranch("muStations", &muStations_);
    bookBranch("muIsoTrk", &muIsoTrk_);
    bookBranch("muPFChIso", &muPFChIso_);
    bookBranch("muPFPhoIso", &muPFPhoIso_);
    bookBranch("muPFNeuIso", &muPFNeuIso_);
    bookBranch("muPFPUIso", &muPFPUIso_);

    bookBranch("muSelectors", &muSelectors_);
    bookBranch("muIDSoft", &muIDSoft_);
    bookBranch("muIDLoose", &muIDLoose_);
    bookBranch("muIDMedium", &muIDMedium_);
    bookBranch("muIDMediumPrompt", &muIDMediumPrompt_);
    bookBranch("muIDTight", &muIDTight_);
    bookBranch("muIDGlobalHighPt", &muIDGlobalHighPt_);
    bookBranch("muIDTrkHighPt", &muIDTrkHighPt_);
    bookBranch("muIDInTime", &muIDInTime_);

    if (doGenParticles_)
      bookBranch("mu_genMatchedIndex", &mu_genMatchedIndex_);
  }
}

ggHiNtuplizer::~ggHiNtuplizer() {}

void ggHiNtuplizer::analyze(const edm::Event& e, const edm::EventSetup& es) {
  // cleanup from previous event

  if (doGenParticles_) {
    nPUInfo_ = 0;
    nPU_.clear();
    puBX_.clear();
    puTrue_.clear();

    nMC_ = 0;

    mcVtx_x_.clear();
    mcVtx_y_.clear();
    mcVtx_z_.clear();

    mcPID_.clear();
    mcStatus_.clear();
    mcPt_.clear();
    mcEta_.clear();
    mcPhi_.clear();
    mcE_.clear();
    mcEt_.clear();
    mcMass_.clear();

    mcParentage_.clear();
    mcMomPID_.clear();
    mcMomKey_.clear();
    mcMomPt_.clear();
    mcMomEta_.clear();
    mcMomPhi_.clear();
    mcMomMass_.clear();
    mcGMomPID_.clear();
    mcGMomKey_.clear();
    mcGGMomPID_.clear();
    mcGGMomKey_.clear();
    mcIndex_.clear();
    mcSube_.clear();

    mcCalIsoDR03_.clear();
    mcCalIsoDR04_.clear();
    mcTrkIsoDR03_.clear();
    mcTrkIsoDR04_.clear();
  }

  if (doSuperClusters_) {
    nSC_ = 0;
    scE_.clear();
    scRawE_.clear();
    scEta_.clear();
    scPhi_.clear();
  }

  if (doElectrons_) {
    nEle_ = 0;

    eleD0_.clear();
    eleDz_.clear();
    eleD0Err_.clear();
    eleDzErr_.clear();
    eleTrkPt_.clear();
    eleTrkEta_.clear();
    eleTrkPhi_.clear();
    eleTrkCharge_.clear();
    eleTrkPtErr_.clear();
    eleTrkChi2_.clear();
    eleTrkNdof_.clear();
    eleTrkNormalizedChi2_.clear();
    eleTrkValidHits_.clear();
    eleTrkLayers_.clear();
    eleMissHits_.clear();
    eleIP3D_.clear();
    eleIP3DErr_.clear();

    elePt_.clear();
    eleRawPt_.clear();
    eleEta_.clear();
    elePhi_.clear();
    eleCharge_.clear();
    eleEn_.clear();

    eleMVAIso_.clear();
    for (auto& w : eleMVAIsoWP_)
      w.second.clear();
    eleMVAId_.clear();
    for (auto& w : eleMVAIdWP_)
      w.second.clear();
    for (auto& w : eleCutIdWP_)
      w.second.clear();

    eleSCEn_.clear();
    eleESEn_.clear();
    eleSCEta_.clear();
    eleSCPhi_.clear();
    eleSCRawEn_.clear();
    eleSCEtaWidth_.clear();
    eleSCPhiWidth_.clear();
    eleSCClustersSize_.clear();
    eleSeedEn_.clear();
    eleSeedEta_.clear();
    eleSeedPhi_.clear();

    eleHoverE_.clear();
    eleHoverEBc_.clear();
    eleRawHoverE_.clear();
    eleRawHoverEBc_.clear();
    eleEoverP_.clear();
    eleEoverPInv_.clear();
    eleEcalE_.clear();
    eleRawEcalE_.clear();
    elePAtVtx_.clear();
    elePAtSC_.clear();
    elePAtCluster_.clear();
    elePAtSeed_.clear();
    eleBrem_.clear();
    eledEtaAtVtx_.clear();
    eledPhiAtVtx_.clear();
    eledEtaSeedAtVtx_.clear();
    eleSigmaIEtaIEta_.clear();
    eleSigmaIPhiIPhi_.clear();

    eleConvVeto_.clear();

    elePFChIso_.clear();
    elePFPhoIso_.clear();
    elePFNeuIso_.clear();
    elePFPUIso_.clear();

    elePFRelIsoWithEA_.clear();
    elePFRelIsoWithDBeta_.clear();
    eleEffAreaTimesRho_.clear();

    if (doPfIso_) {
      elePFChIso03_.clear();
      elePFPhoIso03_.clear();
      elePFNeuIso03_.clear();
      elePFChIso04_.clear();
      elePFPhoIso04_.clear();
      elePFNeuIso04_.clear();
    }

    eleR9_.clear();
    eleE3x3_.clear();
    eleE5x5_.clear();
    eleR9Full5x5_.clear();
    eleE3x3Full5x5_.clear();
    eleE5x5Full5x5_.clear();
    eleSigmaIEtaIEta_2012_.clear();

    eleSeedCryEta_.clear();
    eleSeedCryPhi_.clear();
    eleSeedCryIeta_.clear();
    eleSeedCryIphi_.clear();

    ele_genMatchedIndex_.clear();
  }

  if (doPhotons_) {
    nPho_ = 0;

    phoE_.clear();
    phoEt_.clear();
    phoEta_.clear();
    phoPhi_.clear();

    phoEcorrStdEcal_.clear();
    phoEcorrPhoEcal_.clear();
    phoEcorrRegr1_.clear();
    phoEcorrRegr2_.clear();
    phoEcorrErrStdEcal_.clear();
    phoEcorrErrPhoEcal_.clear();
    phoEcorrErrRegr1_.clear();
    phoEcorrErrRegr2_.clear();

    phoSCE_.clear();
    phoSCRawE_.clear();
    phoSCEta_.clear();
    phoSCPhi_.clear();
    phoSCEtaWidth_.clear();
    phoSCPhiWidth_.clear();
    phoSCBrem_.clear();
    phoSCnHits_.clear();
    phoSCflags_.clear();
    phoSCinClean_.clear();
    phoSCinUnClean_.clear();
    phoSCnBC_.clear();
    phoESEn_.clear();

    phoIsPFPhoton_.clear();
    phoIsStandardPhoton_.clear();
    phoHasPixelSeed_.clear();
    phoHasConversionTracks_.clear();
    phoHadTowerOverEm_.clear();
    phoHoverE_.clear();
    phoHoverEValid_.clear();
    phoSigmaIEtaIEta_.clear();
    phoR9_.clear();
    phoE1x5_.clear();
    phoE2x5_.clear();
    phoE3x3_.clear();
    phoE5x5_.clear();
    phoMaxEnergyXtal_.clear();
    phoSigmaEtaEta_.clear();
    phoSigmaIEtaIEta_2012_.clear();
    phoR9_2012_.clear();
    phoE1x5_2012_.clear();
    phoE2x5_2012_.clear();
    phoE3x3_2012_.clear();
    phoE5x5_2012_.clear();
    phoMaxEnergyXtal_2012_.clear();
    phoSigmaEtaEta_2012_.clear();

    if (doPhoEReg_) {
      phoHadTowerOverEm1_.clear();
      phoHadTowerOverEm2_.clear();
      phoHoverE1_.clear();
      phoHoverE2_.clear();

      phoSigmaIEtaIPhi_.clear();
      phoSigmaIPhiIPhi_.clear();
      phoR1x5_.clear();
      phoR2x5_.clear();
      phoE2nd_.clear();
      phoETop_.clear();
      phoEBottom_.clear();
      phoELeft_.clear();
      phoERight_.clear();
      phoE1x3_.clear();
      phoE2x2_.clear();
      phoE2x5Max_.clear();
      phoE2x5Top_.clear();
      phoE2x5Bottom_.clear();
      phoE2x5Left_.clear();
      phoE2x5Right_.clear();
      //phoSMMajor_.clear();
      //phoSMMinor_.clear();
      //phoSMAlpha_.clear();

      phoSigmaIEtaIPhi_2012_.clear();
      phoSigmaIPhiIPhi_2012_.clear();
      phoR1x5_2012_.clear();
      phoR2x5_2012_.clear();
      phoE2nd_2012_.clear();
      phoETop_2012_.clear();
      phoEBottom_2012_.clear();
      phoELeft_2012_.clear();
      phoERight_2012_.clear();
      phoE1x3_2012_.clear();
      phoE2x2_2012_.clear();
      phoE2x5Max_2012_.clear();
      phoE2x5Top_2012_.clear();
      phoE2x5Bottom_2012_.clear();
      phoE2x5Left_2012_.clear();
      phoE2x5Right_2012_.clear();
      //phoSMMajor_2012_.clear();
      //phoSMMinor_2012_.clear();
      //phoSMAlpha_2012_.clear();
    }

    phoBC1E_.clear();
    phoBC1Ecorr_.clear();
    phoBC1Eta_.clear();
    phoBC1Phi_.clear();
    phoBC1size_.clear();
    phoBC1flags_.clear();
    phoBC1inClean_.clear();
    phoBC1inUnClean_.clear();
    phoBC1rawID_.clear();

    if (useValMapIso_) {
      pho_ecalClusterIsoR1_.clear();
      pho_ecalClusterIsoR2_.clear();
      pho_ecalClusterIsoR3_.clear();
      pho_ecalClusterIsoR4_.clear();
      pho_ecalClusterIsoR5_.clear();
      pho_hcalRechitIsoR1_.clear();
      pho_hcalRechitIsoR2_.clear();
      pho_hcalRechitIsoR3_.clear();
      pho_hcalRechitIsoR4_.clear();
      pho_hcalRechitIsoR5_.clear();
      pho_trackIsoR1PtCut20_.clear();
      pho_trackIsoR2PtCut20_.clear();
      pho_trackIsoR3PtCut20_.clear();
      pho_trackIsoR4PtCut20_.clear();
      pho_trackIsoR5PtCut20_.clear();
      pho_swissCrx_.clear();
      pho_seedTime_.clear();
    }

    pho_genMatchedIndex_.clear();

    // rechit info
    if (doRecHitsEB_ || doRecHitsEE_) {
      nRH_ = 0;
      rhRawId_.clear();
      rhieta_.clear();
      rhiphi_.clear();
      rhix_.clear();
      rhiy_.clear();
      rhE_.clear();
      rhEt_.clear();
      rhEta_.clear();
      rhPhi_.clear();
      rhChi2_.clear();
      rhEerror_.clear();
      rhFlags_.clear();
      rhPhoIdx_.clear();
      rhBCIdx_.clear();
    }

    //photon pf isolation stuff
    if (doPfIso_) {
      pfcIso1_.clear();
      pfcIso2_.clear();
      pfcIso3_.clear();
      pfcIso4_.clear();
      pfcIso5_.clear();
      pfpIso1_.clear();
      pfpIso2_.clear();
      pfpIso3_.clear();
      pfpIso4_.clear();
      pfpIso5_.clear();
      pfnIso1_.clear();
      pfnIso2_.clear();
      pfnIso3_.clear();
      pfnIso4_.clear();
      pfnIso5_.clear();
      pfpIso1subSC_.clear();
      pfpIso2subSC_.clear();
      pfpIso3subSC_.clear();
      pfpIso4subSC_.clear();
      pfpIso5subSC_.clear();

      pfcIso1subUE_.clear();
      pfcIso2subUE_.clear();
      pfcIso3subUE_.clear();
      pfcIso4subUE_.clear();
      pfcIso5subUE_.clear();
      pfpIso1subUE_.clear();
      pfpIso2subUE_.clear();
      pfpIso3subUE_.clear();
      pfpIso4subUE_.clear();
      pfpIso5subUE_.clear();
      pfnIso1subUE_.clear();
      pfnIso2subUE_.clear();
      pfnIso3subUE_.clear();
      pfnIso4subUE_.clear();
      pfnIso5subUE_.clear();
      pfpIso1subSCsubUE_.clear();
      pfpIso2subSCsubUE_.clear();
      pfpIso3subSCsubUE_.clear();
      pfpIso4subSCsubUE_.clear();
      pfpIso5subSCsubUE_.clear();
      pfcIso1pTgt1p0subUE_.clear();
      pfcIso2pTgt1p0subUE_.clear();
      pfcIso3pTgt1p0subUE_.clear();
      pfcIso4pTgt1p0subUE_.clear();
      pfcIso5pTgt1p0subUE_.clear();
      pfcIso1pTgt2p0subUE_.clear();
      pfcIso2pTgt2p0subUE_.clear();
      pfcIso3pTgt2p0subUE_.clear();
      pfcIso4pTgt2p0subUE_.clear();
      pfcIso5pTgt2p0subUE_.clear();
      pfcIso1pTgt3p0subUE_.clear();
      pfcIso2pTgt3p0subUE_.clear();
      pfcIso3pTgt3p0subUE_.clear();
      pfcIso4pTgt3p0subUE_.clear();
      pfcIso5pTgt3p0subUE_.clear();

      pfcIso2subUEec_.clear();
      pfcIso3subUEec_.clear();
      pfcIso4subUEec_.clear();

      pfpIso2subUEec_.clear();
      pfpIso3subUEec_.clear();
      pfpIso4subUEec_.clear();

      pfnIso2subUEec_.clear();
      pfnIso3subUEec_.clear();
      pfnIso4subUEec_.clear();

      pfcIso2pTgt2p0subUEec_.clear();
      pfcIso3pTgt2p0subUEec_.clear();
      pfcIso4pTgt2p0subUEec_.clear();
    }
  }

  if (doMuons_) {
    nMu_ = 0;

    muPt_.clear();
    muEta_.clear();
    muPhi_.clear();
    muL1Eta_.clear();
    muL1Phi_.clear();
    muMVAIso_.clear();
    for (auto& w : muMVAIsoWP_)
      w.second.clear();
    muCharge_.clear();
    muType_.clear();
    muIsGood_.clear();

    muIsGlobal_.clear();
    muIsTracker_.clear();
    muIsPF_.clear();
    muIsSTA_.clear();

    muD0_.clear();
    muDz_.clear();
    muD0Err_.clear();
    muDzErr_.clear();
    muChi2NDF_.clear();
    muInnerD0_.clear();
    muInnerDz_.clear();
    muIP3D_.clear();
    muIP3DErr_.clear();

    muInnerD0Err_.clear();
    muInnerDzErr_.clear();
    muInnerPt_.clear();
    muInnerPtErr_.clear();
    muInnerEta_.clear();

    muTrkLayers_.clear();
    muPixelLayers_.clear();
    muPixelHits_.clear();
    muMuonHits_.clear();
    muTrkQuality_.clear();
    muStations_.clear();
    muIsoTrk_.clear();
    muPFChIso_.clear();
    muPFPhoIso_.clear();
    muPFNeuIso_.clear();
    muPFPUIso_.clear();

    muSelectors_.clear();
    muIDSoft_.clear();
    muIDLoose_.clear();
    muIDMedium_.clear();
    muIDMediumPrompt_.clear();
    muIDTight_.clear();
    muIDGlobalHighPt_.clear();
    muIDTrkHighPt_.clear();
    muIDInTime_.clear();

    mu_genMatchedIndex_.clear();
  }

  run_ = e.id().run();
  event_ = e.id().event();
  lumis_ = e.luminosityBlock();

  edm::Handle<double> rhoH;
  e.getByToken(rhoToken_, rhoH);
  rho_ = *rhoH;

  // MC truth
  if (doGenParticles_) {
    fillGenPileupInfo(e);
    fillGenParticles(e);
  }

  edm::Handle<std::vector<reco::Vertex>> vertices;
  e.getByToken(vertexToken_, vertices);

  // best-known primary vertex coordinates
  reco::Vertex pv(math::XYZPoint(0, 0, -999), math::Error<3>::type());
  for (auto const& vertex : *vertices) {
    if (!vertex.isFake()) {
      pv = vertex;
      break;
    }
  }

  tb = &es.getData(ttbESToken_);

  if (doRecHitsEB_ || doRecHitsEE_ || doElectrons_)
    geo = &es.getData(geometryToken_);

  if (doSuperClusters_)
    fillSC(e);
  if (doElectrons_)
    fillElectrons(e, es, pv);
  if (doPhotons_)
    fillPhotons(e, es, pv);
  if (doMuons_)
    fillMuons(e, es, pv);

  tree_->Fill();
}

void ggHiNtuplizer::fillGenPileupInfo(const edm::Event& e) {
  // Fills information about pileup from MC truth.
  edm::Handle<std::vector<PileupSummaryInfo>> pileup;
  e.getByToken(pileupToken_, pileup);

  for (const auto& pu : *pileup) {
    nPU_.push_back(pu.getPU_NumInteractions());
    puTrue_.push_back(pu.getTrueNumInteractions());
    puBX_.push_back(pu.getBunchCrossing());

    ++nPUInfo_;
  }
}

void ggHiNtuplizer::fillGenParticles(const edm::Event& e) {
  if (doPackedGenParticle_) {
    edm::Handle<std::vector<pat::PackedGenParticle>> packedGenParticles;
    e.getByToken(packedGenParticlesToken_, packedGenParticles);

    edm::Handle<edm::View<pat::PackedGenParticle>> signalPackedGenParticles;
    bool hasSignalPackedGen = e.getByToken(signalPackedGenParticlesToken_, signalPackedGenParticles);

    fillGenCandidates(packedGenParticles, signalPackedGenParticles, hasSignalPackedGen);
  } else {
    // ToDo
    edm::Handle<std::vector<reco::GenParticle>> genParticles;
    e.getByToken(genParticlesToken_, genParticles);

    edm::Handle<edm::View<reco::GenParticle> > dummyHandle;

    fillGenCandidates(genParticles, dummyHandle, false);
  }
}

template <typename T, typename T2>
void ggHiNtuplizer::fillGenCandidates(const edm::Handle<std::vector<T>>& handle,
                                      const edm::Handle<edm::View<T2>>& signalPartHandle,
                                      bool hasSignalPart) {
  for (size_t ig = 0; ig < handle->size(); ig++) {
    const auto& p = edm::Ref<std::vector<T>>(handle, ig);
    // skip all primary particles if not particle gun MC
    //if (!isParticleGun_ && !p->mother())
    if (!isParticleGun_ && p->numberOfMothers() == 0)
      continue;

    // require last copy
    if (!p->statusFlags().isLastCopy())
      continue;

    // stable particles with pT > 5 GeV
    bool isStableFast = (p->status() == 1 && p->pt() > 5.0);

    // stable leptons
    const auto pdgId = std::abs(p->pdgId());
    bool isStableLepton = (p->status() == 1 && pdgId >= 11 && pdgId <= 16);

    // (unstable) Z, W, H, top, bottom
    bool isHeavy = (pdgId == 23 || pdgId == 24 || pdgId == 25 || pdgId == 6 || pdgId == 5);

    // reduce size of output root file
    if (!isStableFast && !isStableLepton && !isHeavy)
      continue;

    mcVtx_x_.push_back(p->vx());
    mcVtx_y_.push_back(p->vy());
    mcVtx_z_.push_back(p->vz());

    mcPID_.push_back(p->pdgId());
    mcStatus_.push_back(p->status());
    mcPt_.push_back(p->pt());
    mcEta_.push_back(p->eta());
    mcPhi_.push_back(p->phi());
    mcE_.push_back(p->energy());
    mcEt_.push_back(p->et());
    mcMass_.push_back(p->mass());

    GenParticleParentage particleHistory(p);
    const auto& partRef = particleHistory.match();
    mcParentage_.push_back((particleHistory.hasLeptonParent() << 4) + (particleHistory.hasBosonParent() << 3) +
                           (particleHistory.hasNonPromptParent() << 2) + (particleHistory.hasQCDParent() << 1) +
                           (particleHistory.hasExoticParent() << 0));

    int momPID = -999;
    int momKey = -1;
    float momPt = -999;
    float momEta = -999;
    float momPhi = -999;
    float momMass = -999;
    int gmomPID = -999;
    int gmomKey = -1;
    int ggmomPID = -999;
    int ggmomKey = -1;

    // mother
    const auto& momRef = GenParticleParentage::findGenMother(partRef);
    if (momRef.isNonnull() && momRef.isAvailable()) {
      momPID = momRef->pdgId();
      momKey = std::abs(momPID) < 1E5 ? (std::abs(momPID) + momRef.key()*1E5) : -1;
      momPt = momRef->pt();
      momEta = momRef->eta();
      momPhi = momRef->phi();
      momMass = momRef->mass();

      // grandmother
      const auto& grandmother = GenParticleParentage::findGenMother(momRef);
      if (grandmother.isNonnull() && grandmother.isAvailable()) {
        gmomPID = grandmother->pdgId();
        gmomKey = std::abs(gmomPID) < 1E5 ? (std::abs(gmomPID) + grandmother.key()*1E5) : -1;

        // grand-grandmother
        const auto& ggmother = GenParticleParentage::findGenMother(grandmother);
        if (ggmother.isNonnull() && ggmother.isAvailable()) {
          ggmomPID = ggmother->pdgId();
          ggmomKey = std::abs(ggmomPID) < 1E5 ? (std::abs(ggmomPID) + ggmother.key()*1E5) : -1;
        }
      }
    }

    mcMomPID_.push_back(momPID);
    mcMomKey_.push_back(momKey);
    mcMomPt_.push_back(momPt);
    mcMomEta_.push_back(momEta);
    mcMomPhi_.push_back(momPhi);
    mcMomMass_.push_back(momMass);
    mcGMomPID_.push_back(gmomPID);
    mcGMomKey_.push_back(gmomKey);
    mcGGMomPID_.push_back(ggmomPID);
    mcGGMomKey_.push_back(ggmomKey);

    mcIndex_.push_back(partRef.key());

    bool isSigPart = false;
    if (doPackedGenParticle_) {
      int tmpSube = -999;
      if (hasSignalPart) {
        for (auto pSig = signalPartHandle->begin(); pSig != signalPartHandle->end(); ++pSig) {
          if (&(*pSig) == &(*p)) {
            tmpSube = 0;
            isSigPart = true;
            break;
          }
        }
        if (!isSigPart) {
          tmpSube = 1;
        }
      }
      mcSube_.push_back(tmpSube);
    } else {
      int tmpCollId = (dynamic_cast<const reco::GenParticle&>(*p)).collisionId();
      isSigPart = (tmpCollId == 0);
      mcSube_.push_back(tmpCollId);
    }

    if (hasSignalPart && isSigPart) {
      mcCalIsoDR03_.push_back(getGenCalIso(signalPartHandle, p, 0.3 * 0.3, false, false));
      mcCalIsoDR04_.push_back(getGenCalIso(signalPartHandle, p, 0.4 * 0.4, false, false));
      mcTrkIsoDR03_.push_back(getGenTrkIso(signalPartHandle, p, 0.3 * 0.3));
      mcTrkIsoDR04_.push_back(getGenTrkIso(signalPartHandle, p, 0.4 * 0.4));
    } else {
      mcCalIsoDR03_.push_back(-999999);
      mcCalIsoDR04_.push_back(-999999);
      mcTrkIsoDR03_.push_back(-999999);
      mcTrkIsoDR04_.push_back(-999999);
    }

    nMC_++;
  }
}

template <typename T, typename T2>
float ggHiNtuplizer::getGenCalIso(
    const edm::Handle<edm::View<T>>& handle, T2 thisPart, float dR2Max, bool removeMu, bool removeNu) {
  float etSum = 0;

  for (auto p = handle->begin(); p != handle->end(); ++p) {
    if (getGenCalIsoPass(&(*p), &(*thisPart), dR2Max, removeMu, removeNu)) {
      etSum += p->et();
    }
  }

  return etSum;
}

bool ggHiNtuplizer::getGenCalIsoPass(
    const reco::Candidate* p, const reco::Candidate* thisPart, float dR2Max, bool removeMu, bool removeNu) {
  if (p == thisPart)
    return false;
  if (p->status() != 1)
    return false;

  // no collisionId info for PackedGenParticle
  if (!doPackedGenParticle_) {
    if ((dynamic_cast<const reco::GenParticle&>(*thisPart)).collisionId() !=
        (dynamic_cast<const reco::GenParticle&>(*p)).collisionId())
      return false;
  }

  int pdgCode = abs(p->pdgId());

  // skip muons/neutrinos, if requested
  if (removeMu && pdgCode == 13)
    return false;
  if (removeNu && (pdgCode == 12 || pdgCode == 14 || pdgCode == 16))
    return false;

  // must be within deltaR cone
  float dR2 = reco::deltaR2(thisPart->momentum(), p->momentum());
  if (dR2 > dR2Max)
    return false;

  return true;
}

template <typename T, typename T2>
float ggHiNtuplizer::getGenTrkIso(const edm::Handle<edm::View<T>>& handle, T2 thisPart, float dR2Max) {
  float ptSum = 0;

  for (auto p = handle->begin(); p != handle->end(); ++p) {
    if (getGenTrkIsoPass(&(*p), &(*thisPart), dR2Max)) {
      ptSum += p->pt();
    }
  }

  return ptSum;
}

bool ggHiNtuplizer::getGenTrkIsoPass(const reco::Candidate* p, const reco::Candidate* thisPart, float dR2Max) {
  if (p == thisPart)
    return false;
  if (p->status() != 1)
    return false;
  // exclude neutral particles
  if (p->charge() == 0)
    return false;

  // no collisionId info for PackedGenParticle
  if (!doPackedGenParticle_) {
    if ((dynamic_cast<const reco::GenParticle&>(*thisPart)).collisionId() !=
        (dynamic_cast<const reco::GenParticle&>(*p)).collisionId())
      return false;
  }

  // must be within deltaR2 cone
  float dR2 = reco::deltaR2(thisPart->momentum(), p->momentum());
  if (dR2 > dR2Max)
    return false;

  return true;
}

void ggHiNtuplizer::fillSC(edm::Event const& e) {
  edm::Handle<std::vector<reco::SuperCluster>> scHandle;
  e.getByToken(scToken_, scHandle);

  for (auto const& sc : *scHandle) {
    scE_.push_back(sc.energy());
    scRawE_.push_back(sc.rawEnergy());
    scEta_.push_back(sc.eta());
    scPhi_.push_back(sc.phi());

    ++nSC_;
  }
}

void ggHiNtuplizer::fillElectrons(const edm::Event& e, const edm::EventSetup& es, reco::Vertex& pv) {
  // Fills tree branches with electrons.
  edm::Handle<edm::View<pat::Electron>> gsfElectrons;
  e.getByToken(electronsToken_, gsfElectrons);

  edm::Handle<reco::ConversionCollection> conversions;
  e.getByToken(conversionsToken_, conversions);

  edm::Handle<reco::BeamSpot> beamSpot;
  e.getByToken(beamSpotToken_, beamSpot);

  // loop over electrons
  for (auto ele = gsfElectrons->begin(); ele != gsfElectrons->end(); ++ele) {
    eleD0_.push_back(ele->gsfTrack()->dxy(pv.position()));
    eleDz_.push_back(ele->gsfTrack()->dz(pv.position()));
    eleD0Err_.push_back(ele->gsfTrack()->dxyError());
    eleDzErr_.push_back(ele->gsfTrack()->dzError());
    eleTrkPt_.push_back(ele->gsfTrack()->pt());
    eleTrkEta_.push_back(ele->gsfTrack()->eta());
    eleTrkPhi_.push_back(ele->gsfTrack()->phi());
    eleTrkCharge_.push_back(ele->gsfTrack()->charge());
    eleTrkPtErr_.push_back(ele->gsfTrack()->ptError());
    eleTrkChi2_.push_back(ele->gsfTrack()->chi2());
    eleTrkNdof_.push_back(ele->gsfTrack()->ndof());
    eleTrkNormalizedChi2_.push_back(ele->gsfTrack()->normalizedChi2());
    eleTrkValidHits_.push_back(ele->gsfTrack()->numberOfValidHits());
    eleTrkLayers_.push_back(ele->gsfTrack()->hitPattern().trackerLayersWithMeasurement());
    eleMissHits_.push_back(ele->gsfTrack()->numberOfLostHits());

    elePt_.push_back(ele->pt());
    eleRawPt_.push_back(ele->hasUserFloat("rawPt") ? ele->userFloat("rawPt") : ele->pt());
    eleEta_.push_back(ele->eta());
    elePhi_.push_back(ele->phi());
    eleCharge_.push_back(ele->charge());
    eleEn_.push_back(ele->energy());

    eleMVAIso_.push_back(ele->hasUserFloat("hiMVAIso") ? ele->userFloat("hiMVAIso") : -99);
    for (auto& w : eleMVAIsoWP_)
      w.second.push_back(ele->hasUserInt("hiMVAIso"+w.first) ? ele->userInt("hiMVAIso"+w.first) : -1);
    eleMVAId_.push_back(ele->hasUserFloat("hiMVAId") ? ele->userFloat("hiMVAId") : -99);
    for (auto& w : eleMVAIdWP_)
      w.second.push_back(ele->hasUserInt("hiMVAId"+w.first) ? ele->userInt("hiMVAId"+w.first) : -1);
    for (auto& w : eleCutIdWP_)
      w.second.push_back(ele->hasUserInt("hiCutId"+w.first) ? ele->userInt("hiCutId"+w.first) : -1);

    eleSCEn_.push_back(ele->superCluster()->energy());
    eleESEn_.push_back(ele->superCluster()->preshowerEnergy());
    eleSCEta_.push_back(ele->superCluster()->eta());
    eleSCPhi_.push_back(ele->superCluster()->phi());
    eleSCRawEn_.push_back(ele->superCluster()->rawEnergy());
    eleSCEtaWidth_.push_back(ele->superCluster()->etaWidth());
    eleSCPhiWidth_.push_back(ele->superCluster()->phiWidth());
    eleSCClustersSize_.push_back(ele->superCluster()->clustersSize());
    eleSeedEn_.push_back(ele->superCluster()->seed()->energy());
    eleSeedEta_.push_back(ele->superCluster()->seed()->eta());
    eleSeedPhi_.push_back(ele->superCluster()->seed()->phi());

    eleHoverE_.push_back(ele->hcalOverEcal());
    eleHoverEBc_.push_back(ele->hcalOverEcalBc());
    eleRawHoverE_.push_back(ele->full5x5_hcalOverEcal());
    eleRawHoverEBc_.push_back(ele->full5x5_hcalOverEcalBc());
    eleEoverP_.push_back(ele->eSuperClusterOverP());
    eleEoverPInv_.push_back(1. / ele->ecalEnergy() - 1. / ele->trackMomentumAtVtx().R());
    eleEcalE_.push_back(ele->ecalEnergy());
    eleRawEcalE_.push_back(ele->hasUserFloat("rawEcalEnergy") ? ele->userFloat("rawEcalEnergy") : ele->ecalEnergy());
    elePAtVtx_.push_back(ele->trackMomentumAtVtx().R());
    elePAtSC_.push_back(ele->trackMomentumAtCalo().R());
    elePAtCluster_.push_back(ele->trackMomentumAtEleClus().R());
    elePAtSeed_.push_back(ele->trackMomentumOut().R());
    eledEtaAtVtx_.push_back(ele->deltaEtaSuperClusterTrackAtVtx());
    eledPhiAtVtx_.push_back(ele->deltaPhiSuperClusterTrackAtVtx());
    eledEtaSeedAtVtx_.push_back(ele->deltaEtaSeedClusterTrackAtVtx());
    eleSigmaIEtaIEta_.push_back(ele->sigmaIetaIeta());
    eleSigmaIPhiIPhi_.push_back(ele->sigmaIphiIphi());
    eleBrem_.push_back(ele->fbrem());

    /* updated in CMSSW_10_6_X */
    bool passConvVeto = !ConversionTools::hasMatchedConversion(*ele, *conversions, beamSpot->position());
    eleConvVeto_.push_back((int)passConvVeto);

    // full 5x5
    eleR9_.push_back(ele->r9());
    eleE3x3_.push_back(ele->r9() * ele->superCluster()->energy());
    eleE5x5_.push_back(ele->e5x5());
    eleR9Full5x5_.push_back(ele->full5x5_r9());
    eleE3x3Full5x5_.push_back(ele->full5x5_r9() * ele->superCluster()->energy());
    eleE5x5Full5x5_.push_back(ele->full5x5_e5x5());
    eleSigmaIEtaIEta_2012_.push_back(ele->full5x5_sigmaIetaIeta());

    // isolation
    reco::GsfElectron::PflowIsolationVariables pfIso = ele->pfIsolationVariables();
    elePFChIso_.push_back(pfIso.sumChargedHadronPt);
    elePFPhoIso_.push_back(pfIso.sumPhotonEt);
    elePFNeuIso_.push_back(pfIso.sumNeutralHadronEt);
    elePFPUIso_.push_back(pfIso.sumPUPt);

    if (doEffectiveAreas_) {
      double area = effectiveAreas_.getEffectiveArea(ele->superCluster()->eta());
      elePFRelIsoWithEA_.push_back(
          (pfIso.sumChargedHadronPt + std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - rho_ * area)) /
          ele->pt());
      elePFRelIsoWithDBeta_.push_back(
          (pfIso.sumChargedHadronPt +
           std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt)) /
          ele->pt());
      eleEffAreaTimesRho_.push_back(area * rho_);
    }

    // initialize with unphysical values
    float eleIP3D = -999;
    float eleIP3DErr = -999;
    if (pv.isValid()) {
      // 3D impact parameter
      reco::TransientTrack tt = tb->build(ele->gsfTrack().get());
      eleIP3D = IPTools::absoluteImpactParameter3D(tt, pv).second.value();
      eleIP3DErr = IPTools::absoluteImpactParameter3D(tt, pv).second.error();
    }

    eleIP3D_.push_back(eleIP3D);
    eleIP3DErr_.push_back(eleIP3DErr);

    // calculation on the fly
    if (doPfIso_) {
      pfIsoCalculator pfIsoCal;
      pfIsoCal.setUsePackedCandidates(isPackedPFCandidate_);
      pfIsoCal.setVertex(pv.position());
      if (isPackedPFCandidate_) {
        edm::Handle<edm::View<pat::PackedCandidate>> candidatesPacked;
        e.getByToken(packedPFCandToken_, candidatesPacked);
        pfIsoCal.setCandidatesPacked(candidatesPacked);
      }
      /*
      else {
        edm::Handle<edm::View< reco::PFCandidate >> candidatesReco;
        e.getByToken(recoPFCandToken_, candidatesReco);
        pfIsoCal.setCandidatesReco(candidatesReco);
      }
      */
      if (std::abs(ele->superCluster()->eta()) > 1.566) {
        elePFChIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h, 0.3, 0.015, 0.));
        elePFChIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h, 0.4, 0.015, 0.));
        elePFPhoIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::gamma, 0.3, 0.08, 0.));
        elePFPhoIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::gamma, 0.4, 0.08, 0.));
      } else {
        elePFChIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h, 0.3, 0.0, 0.));
        elePFChIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h, 0.4, 0.0, 0.));
        elePFPhoIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::gamma, 0.3, 0.0, 0.));
        elePFPhoIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::gamma, 0.4, 0.0, 0.));
      }

      elePFNeuIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h0, 0.3, 0., 0.));
      elePFNeuIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h0, 0.4, 0., 0.));
    }

    // local coordinates
    edm::Ptr<reco::CaloCluster> theseed = ele->superCluster()->seed();
    auto subdetid = theseed->hitsAndFractions().at(0).first.subdetId();

    /* updated in CMSSW_10_4_X */
    //EcalClusterLocal local;

    /* x, y instead of eta, phi in the endcap */
    float eta, phi, thetatilt, phitilt;
    int ieta, iphi;

    if (subdetid == EcalBarrel) {
      //local.localCoordsEB(*theseed, es, eta, phi, ieta, iphi, thetatilt, phitilt);
      egammaTools::localEcalClusterCoordsEB(*theseed, *geo, eta, phi, ieta, iphi, thetatilt, phitilt);
    } else {
      //local.localCoordsEE(*theseed, es, eta, phi, ieta, iphi, thetatilt, phitilt);
      egammaTools::localEcalClusterCoordsEE(*theseed, *geo, eta, phi, ieta, iphi, thetatilt, phitilt);
    }

    eleSeedCryEta_.push_back(eta);
    eleSeedCryPhi_.push_back(phi);
    eleSeedCryIeta_.push_back(ieta);
    eleSeedCryIphi_.push_back(iphi);

    /////////////////////////////// MC matching //////////////////////////
    if (doGenParticles_) {
      float minDeltaR2(0.0225);
      int matchedIndex = -1;
      for (size_t igen = 0; igen < mcEt_.size(); ++igen) {
        if (mcStatus_[igen] != 1 || mcPID_[igen] != -11*ele->charge())
          continue;
        const auto deltaR2 = reco::deltaR2(ele->eta(), ele->phi(), mcEta_[igen], mcPhi_[igen]);
        if (deltaR2 < minDeltaR2 && std::abs(mcPt_[igen] - ele->pt()) < 2*mcPt_[igen]) {
          minDeltaR2 = deltaR2;
          matchedIndex = igen;
        }
      }
      ele_genMatchedIndex_.push_back(matchedIndex);
    }

    ++nEle_;
  }  // electrons loop
}

void ggHiNtuplizer::fillPhotons(const edm::Event& e, const edm::EventSetup& es, reco::Vertex& pv) {
  edm::Handle<edm::View<pat::Photon>> patPhotons;
  e.getByToken(photonsToken_, patPhotons);

  edm::Handle<EcalRecHitCollection> recHitsEBHandle;
  if (doRecHitsEB_) {
    e.getByToken(recHitsEB_, recHitsEBHandle);
  }
  edm::Handle<EcalRecHitCollection> recHitsEEHandle;
  if (doRecHitsEE_) {
    e.getByToken(recHitsEE_, recHitsEEHandle);
  }

  // loop over photons
  for (edm::View<pat::Photon>::const_iterator pho = patPhotons->begin(); pho != patPhotons->end(); ++pho) {
    //const reco::Photon * pho = dynamic_cast<const reco::Photon *>(pPho->originalObject());
    //edm::Ptr<reco::Candidate> ptr(pho.originalObjectRef());

    phoE_.push_back(pho->energy());
    phoEt_.push_back(pho->et());
    phoEta_.push_back(pho->eta());
    phoPhi_.push_back(pho->phi());

    // energies from different types of corrections
    phoEcorrStdEcal_.push_back(pho->getCorrectedEnergy(reco::Photon::P4type::ecal_standard));
    phoEcorrPhoEcal_.push_back(pho->getCorrectedEnergy(reco::Photon::P4type::ecal_photons));
    phoEcorrRegr1_.push_back(pho->getCorrectedEnergy(reco::Photon::P4type::regression1));
    phoEcorrRegr2_.push_back(pho->getCorrectedEnergy(reco::Photon::P4type::regression2));
    // errors for those corrections
    phoEcorrErrStdEcal_.push_back(pho->getCorrectedEnergyError(reco::Photon::P4type::ecal_standard));
    phoEcorrErrPhoEcal_.push_back(pho->getCorrectedEnergyError(reco::Photon::P4type::ecal_photons));
    phoEcorrErrRegr1_.push_back(pho->getCorrectedEnergyError(reco::Photon::P4type::regression1));
    phoEcorrErrRegr2_.push_back(pho->getCorrectedEnergyError(reco::Photon::P4type::regression2));

    // SuperCluster info
    phoSCE_.push_back(pho->superCluster()->energy());
    phoSCRawE_.push_back(pho->superCluster()->rawEnergy());
    phoSCEta_.push_back(pho->superCluster()->eta());
    phoSCPhi_.push_back(pho->superCluster()->phi());
    phoSCEtaWidth_.push_back(pho->superCluster()->etaWidth());
    phoSCPhiWidth_.push_back(pho->superCluster()->phiWidth());
    phoSCBrem_.push_back(pho->superCluster()->phiWidth() / pho->superCluster()->etaWidth());
    phoSCnHits_.push_back(pho->superCluster()->size());
    phoSCflags_.push_back(pho->superCluster()->flags());
    phoSCinClean_.push_back((int)pho->superCluster()->isInClean());
    phoSCinUnClean_.push_back((int)pho->superCluster()->isInUnclean());
    phoSCnBC_.push_back((int)pho->superCluster()->clustersSize());
    phoESEn_.push_back(pho->superCluster()->preshowerEnergy());

    phoIsPFPhoton_.push_back((int)pho->isPFlowPhoton());
    phoIsStandardPhoton_.push_back((int)pho->isStandardPhoton());
    phoHasPixelSeed_.push_back((int)pho->hasPixelSeed());
    phoHasConversionTracks_.push_back((int)pho->hasConversionTracks());
    phoHadTowerOverEm_.push_back(pho->hadTowOverEm());
    phoHoverE_.push_back(pho->hadronicOverEm());
    phoHoverEValid_.push_back(pho->hadronicOverEmValid());
    phoSigmaIEtaIEta_.push_back(pho->sigmaIetaIeta());
    phoR9_.push_back(pho->r9());

    // additional shower shape variables
    phoE3x3_.push_back(pho->e3x3());
    phoE1x5_.push_back(pho->e1x5());
    phoE2x5_.push_back(pho->e2x5());
    phoE5x5_.push_back(pho->e5x5());
    phoMaxEnergyXtal_.push_back(pho->maxEnergyXtal());
    phoSigmaEtaEta_.push_back(pho->sigmaEtaEta());

    if (doPhoEReg_) {
      phoHadTowerOverEm1_.push_back(pho->hadTowOverEm(1));
      phoHadTowerOverEm2_.push_back(pho->hadTowOverEm(2));
      phoHoverE1_.push_back(pho->hadronicOverEm(1));
      phoHoverE2_.push_back(pho->hadronicOverEm(2));

      phoSigmaIEtaIPhi_.push_back(pho->showerShapeVariables().sigmaIetaIphi);
      phoSigmaIPhiIPhi_.push_back(pho->showerShapeVariables().sigmaIphiIphi);
      phoR1x5_.push_back(pho->r1x5());
      phoR2x5_.push_back(pho->r2x5());
      phoE2nd_.push_back(pho->showerShapeVariables().e2nd);
      phoETop_.push_back(pho->showerShapeVariables().eTop);
      phoEBottom_.push_back(pho->showerShapeVariables().eBottom);
      phoELeft_.push_back(pho->showerShapeVariables().eLeft);
      phoERight_.push_back(pho->showerShapeVariables().eRight);
      phoE1x3_.push_back(pho->showerShapeVariables().e1x3);
      phoE2x2_.push_back(pho->showerShapeVariables().e2x2);
      phoE2x5Max_.push_back(pho->showerShapeVariables().e2x5Max);
      phoE2x5Top_.push_back(pho->showerShapeVariables().e2x5Top);
      phoE2x5Bottom_.push_back(pho->showerShapeVariables().e2x5Bottom);
      phoE2x5Left_.push_back(pho->showerShapeVariables().e2x5Left);
      phoE2x5Right_.push_back(pho->showerShapeVariables().e2x5Right);
      //phoSMMajor_.push_back(pho->showerShapeVariables().smMajor);
      //phoSMMinor_.push_back(pho->showerShapeVariables().smMinor);
      //phoSMAlpha_.push_back(pho->showerShapeVariables().smAlpha);

      phoSigmaIEtaIPhi_2012_.push_back(pho->full5x5_showerShapeVariables().sigmaIetaIphi);
      phoSigmaIPhiIPhi_2012_.push_back(pho->full5x5_showerShapeVariables().sigmaIphiIphi);
      phoR1x5_2012_.push_back(pho->full5x5_r1x5());
      phoR2x5_2012_.push_back(pho->full5x5_r2x5());
      phoE2nd_2012_.push_back(pho->full5x5_showerShapeVariables().e2nd);
      phoETop_2012_.push_back(pho->full5x5_showerShapeVariables().eTop);
      phoEBottom_2012_.push_back(pho->full5x5_showerShapeVariables().eBottom);
      phoELeft_2012_.push_back(pho->full5x5_showerShapeVariables().eLeft);
      phoERight_2012_.push_back(pho->full5x5_showerShapeVariables().eRight);
      phoE1x3_2012_.push_back(pho->full5x5_showerShapeVariables().e1x3);
      phoE2x2_2012_.push_back(pho->full5x5_showerShapeVariables().e2x2);
      phoE2x5Max_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Max);
      phoE2x5Top_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Top);
      phoE2x5Bottom_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Bottom);
      phoE2x5Left_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Left);
      phoE2x5Right_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Right);
      //phoSMMajor_2012_.push_back(pho->full5x5_showerShapeVariables().smMajor);
      //phoSMMinor_2012_.push_back(pho->full5x5_showerShapeVariables().smMinor);
      //phoSMAlpha_2012_.push_back(pho->full5x5_showerShapeVariables().smAlpha);
    }

    // full 5x5
    phoSigmaIEtaIEta_2012_.push_back(pho->full5x5_sigmaIetaIeta());
    phoR9_2012_.push_back(pho->full5x5_r9());
    phoE1x5_2012_.push_back(pho->full5x5_e1x5());
    phoE2x5_2012_.push_back(pho->full5x5_e2x5());
    phoE3x3_2012_.push_back(pho->full5x5_e3x3());
    phoE5x5_2012_.push_back(pho->full5x5_e5x5());
    phoMaxEnergyXtal_2012_.push_back(pho->full5x5_maxEnergyXtal());
    phoSigmaEtaEta_2012_.push_back(pho->full5x5_sigmaEtaEta());

    // seed BC
    if (pho->superCluster()->seed().isAvailable() && pho->superCluster()->seed().isNonnull()) {
      phoBC1E_.push_back(pho->superCluster()->seed()->energy());
      phoBC1Ecorr_.push_back(pho->superCluster()->seed()->correctedEnergy());
      phoBC1Eta_.push_back(pho->superCluster()->seed()->eta());
      phoBC1Phi_.push_back(pho->superCluster()->seed()->phi());
      phoBC1size_.push_back(pho->superCluster()->seed()->size());
      phoBC1flags_.push_back(pho->superCluster()->seed()->flags());
      phoBC1inClean_.push_back(pho->superCluster()->seed()->isInClean());
      phoBC1inUnClean_.push_back(pho->superCluster()->seed()->isInUnclean());
      phoBC1rawID_.push_back(pho->superCluster()->seed()->seed().rawId());
    } else {
      phoBC1E_.push_back(-999);
      phoBC1Ecorr_.push_back(-999);
      phoBC1Eta_.push_back(-999);
      phoBC1Phi_.push_back(-999);
      phoBC1size_.push_back(-999);
      phoBC1flags_.push_back(-999);
      phoBC1inClean_.push_back(-999);
      phoBC1inUnClean_.push_back(-999);
      phoBC1rawID_.push_back(0);
    }

    bool debugPhoUserDataNames = false;
    if (debugPhoUserDataNames) {
      std::cout << "debugging photon user data names" << std::endl;

      // https://github.com/cms-sw/cmssw/blob/master/DataFormats/PatCandidates/interface/PATObject.h#L334
      const std::vector<std::string> namesTmp = pho->userDataNames();
      int lenNames = namesTmp.size();
      std::cout << "There are " << lenNames << " names." << std::endl;
      for (int iName = 0; iName < lenNames; ++iName) {
        std::cout << "iName = " << iName << " , nameTmp = " << namesTmp[iName].c_str() << std::endl;
        std::cout << "typename = " << pho->userDataObjectType(namesTmp[iName]) << std::endl;
      }
    }

    if (useValMapIso_) {
      const reco::HIPhotonIsolation* hiIso = pho->userData<reco::HIPhotonIsolation>(std::string("photonIsolationHI"));

      pho_ecalClusterIsoR1_.push_back(hiIso->ecalClusterIsoR1());
      pho_ecalClusterIsoR2_.push_back(hiIso->ecalClusterIsoR2());
      pho_ecalClusterIsoR3_.push_back(hiIso->ecalClusterIsoR3());
      pho_ecalClusterIsoR4_.push_back(hiIso->ecalClusterIsoR4());
      pho_ecalClusterIsoR5_.push_back(hiIso->ecalClusterIsoR5());
      pho_hcalRechitIsoR1_.push_back(hiIso->hcalRechitIsoR1());
      pho_hcalRechitIsoR2_.push_back(hiIso->hcalRechitIsoR2());
      pho_hcalRechitIsoR3_.push_back(hiIso->hcalRechitIsoR3());
      pho_hcalRechitIsoR4_.push_back(hiIso->hcalRechitIsoR4());
      pho_hcalRechitIsoR5_.push_back(hiIso->hcalRechitIsoR5());
      pho_trackIsoR1PtCut20_.push_back(hiIso->trackIsoR1PtCut20());
      pho_trackIsoR2PtCut20_.push_back(hiIso->trackIsoR2PtCut20());
      pho_trackIsoR3PtCut20_.push_back(hiIso->trackIsoR3PtCut20());
      pho_trackIsoR4PtCut20_.push_back(hiIso->trackIsoR4PtCut20());
      pho_trackIsoR5PtCut20_.push_back(hiIso->trackIsoR5PtCut20());
      pho_swissCrx_.push_back(hiIso->swissCrx());
      pho_seedTime_.push_back(hiIso->seedTime());
    }

    if (doRecHitsEB_ || doRecHitsEE_) {
      int iBC = 0;
      for (reco::CaloCluster_iterator bc = pho->superCluster()->clustersBegin();
           bc != pho->superCluster()->clustersEnd();
           ++bc) {
        for (const auto& hitBC : (*bc)->hitsAndFractions()) {
          const DetId& rhDetId = hitBC.first;
          edm::Handle<EcalRecHitCollection>* recHitsHandle;

          if (rhDetId.subdetId() == EcalBarrel && doRecHitsEB_) {
            recHitsHandle = (&recHitsEBHandle);
          } else if (rhDetId.subdetId() == EcalEndcap && doRecHitsEE_) {
            recHitsHandle = (&recHitsEEHandle);
          } else
            continue;

          EcalRecHitCollection::const_iterator rhIter = (*recHitsHandle)->find(rhDetId);
          const EcalRecHit& rh = (*rhIter);

          rhRawId_.push_back(rhDetId.rawId());
          if (rhDetId.subdetId() == EcalBarrel) {
            rhieta_.push_back(EBDetId(rhDetId).ieta());
            rhiphi_.push_back(EBDetId(rhDetId).iphi());
            rhix_.push_back(-999);
            rhiy_.push_back(-999);
          } else if (rhDetId.subdetId() == EcalEndcap) {
            rhieta_.push_back(-999);
            rhiphi_.push_back(-999);
            rhix_.push_back(EEDetId(rhDetId).ix());
            rhiy_.push_back(EEDetId(rhDetId).iy());
          } else {
            rhieta_.push_back(-999);
            rhiphi_.push_back(-999);
            rhix_.push_back(-999);
            rhiy_.push_back(-999);
          }
          rhE_.push_back(rh.energy());

          math::XYZPoint posXYZ((geo->getPosition(rhDetId)).x() - pv.x(),
                                (geo->getPosition(rhDetId)).y() - pv.y(),
                                (geo->getPosition(rhDetId)).z() - pv.z());

          rhEt_.push_back(rh.energy() * sin(posXYZ.theta()));
          rhEta_.push_back(posXYZ.eta());
          rhPhi_.push_back(posXYZ.phi());
          rhChi2_.push_back(rh.chi2());
          rhEerror_.push_back(rh.energyError());

          uint32_t flagTmp = 0;
          for (unsigned int iFlag = 0; iFlag < 32; ++iFlag) {
            if (rh.checkFlag(iFlag))
              flagTmp |= 1 << iFlag;
          }
          rhFlags_.push_back(flagTmp);

          rhPhoIdx_.push_back(nPho_);
          rhBCIdx_.push_back(iBC);
          nRH_++;
        }
        iBC++;
      }
    }

    if (doPfIso_) {
      pfIsoCalculator pfIso;
      pfIso.setUsePackedCandidates(isPackedPFCandidate_);
      pfIso.setVertex(pv.position());
      if (isPackedPFCandidate_) {
        edm::Handle<edm::View<pat::PackedCandidate>> candidatesPacked;
        e.getByToken(packedPFCandToken_, candidatesPacked);
        pfIso.setCandidatesPacked(candidatesPacked);
      }
      /*
      else {
        edm::Handle<edm::View< reco::PFCandidate >> candidatesReco;
        e.getByToken(recoPFCandToken_, candidatesReco);
        pfIso.setCandidatesReco(candidatesReco);
      }
      */

      bool debug = false;
      if (debug) {
        // https://github.com/cms-sw/cmssw/blob/master/DataFormats/PatCandidates/interface/Photon.h#L326
        std::cout << "pho numberOfSourceCandidatePtrs = " << pho->numberOfSourceCandidatePtrs() << std::endl;
      }

      // particle flow isolation
      pfcIso1_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfpIso1_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso2_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso3_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso4_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso5_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfnIso1_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso2_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso3_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso4_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso5_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfpIso1subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso2subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso3subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso4subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso5subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));

      pfcIso1subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfpIso1subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso2subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso3subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso4subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso5subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfnIso1subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso2subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso3subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso4subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso5subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfpIso1subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso2subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso3subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso4subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso5subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));

      pfcIso1pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.1, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.5, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));

      pfcIso1pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.1, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.5, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));

      pfcIso1pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.1, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.5, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));

      pfcIso2subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfcIso3subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfcIso4subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));

      pfpIso2subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfpIso3subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfpIso4subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));

      pfnIso2subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfnIso3subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfnIso4subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));

      pfcIso2pTgt2p0subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 2.0, 0, pfIsoCalculator::removePFcand, true));
      pfcIso3pTgt2p0subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 2.0, 0, pfIsoCalculator::removePFcand, true));
      pfcIso4pTgt2p0subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 2.0, 0, pfIsoCalculator::removePFcand, true));
    }

    /////////////////////////////// MC matching //////////////////////////
    if (doGenParticles_) {
      constexpr float delta2 = 0.15 * 0.15;

      bool gpTemp = false;
      float currentMaxPt = -1;
      int matchedIndex = -1;

      for (unsigned igen = 0; igen < mcEt_.size(); ++igen) {
        if (mcStatus_[igen] != 1 || mcPID_[igen] != 22)
          continue;
        if (reco::deltaR2(pho->eta(), pho->phi(), mcEta_[igen], mcPhi_[igen]) < delta2 && mcPt_[igen] > currentMaxPt) {
          gpTemp = true;
          currentMaxPt = mcPt_[igen];
          matchedIndex = igen;
        }
      }

      // if no matching photon was found try with other particles
      std::vector<int> otherPdgIds_ = {1, 11};
      if (!gpTemp) {
        currentMaxPt = -1;
        for (unsigned igen = 0; igen < mcEt_.size(); ++igen) {
          if (mcStatus_[igen] != 1 ||
              find(otherPdgIds_.begin(), otherPdgIds_.end(), std::abs(mcPID_[igen])) == otherPdgIds_.end())
            continue;
          if (reco::deltaR2(pho->eta(), pho->phi(), mcEta_[igen], mcPhi_[igen]) < delta2 &&
              mcPt_[igen] > currentMaxPt) {
            gpTemp = true;
            currentMaxPt = mcPt_[igen];
            matchedIndex = igen;
          }
        }
      }

      pho_genMatchedIndex_.push_back(matchedIndex);
    }

    nPho_++;
  }  // photons loop
}

void ggHiNtuplizer::fillMuons(const edm::Event& e, const edm::EventSetup& es, reco::Vertex& pv) {
  // Fills tree branches with muons.
  edm::Handle<edm::View<pat::Muon>> recoMuons;
  e.getByToken(muonsToken_, recoMuons);

  for (const auto& mu : *recoMuons) {
    if (mu.pt() < muonPtMin_)
      continue;
    if (!(mu.isPFMuon() || mu.isGlobalMuon() || mu.isTrackerMuon()))
      continue;

    muPt_.push_back(mu.pt());
    muEta_.push_back(mu.eta());
    muPhi_.push_back(mu.phi());
    muL1Eta_.push_back(mu.hasUserFloat("l1Eta") ? mu.userFloat("l1Eta") : -99);
    muL1Phi_.push_back(mu.hasUserFloat("l1Phi") ? mu.userFloat("l1Phi") : -99);
    muMVAIso_.push_back(mu.hasUserFloat("hiMVAIso") ? mu.userFloat("hiMVAIso") : -99);
    for (auto& w : muMVAIsoWP_)
      w.second.push_back(mu.hasUserInt("hiMVAIso"+w.first) ? mu.userInt("hiMVAIso"+w.first) : -1);
    muCharge_.push_back(mu.charge());
    muType_.push_back(mu.type());
    muIsGood_.push_back(muon::isGoodMuon(mu, muon::selectionTypeFromString("TMOneStationTight")));

    muIsGlobal_.push_back((int)mu.isGlobalMuon());
    muIsTracker_.push_back((int)mu.isTrackerMuon());
    muIsPF_.push_back((int)mu.isPFMuon());
    muIsSTA_.push_back((int)mu.isStandAloneMuon());

    muD0_.push_back(mu.muonBestTrack()->dxy(pv.position()));
    muDz_.push_back(mu.muonBestTrack()->dz(pv.position()));
    muD0Err_.push_back(mu.muonBestTrack()->dxyError());
    muDzErr_.push_back(mu.muonBestTrack()->dzError());

    // initialize with unphysical values
    float muIP3D = -999;
    float muIP3DErr = -999;
    if (pv.isValid()) {
      // 3D impact parameter
      reco::TransientTrack tt = tb->build(mu.muonBestTrack().get());
      muIP3D = IPTools::absoluteImpactParameter3D(tt, pv).second.value();
      muIP3DErr = IPTools::absoluteImpactParameter3D(tt, pv).second.error();
    }
    muIP3D_.push_back(muIP3D);
    muIP3DErr_.push_back(muIP3DErr);

    const reco::TrackRef glbMu = mu.globalTrack();
    const reco::TrackRef innMu = mu.innerTrack();

    if (glbMu.isNull()) {
      muChi2NDF_.push_back(-99);
      muMuonHits_.push_back(-99);
    } else {
      muChi2NDF_.push_back(glbMu->normalizedChi2());
      muMuonHits_.push_back(glbMu->hitPattern().numberOfValidMuonHits());
    }

    if (innMu.isNull()) {
      muInnerD0_.push_back(-99);
      muInnerDz_.push_back(-99);

      muInnerD0Err_.push_back(-99);
      muInnerDzErr_.push_back(-99);
      muInnerPt_.push_back(-99);
      muInnerPtErr_.push_back(-99);
      muInnerEta_.push_back(-99);

      muTrkLayers_.push_back(-99);
      muPixelLayers_.push_back(-99);
      muPixelHits_.push_back(-99);
      muTrkQuality_.push_back(-99);
    } else {
      muInnerD0_.push_back(innMu->dxy(pv.position()));
      muInnerDz_.push_back(innMu->dz(pv.position()));

      muInnerD0Err_.push_back(innMu->dxyError());
      muInnerDzErr_.push_back(innMu->dzError());
      muInnerPt_.push_back(innMu->pt());
      muInnerPtErr_.push_back(innMu->ptError());
      muInnerEta_.push_back(innMu->eta());

      muTrkLayers_.push_back(innMu->hitPattern().trackerLayersWithMeasurement());
      muPixelLayers_.push_back(innMu->hitPattern().pixelLayersWithMeasurement());
      muPixelHits_.push_back(innMu->hitPattern().numberOfValidPixelHits());
      muTrkQuality_.push_back(innMu->quality(reco::TrackBase::highPurity));
    }

    muStations_.push_back(mu.numberOfMatchedStations());
    muIsoTrk_.push_back(mu.isolationR03().sumPt);
    muPFChIso_.push_back(mu.pfIsolationR04().sumChargedHadronPt);
    muPFPhoIso_.push_back(mu.pfIsolationR04().sumPhotonEt);
    muPFNeuIso_.push_back(mu.pfIsolationR04().sumNeutralHadronEt);
    muPFPUIso_.push_back(mu.pfIsolationR04().sumPUPt);

    muSelectors_.push_back(mu.selectors());
    muIDSoft_.push_back(mu.passed(reco::Muon::SoftMvaId));
    muIDLoose_.push_back(mu.passed(reco::Muon::CutBasedIdLoose));
    muIDMedium_.push_back(mu.passed(reco::Muon::CutBasedIdMedium));
    muIDMediumPrompt_.push_back(mu.passed(reco::Muon::CutBasedIdMediumPrompt));
    muIDTight_.push_back(mu.passed(reco::Muon::CutBasedIdTight));
    muIDGlobalHighPt_.push_back(mu.passed(reco::Muon::CutBasedIdGlobalHighPt));
    muIDTrkHighPt_.push_back(mu.passed(reco::Muon::CutBasedIdTrkHighPt));
    muIDInTime_.push_back(mu.passed(reco::Muon::InTimeMuon));

    /////////////////////////////// MC matching //////////////////////////
    if (doGenParticles_) {
      float minDeltaR2(0.0225);
      int matchedIndex = -1;
      for (size_t igen = 0; igen < mcEt_.size(); ++igen) {
        if (mcStatus_[igen] != 1 || mcPID_[igen] != -13*mu.charge())
          continue;
        const auto deltaR2 = reco::deltaR2(mu.eta(), mu.phi(), mcEta_[igen], mcPhi_[igen]);
        if (deltaR2 < minDeltaR2 && std::abs(mcPt_[igen] - mu.pt()) < mcPt_[igen]) {
          minDeltaR2 = deltaR2;
          matchedIndex = igen;
        }
      }
      mu_genMatchedIndex_.push_back(matchedIndex);
    }

    nMu_++;
  }  // muons loop
}

DEFINE_FWK_MODULE(ggHiNtuplizer);
