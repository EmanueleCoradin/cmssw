#include <alpaka/alpaka.hpp>
#include <xtd/math/sqrt.h>
#include <xtd/math/asinh.h>
#include <xtd/math/atan2.h>
#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesExtractorKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PixelTrackFeaturesSoAView  = PixelTrackFeaturesSoA::View;
  using PixelRecHitFeaturesSoAView = RecHitFeatures::PixelRecHitFeaturesSoA::View;
  using TrackHitSoA  = ::reco::TrackHitSoA;
  using HitFeaturesIDX = RecHitFeatures::HitFeature;

    struct CAPreselectionKernel{
        // Kernel used to preselect tracks which were flagged as good enough by the CA
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxTracks,
            const int maxTracksPreselection,
            const int minNumberOfHits,
            const ::pixelTrack::Quality minQuality,
            const ::reco::TrackSoAConstView tracks,
            int* nKeptTracks,
            int* nKeptHits,
            int* oldIndex
        ) const {
            const int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            if(i==0){
                printf("PixelTrackFeaturesExtractorKernels: nTracks=%d\n", tracks.nTracks());
                assert(tracks.nTracks()<maxTracks);
            }

            if (i < tracks.nTracks() && 
                tracks[i].quality() >= minQuality && 
                nHits(tracks, i)>=minNumberOfHits)
            {
                int idx = alpaka::atomicOp<alpaka::AtomicAdd>(acc, nKeptTracks, 1);
                oldIndex[idx] = i;
                nKeptHits[idx] = nHits(tracks, i);
            }
        } 
    };

    struct FeaturesExtractorKernel{ 
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxTracksPreselection,
            const ::reco::TrackSoAConstView tracks,
            const ::reco::TrackHitSoAConstView track_hits,
            const ::reco::TrackingRecHitConstView hits,
            PixelTrackFeaturesSoAView trackFeatures,
            PixelRecHitFeaturesSoAView hitFeatures,
            int* nKeptTracks,
            int* oldIndex
        ) const {

            const int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            const float NaN = std::numeric_limits<float>::quiet_NaN();
        
            if (i < *nKeptTracks){
                int idx = oldIndex[i];
                assert(idx>=0);

                constexpr int cPhi = 0;
                constexpr int cTip = 5;
                constexpr int cInvPt = 9;
                constexpr int cZip = 14;
                
                const auto track = tracks[idx];
                const auto cov   = track.covariance();
                const auto state = track.state();
                const float pt   = track.pt();
                const int n_hits  = nHits(tracks, idx);
                const int ndof   = n_hits * 2 - 5;
                const float ptError = xtd::sqrt(cov(cInvPt)) * pt * pt;

                assert(n_hits<=RecHitFeatures::MaxHitsPerTrack);

                trackFeatures.chi2(i)     = track.chi2() * ndof;
                trackFeatures.dzError(i)  = xtd::sqrt(cov(cZip));
                trackFeatures.dxyError(i) = xtd::sqrt(cov(cTip));
                trackFeatures.eta(i)      = track.eta();
                trackFeatures.ndof(i)     = ndof;
                trackFeatures.phi(i)      = track.state()(0);
                trackFeatures.phiError(i) = xtd::sqrt(cov(cPhi));
                trackFeatures.pt(i)       = pt;
                trackFeatures.ptError(i)  = ptError;
                trackFeatures.qoverp(i)   = state(2);
                trackFeatures.dzBS(i)     = state(4);
                trackFeatures.dxyBS(i)    = state(1);

                //Prefill hit features:
                auto hitMatrix = hitFeatures.hits(i);
                for (int h = 0; h < RecHitFeatures::MaxHitsPerTrack; ++h) {
                    for (int f = 0; f < RecHitFeatures::HitFeatures; ++f) {
                        hitMatrix(h, f) = NaN;
                    }
                }

                uint32_t inStart = (idx == 0) ? 0 : tracks[idx - 1].hitOffsets();
                uint32_t inEnd   = track.hitOffsets();
                uint32_t nHits   = inEnd - inStart;

                nHits = std::min(nHits, uint32_t(RecHitFeatures::MaxHitsPerTrack));

                for (uint32_t h = 0; h < nHits; ++h) {
                    auto hit_id = track_hits[inStart + h].id();
                    const auto hit = hits[hit_id];
                    const float x = hit.xGlobal();
                    const float y = hit.yGlobal();
                    const float z = hit.zGlobal();
                    const float r = hit.rGlobal();

                    hitMatrix(h, HitFeaturesIDX::x) = x;
                    hitMatrix(h, HitFeaturesIDX::y) = y;
                    hitMatrix(h, HitFeaturesIDX::z) = z;

                    hitMatrix(h, HitFeaturesIDX::xErr) =
                        hit.xerrLocal();
                    hitMatrix(h, HitFeaturesIDX::yErr) =
                        hit.yerrLocal();

                    hitMatrix(h, HitFeaturesIDX::r) = r;                        
                    hitMatrix(h, HitFeaturesIDX::eta) =
                        (r > 0.f) ? xtd::asinh(z / r) : 0.f;
                    hitMatrix(h, HitFeaturesIDX::phi) =
                        xtd::atan2(y, x);
                }
            }
            else if (i < maxTracksPreselection)
            {
                oldIndex[i] = -1;
                trackFeatures.chi2(i)     = NaN;
                trackFeatures.dzError(i)  = NaN;
                trackFeatures.dxyError(i) = NaN;
                trackFeatures.eta(i)      = NaN;
                trackFeatures.ndof(i)     = -1;
                trackFeatures.phi(i)      = NaN;
                trackFeatures.phiError(i) = NaN;
                trackFeatures.pt(i)       = NaN;
                trackFeatures.ptError(i)  = NaN;
                trackFeatures.qoverp(i)   = NaN;
                trackFeatures.dzBS(i)     = NaN;
                trackFeatures.dxyBS(i)    = NaN;

                auto hitMatrix = hitFeatures.hits(i);
                for (int h = 0; h < RecHitFeatures::MaxHitsPerTrack; ++h) {
                    for (int f = 0; f < RecHitFeatures::HitFeatures; ++f) {
                        hitMatrix(h, f) = NaN;
                    }
                }
            }
        } 
    };

    struct PixelTrackFilterKernel{ 
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxTracksPreselection,
            const ::reco::TrackSoAConstView tracks,
            const ::reco::TrackHitSoAConstView track_hits,
            int* oldIndex,
            const int* nKeptTracks,
            const int* nKeptHits,
            ::reco::TrackSoAView tracks_out,
            ::reco::TrackHitSoAView track_hits_out
        ) const {

            const int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            const int nTracks = *nKeptTracks;

            if (i==0) {
                tracks_out.nTracks() = nTracks;
            }

            if (i < maxTracksPreselection){
                int idx = oldIndex[i];
                if(idx >= 0){
                    const auto track = tracks[idx];
                    tracks_out[i] = track;
                    tracks_out[i].hitOffsets() = nKeptHits[i];

                    //Access the hits associated to the track:
                    uint32_t inStart  = (idx == 0) ? 0 : tracks[idx-1].hitOffsets();
                    uint32_t inEnd    = track.hitOffsets();
                    uint32_t outStart = (i == 0) ? 0 :  nKeptHits[i-1];
                    
                    for (uint32_t h = 0; h < (inEnd - inStart); ++h) {
                        track_hits_out[outStart+h].id()    = track_hits[inStart + h].id();
                        track_hits_out[outStart+h].detId() = track_hits[inStart + h].detId();
                    }
                }
            }
        } 
    };

    struct HitOffsetCompactKernel{
        //TODO: implement the prefix scan provided my cmssw
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int  maxTracksPreselection,
            int* oldIndex,
            int* nKeptTracks,
            int* nKeptHits
        ) const 
        {
            int nTracks = 0;
            for (int j = 0; j < maxTracksPreselection; j++){
                if (oldIndex[j] != -1){
                    oldIndex[nTracks]  = oldIndex[j];
                    nKeptHits[nTracks] = nKeptHits[j];   
                    nTracks++;
                }
            }
            *nKeptTracks = nTracks;

            //loop over the tracks we kept and do the prefix sum
            for (int i = 1; i < nTracks; i++)
            {
                nKeptHits[i] += nKeptHits[i-1]; 
            }
            for (int i = nTracks; i < maxTracksPreselection; i++){
                nKeptHits[i] = nKeptHits[nTracks-1];
            }
        }
    };


    void launchCAPreselectionKernel(
        Queue& queue,
        const int maxTracks,
        const int maxTracksPreselection,
        const int minNumberOfHits,
        const ::pixelTrack::Quality minQuality,
        const ::reco::TrackSoAConstView tracks,
        int* nKeptTracks,
        int* nKeptHits,
        int* oldIndex)
    {
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(maxTracks, threadsPerBlock);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            CAPreselectionKernel{},
            maxTracks,
            maxTracksPreselection,
            minNumberOfHits,
            minQuality,
            tracks,
            nKeptTracks,
            nKeptHits,
            oldIndex
        );
    }

    void launchFeaturesExtractorKernel(
        Queue& queue,
        const int maxTracksPreselection,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        const ::reco::TrackingRecHitConstView hits,
        PixelTrackFeaturesSoAView trackFeatures,
        PixelRecHitFeaturesSoAView hitFeatures,
        int* nKeptTracks,
        int* oldIndex)
    {
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(maxTracksPreselection, threadsPerBlock);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            FeaturesExtractorKernel{},
            maxTracksPreselection,
            tracks,
            track_hits,
            hits,
            trackFeatures,
            hitFeatures,
            nKeptTracks,
            oldIndex
        );
    }

    void launchPixelTrackFilterKernel(
        Queue& queue,
        const int maxTracksPreselection,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        int* oldIndex,
        const int* nKeptTracks,
        const int* nKeptHits,
        ::reco::TrackSoAView tracks_out,
        ::reco::TrackHitSoAView track_hits_out)
    {
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(maxTracksPreselection, threadsPerBlock);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            PixelTrackFilterKernel{},
            maxTracksPreselection,
            tracks,
            track_hits,
            oldIndex,
            nKeptTracks,
            nKeptHits,
            tracks_out,
            track_hits_out
        );
    }

    reco::TracksSoACollection launchProduceOutputTracks(
        Queue& queue,
        const int maxTracksPreselection,
        const int avgHitsPerTrack,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        int* oldIndex,
        const int* nKeptTracks,
        const int* nKeptHits
    )
    {
        reco::TracksSoACollection tracks_out({{int(maxTracksPreselection), int(maxTracksPreselection * avgHitsPerTrack)}}, queue);
        
        /*
        const auto device = alpaka::getDev(queue);
        auto ntracks_d = cms::alpakatools::make_device_view(device, tracks_out.view().nTracks());
        alpaka::memset(queue, ntracks_d, 0);
        */

        launchPixelTrackFilterKernel(
            queue,
            maxTracksPreselection,
            tracks,
            track_hits,
            oldIndex,
            nKeptTracks,
            nKeptHits,
            tracks_out.view(),
            tracks_out.view<TrackHitSoA>()
        );

        return tracks_out;
    }

    void launchHitOffsetCompactKernel(
        Queue& queue,
        const int  maxTracksPreselection,
        int* oldIndex,
        int* nKeptTracks,
        int* nKeptHits)
    {
        constexpr uint32_t threadsPerBlock = 1;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(1, 1);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
        
        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            HitOffsetCompactKernel{},
            maxTracksPreselection,
            oldIndex,
            nKeptTracks,
            nKeptHits
        );
    }

    struct ScoreFilterKernel{
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxTracksPreselection,
            const double scoreThreshold,
            int* nKeptTracks,
            int* nKeptHits,
            int* oldIndex,
            const PixelTrackScoresSoA::View trackScores) const 
        {
            const int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            if (i < maxTracksPreselection){
                const float score = trackScores[i].score();
                //printf("Track %d: score=%f\n", i, score);   
                if (score < scoreThreshold && oldIndex[i] != -1){
                    oldIndex[i] = -1;
                }
            }
        }
    };

    void launchScoreFilterKernel(
        Queue& queue,
        const int maxTracksPreselection,
        const double scoreThreshold,
        int* nKeptTracks,
        int* nKeptHits,
        int* oldIndex,
        const PixelTrackScoresSoA::View trackScores
    ){
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(maxTracksPreselection, threadsPerBlock);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            ScoreFilterKernel{},
            maxTracksPreselection,
            scoreThreshold,
            nKeptTracks,
            nKeptHits,
            oldIndex,
            trackScores
        );
    }
}