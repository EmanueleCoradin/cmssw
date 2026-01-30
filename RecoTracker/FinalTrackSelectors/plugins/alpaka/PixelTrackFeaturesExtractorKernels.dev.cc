#include <alpaka/alpaka.hpp>

#include <xtd/math/asinh.h>
#include <xtd/math/atan2.h>
#include <xtd/math/sqrt.h>
#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesExtractorKernels.h"

#ifndef KERNELS_DEBUG
#define KERNELS_DEBUG
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PixelTrackFeaturesSoAView  = PixelTrackFeaturesSoA::View;
  using PixelRecHitFeaturesSoAView = RecHitFeatures::PixelRecHitFeaturesSoA::View;
  using TrackHitSoA  = ::reco::TrackHitSoA;
  using HitFeaturesIDX = RecHitFeatures::HitFeature;

    // Kernel to preselect tracks based on quality and number of hits
    struct CAPreselectionKernel{
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxNumberOfTracks,
            const int maxPreselectedTracks,
            const int minNumberOfHits,
            const ::pixelTrack::Quality minimumTrackQuality,
            const ::reco::TrackSoAConstView tracks,
            int* nKeptTracks,
            int* nKeptHits,
            int* originalTrackIndex) const 
        {
            const int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

#ifdef KERNELS_DEBUG
            if(i==0){
                printf("PixelTrackFeaturesExtractorKernels: nTracks=%d\n", tracks.nTracks());
                if(tracks.nTracks() >= maxNumberOfTracks)
                    printf("Warning: nTracks (%d) >= maxNumberOfTracks (%d)\n", tracks.nTracks(), maxNumberOfTracks);
            }
#endif
            const int trackLimit = std::min(maxNumberOfTracks, tracks.nTracks());

            if (i < trackLimit && 
                tracks[i].quality() >= minimumTrackQuality && 
                nHits(tracks, i)>=minNumberOfHits)
            {
                int idx = alpaka::atomicOp<alpaka::AtomicAdd>(acc, nKeptTracks, 1);
                if(idx < maxPreselectedTracks){
                    originalTrackIndex[idx] = i;
                    nKeptHits[idx] = nHits(tracks, i);
                }
#ifdef KERNELS_DEBUG
                if (idx==maxPreselectedTracks-1)
                    printf("PixelTrackFeaturesExtractorKernels: Reached maxPreselectedTracks (%d)\n", maxPreselectedTracks);
#endif
            }
        } 
    };

    struct FeaturesExtractorKernel{ 
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxPreselectedTracks,
            const ::reco::TrackSoAConstView tracks,
            const ::reco::TrackHitSoAConstView track_hits,
            const ::reco::TrackingRecHitConstView hits,
            PixelTrackFeaturesSoAView trackFeatures,
            PixelRecHitFeaturesSoAView hitFeatures,
            const int* nKeptTracks,
            int* originalTrackIndex
        ) const {

            const int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
            const auto nPreselectedTracks = std::min(*nKeptTracks, maxPreselectedTracks);
            
            // Case 1: valid preselected track --> extract features
            if (i < nPreselectedTracks){
                int idx = originalTrackIndex[i];
#ifdef KERNELS_DEBUG
                if (idx<0) printf("PixelTrackFeaturesExtractorKernels: Invalid originalTrackIndex for preselected idx %d\n", i);
#endif
                // Indices into track covariance matrix
                constexpr int cPhi = 0;
                constexpr int cTip = 5;
                constexpr int cInvPt = 9;
                constexpr int cZip = 14;
                
                const auto track = tracks[idx];
                const auto cov   = track.covariance();
                const auto state = track.state();
                const float pt   = track.pt();
                const int n_hits = nHits(tracks, idx);
                const int ndof   = n_hits * 2 - 5;
                const float ptError = xtd::sqrt(cov(cInvPt)) * pt * pt;

#ifdef KERNELS_DEBUG
                if(n_hits>RecHitFeatures::MaxHitsPerTrack) printf("PixelTrackFeaturesExtractorKernels: Number of hits (%d) exceeds MaxHitsPerTrack (%d)\n", n_hits, RecHitFeatures::MaxHitsPerTrack);
#endif
                trackFeatures.chi2(i)     = track.chi2() * ndof;
                trackFeatures.dzError(i)  = xtd::sqrt(cov(cZip));
                trackFeatures.dxyError(i) = xtd::sqrt(cov(cTip));
                trackFeatures.eta(i)      = track.eta();
                trackFeatures.ndof(i)     = ndof;
                trackFeatures.phi(i)      = state(0);
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

                uint32_t hitBegin = (idx == 0) ? 0 : tracks[idx - 1].hitOffsets();
                uint32_t hitEnd   = track.hitOffsets();
                uint32_t nHits   = hitEnd - hitBegin;

                nHits = std::min(nHits, uint32_t(RecHitFeatures::MaxHitsPerTrack));

                for (uint32_t h = 0; h < nHits; ++h) {
                    auto hit_id = track_hits[hitBegin + h].id();
                    const auto hit = hits[hit_id];
                    const float x = hit.xGlobal();
                    const float y = hit.yGlobal();
                    const float z = hit.zGlobal();
                    const float r = hit.rGlobal();

                    hitMatrix(h, HitFeaturesIDX::x) = x;
                    hitMatrix(h, HitFeaturesIDX::y) = y;
                    hitMatrix(h, HitFeaturesIDX::z) = z;

                    hitMatrix(h, HitFeaturesIDX::r) = r;                        
                    hitMatrix(h, HitFeaturesIDX::eta) =
                        (r > 0.f) ? xtd::asinh(z / r) : 0.f;
                    hitMatrix(h, HitFeaturesIDX::phi) =
                        xtd::atan2(y, x);

                    hitMatrix(h, HitFeaturesIDX::xErr) =
                        hit.xerrLocal();
                    hitMatrix(h, HitFeaturesIDX::yErr) =
                        hit.yerrLocal();
                }
            }
            // Case 2: padding entries --> fill with NaNs for inference
            else if (i < maxPreselectedTracks)
            {
                originalTrackIndex[i] = -1; // mark as invalid
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
            const int maxPreselectedTracks,
            const ::reco::TrackSoAConstView tracks,
            const ::reco::TrackHitSoAConstView track_hits,
            int* originalTrackIndex,
            const int* nKeptTracks,
            const int* nKeptHits,
            ::reco::TrackSoAView tracks_out,
            ::reco::TrackHitSoAView track_hits_out
        ) const {

            const int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            const int nTracks = std::min(*nKeptTracks, maxPreselectedTracks);

            if (i==0) {
                tracks_out.nTracks() = nTracks;
            }

            if (i < nTracks){
                int inputTrackIdx = originalTrackIndex[i];
                if(inputTrackIdx >= 0){
                    const auto track = tracks[inputTrackIdx];
                    tracks_out[i] = track;
                    tracks_out[i].hitOffsets() = nKeptHits[i];

                    //Access the hits associated to the track:
                    uint32_t hitBegin  = (inputTrackIdx == 0) ? 0 : tracks[inputTrackIdx-1].hitOffsets();
                    uint32_t hitEnd    = track.hitOffsets();
                    uint32_t outStart = (i == 0) ? 0 :  nKeptHits[i-1];
                    
                    for (uint32_t h = 0; h < (hitEnd - hitBegin); ++h) {
                        track_hits_out[outStart+h].id()    = track_hits[hitBegin + h].id();
                        track_hits_out[outStart+h].detId() = track_hits[hitBegin + h].detId();
                    }
                }
                else{
#ifdef KERNELS_DEBUG
                    printf("PixelTrackFeaturesExtractorKernels: Error inputTrackIdx is negative");
#endif                    
                }
            }
        } 
    };

    struct HitOffsetCompactKernel{
        // TODO: implement the prefix scan provided my cmssw
        // NOTE:
        //  - originalTrackIndex[] is compacted in place
        //  - nKeptHits[] initially stores per-track hit counts
        //  - after this kernel, nKeptHits[] becomes an inclusive prefix sum

        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int  maxPreselectedTracks,
            int* originalTrackIndex,
            int* nKeptTracks,
            int* nKeptHits
        ) const 
        {
            int nTracks = 0;
            for (int j = 0; j < maxPreselectedTracks; j++){
                if (originalTrackIndex[j] != -1){
                    originalTrackIndex[nTracks]  = originalTrackIndex[j];
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
            for (int i = nTracks; i < maxPreselectedTracks; i++){
                nKeptHits[i] = nKeptHits[nTracks-1];
                originalTrackIndex[i] = -1;
            }
        }
    };

    struct ScoreFilterKernel{
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxPreselectedTracks,
            const double scoreThreshold,
            int* originalTrackIndex,
            const PixelTrackScoresSoA::View trackScores) const 
        {
            const int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            if (i < maxPreselectedTracks){
                const float score = trackScores[i].score();  
                // Invalidate track slot if DNN score is below threshold
                if (score < scoreThreshold && originalTrackIndex[i] != -1){
                    originalTrackIndex[i] = -1;
                }
            }
        }
    };

    void launchCAPreselectionKernel(
        Queue& queue,
        const int maxNumberOfTracks,
        const int maxPreselectedTracks,
        const int minNumberOfHits,
        const ::pixelTrack::Quality minimumTrackQuality,
        const ::reco::TrackSoAConstView tracks,
        int* nKeptTracks,
        int* nKeptHits,
        int* originalTrackIndex)
    {
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(maxNumberOfTracks, threadsPerBlock);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            CAPreselectionKernel{},
            maxNumberOfTracks,
            maxPreselectedTracks,
            minNumberOfHits,
            minimumTrackQuality,
            tracks,
            nKeptTracks,
            nKeptHits,
            originalTrackIndex
        );


    }

    void launchFeaturesExtractorKernel(
        Queue& queue,
        const int maxPreselectedTracks,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        const ::reco::TrackingRecHitConstView hits,
        PixelTrackFeaturesSoAView trackFeatures,
        PixelRecHitFeaturesSoAView hitFeatures,
        const int* nKeptTracks,
        int* originalTrackIndex)
    {
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            FeaturesExtractorKernel{},
            maxPreselectedTracks,
            tracks,
            track_hits,
            hits,
            trackFeatures,
            hitFeatures,
            nKeptTracks,
            originalTrackIndex
        );
    }

    void launchScoreFilterKernel(
        Queue& queue,
        const int maxPreselectedTracks,
        const double scoreThreshold,
        int* originalTrackIndex,
        const PixelTrackScoresSoA::View trackScores)
    {
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            ScoreFilterKernel{},
            maxPreselectedTracks,
            scoreThreshold,
            originalTrackIndex,
            trackScores
        );
    }

    void launchPixelTrackFilterKernel(
        Queue& queue,
        const int maxPreselectedTracks,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        int* originalTrackIndex,
        const int* nKeptTracks,
        const int* nKeptHits,
        ::reco::TrackSoAView tracks_out,
        ::reco::TrackHitSoAView track_hits_out)
    {
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks =
            cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
        const auto workDiv =
            cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            PixelTrackFilterKernel{},
            maxPreselectedTracks,
            tracks,
            track_hits,
            originalTrackIndex,
            nKeptTracks,
            nKeptHits,
            tracks_out,
            track_hits_out
        );
    }

    void launchHitOffsetCompactKernel(
        Queue& queue,
        const int  maxPreselectedTracks,
        int* originalTrackIndex,
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
            maxPreselectedTracks,
            originalTrackIndex,
            nKeptTracks,
            nKeptHits
        );
    }

    reco::TracksSoACollection launchProduceOutputTracks(
        Queue& queue,
        const int maxPreselectedTracks,
        const int avgHitsPerTrack,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        int* originalTrackIndex,
        const int* nKeptTracks,
        const int* nKeptHits)
    {
        reco::TracksSoACollection tracks_out({{int(maxPreselectedTracks), int(maxPreselectedTracks * avgHitsPerTrack)}}, queue);

        launchPixelTrackFilterKernel(
            queue,
            maxPreselectedTracks,
            tracks,
            track_hits,
            originalTrackIndex,
            nKeptTracks,
            nKeptHits,
            tracks_out.view(),
            tracks_out.view<TrackHitSoA>()
        );

        return tracks_out;
    }

}