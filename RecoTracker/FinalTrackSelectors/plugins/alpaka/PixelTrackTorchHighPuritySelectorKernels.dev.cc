#include <alpaka/alpaka.hpp>

#include <xtd/math/asinh.h>
#include <xtd/math/atan2.h>
#include <xtd/math/sqrt.h>
#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"

//#define KERNELS_DEBUG

// ------------------------------------------------------------------------------   

namespace ALPAKA_ACCELERATOR_NAMESPACE {
    using PixelTrackFeaturesSoAView  = PixelTrackFeaturesSoA::View;
    using PixelRecHitFeaturesSoAView = RecHitFeatures::PixelRecHitFeaturesSoA::View;
    using TrackHitSoA  = ::reco::TrackHitSoA;
    using HitFeaturesIDX = RecHitFeatures::HitFeature;

// ------------------------------------------------------------------------------
// --------------------------- Definitions of Kernels ---------------------------
// ------------------------------------------------------------------------------

struct PreselectionMaskingKernel
    {
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxNumberOfTracks,
            const int minNumberOfHits,
            const ::pixelTrack::Quality minimumTrackQuality,
            const ::reco::TrackSoAConstView tracks,
            int* preselectionMask,
            int* tmpPreselectedTrackIndices) const 
        {
        /**
            * Applies a fast preselection to pixel tracks based on:
            *  - CAHitNtuplet quality flag
            *  - minimum number of associated hits
            *
            * Inputs:
            *  - tracks              : input TrackSoA
            *  - maxNumberOfTracks   : maximum number of tracks to consider
            *  - minNumberOfHits     : minimum number of hits per track
            *  - minimumTrackQuality : minimum allowed track quality
            *
            * Outputs:
            *  - preselectionMask[i] = 1 if track i passes preselection, 0 otherwise
            *  - tmpPreselectedTrackIndices[i] = i (identity mapping, used for compaction)
            *
            * Notes:
            *  - Only tracks in [0, min(maxNumberOfTracks, tracks.nTracks())) are processed
            *  - Entries beyond this range are left unchanged and are expected to be
            *    pre-initialised by the caller.
            *  - This kernel does not perform compaction; it only prepares the mask
        */


            const int trackLimit = alpaka::math::min(acc, maxNumberOfTracks, tracks.nTracks());
#ifdef KERNELS_DEBUG
            if(cms::alpakatools::once_per_block(acc)){
                printf("nTracks=%d\n", tracks.nTracks());
                if(tracks.nTracks() >= maxNumberOfTracks)
                    printf("PixelTrackTorchHighPuritySelectorKernels Warning: nTracks (%d) >= maxNumberOfTracks (%d)\n", tracks.nTracks(), maxNumberOfTracks);
            }
#endif
            for (auto i : cms::alpakatools::uniform_elements(acc, trackLimit)){
                tmpPreselectedTrackIndices[i] = i;
                bool isGoodQuality = 
                    tracks[i].quality() >= minimumTrackQuality && 
                    nHits(tracks, i)    >= minNumberOfHits;
                preselectionMask[i] = isGoodQuality ? 1 : 0;
            }
        } 
    };

// ------------------------------------------------------------------------------

struct FeaturesExtractorKernel{ 
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxPreselectedTracks,
            const ::reco::TrackSoAConstView tracks,
            const ::reco::TrackHitSoAConstView track_hits,
            const ::reco::TrackingRecHitConstView hits,
            const int* preselectedTrackIndices,
            const int* nPreselectedTracks,
            PixelTrackFeaturesSoAView trackFeatures,
            PixelRecHitFeaturesSoAView hitFeatures,
            int* nKeptHits
        ) const 
        {
        /**
            * Extracts per-track and per-hit features used as input to
            * the Torch HighPurity classifier.
            *
            * For each valid preselected track:
            *  - Per-track features are written to PixelTrackFeaturesSoA
            *  - Per-hit features are written to PixelRecHitFeaturesSoA
            *  - nKeptHits[i] initially stores the number of hits per track
            *    and is later transformed into hit offsets via prefix-scan

            *
            * Padding policy:
            *  - Slots i >= nPreselectedTracks are treated as padding
            *  - All track and hit features for padding slots are filled with NaNs
            *
            * Preconditions:
            *  - preselectedTrackIndices contains a compact list of valid track indices
            *  - The first nPreselectedTracks entries are valid
            * This guarantees fixed-size tensors for Torch inference.
        */

            constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
            const uint32_t nPreselectedTracksBound = alpaka::math::min(acc, *nPreselectedTracks, maxPreselectedTracks);

            for (auto i : cms::alpakatools::uniform_elements(acc, maxPreselectedTracks)) { 
                // Case 1: valid preselected track --> extract features
                if (i < nPreselectedTracksBound){
                    int inputTrackIdx = preselectedTrackIndices[i];
#ifdef KERNELS_DEBUG
                    if (inputTrackIdx<0) printf("PixelTrackTorchHighPuritySelectorKernels: Invalid preselectedTrackIndices for preselected inputTrackIdx %d\n", i);
#endif

                    // Indices to the 5-dimensional track state vector (CMS convention)
                    static constexpr int kStatePhi        = 0;
                    static constexpr int kStateDxy        = 1;
                    static constexpr int kStateQOverPt    = 2;// Packed covariance indices
                    static constexpr int kStateCotTheta   = 3;
                    static constexpr int kStateDz         = 4;

                    // Indices into the 5x5 track covariance matrix (CMS convention)
                    static constexpr int kCovPhiPhi           = 0;   // (0,0)
                    static constexpr int kCovPhiDxy           = 1;   // (0,1)
                    static constexpr int kCovPhiQOverPt	      = 2;   // (0,2)
                    static constexpr int kCovDxyDxy           = 5;   // (1,1)
                    static constexpr int kCovDxyQOverPt	      = 6;   // (1,2)
                    static constexpr int kCovQOverPtQOverPt   = 9;   // (2,2)
                    static constexpr int kCovCotThetaCotTheta = 12;  // (3,3)
                    static constexpr int kCovCotThetaDz	      = 13;  // (3,4)
                    static constexpr int kCovDzDz             = 14;  // (4,4)

                    // Access the track
                    const auto& track   = tracks[inputTrackIdx];
                    const auto& cov     = track.covariance();
                    const auto& state   = track.state();
                    const int numHits   = nHits(tracks, inputTrackIdx);

                    nKeptHits[i] = numHits;

#ifdef KERNELS_DEBUG
                    if(numHits>RecHitFeatures::MaxHitsPerTrack) printf("PixelTrackTorchHighPuritySelectorKernels: Number of hits (%d) exceeds MaxHitsPerTrack (%d)\n", numHits, RecHitFeatures::MaxHitsPerTrack);
#endif
                    // Fill per-track features
                    trackFeatures.chi2(i)           = track.chi2(); // in the SoA chi2 is stored as chi2/ndof
                    trackFeatures.dzError(i)        = xtd::sqrt(cov(kCovDzDz));
                    trackFeatures.dxyError(i)       = xtd::sqrt(cov(kCovDxyDxy));
                    trackFeatures.eta(i)            = track.eta();
                    trackFeatures.nHits(i)          = numHits;
                    trackFeatures.phi(i)            = state(kStatePhi);
                    trackFeatures.phiError(i)       = xtd::sqrt(cov(kCovPhiPhi));
                    trackFeatures.pt(i)             = track.pt();
                    trackFeatures.qOverPtError(i)   = xtd::sqrt(cov(kCovQOverPtQOverPt));
                    trackFeatures.dzBS(i)           = state(kStateDz);
                    trackFeatures.dxyBS(i)          = state(kStateDxy);
                    trackFeatures.nLayers(i)        = track.nLayers();
                    trackFeatures.cotThetaError(i)  = xtd::sqrt(cov(kCovCotThetaCotTheta));
                    trackFeatures.covCotThetaDz(i)  = cov(kCovCotThetaDz);   
                    trackFeatures.covDxyQOverPt(i)  = cov(kCovDxyQOverPt);
                    trackFeatures.covPhiDxy(i)      = cov(kCovPhiDxy);
                    trackFeatures.covPhiQOverPt(i)  = cov(kCovPhiQOverPt);

                    //Prefill hit features:
                    auto hitMatrix = hitFeatures.hits(i);
                    for (int h = 0; h < RecHitFeatures::MaxHitsPerTrack; ++h) {
                        for (int f = 0; f < RecHitFeatures::HitFeatures; ++f) {
                            hitMatrix(h, f) = NaN;
                        }
                    }

                    uint32_t hitBegin = (inputTrackIdx == 0) ? 0 : tracks[inputTrackIdx - 1].hitOffsets();
                    uint32_t hitEnd   = track.hitOffsets();
                    uint32_t nHitsTrack    = hitEnd - hitBegin;

                    nHitsTrack = alpaka::math::min(acc, nHitsTrack, uint32_t(RecHitFeatures::MaxHitsPerTrack));

                    for (uint32_t h = 0; h < nHitsTrack; ++h) {
                        auto hit_id     = track_hits[hitBegin + h].id();
                        const auto& hit = hits[hit_id];

                        const float x  = hit.xGlobal();
                        const float y  = hit.yGlobal();
                        const float z  = hit.zGlobal();
                        const float r  = hit.rGlobal();

                        hitMatrix(h, HitFeaturesIDX::x) = x;
                        hitMatrix(h, HitFeaturesIDX::y) = y;
                        hitMatrix(h, HitFeaturesIDX::z) = z;
                        hitMatrix(h, HitFeaturesIDX::r) = r;

                        hitMatrix(h, HitFeaturesIDX::eta)  =
                            (r > 0.f) ? xtd::asinh(z / r) : 0.f;
                        
                        hitMatrix(h, HitFeaturesIDX::phi)  = xtd::atan2(y, x);
                    }
                }
                // Case 2: padding entries --> fill with NaNs for inference
                else if (i < (uint32_t) maxPreselectedTracks)
                {
                    trackFeatures.chi2(i)           = NaN;
                    trackFeatures.dzError(i)        = NaN;
                    trackFeatures.dxyError(i)       = NaN;
                    trackFeatures.eta(i)            = NaN;
                    trackFeatures.nHits(i)          = NaN; 
                    trackFeatures.phi(i)            = NaN;
                    trackFeatures.phiError(i)       = NaN;
                    trackFeatures.pt(i)             = NaN;
                    trackFeatures.qOverPtError(i)   = NaN;
                    trackFeatures.dzBS(i)           = NaN;
                    trackFeatures.dxyBS(i)          = NaN;
                    trackFeatures.nLayers(i)        = NaN;
                    trackFeatures.cotThetaError(i)  = NaN;
                    trackFeatures.covCotThetaDz(i)  = NaN;   
                    trackFeatures.covDxyQOverPt(i)  = NaN;
                    trackFeatures.covPhiDxy(i)      = NaN;
                    trackFeatures.covPhiQOverPt(i)  = NaN;

                    auto hitMatrix = hitFeatures.hits(i);
                    for (int h = 0; h < RecHitFeatures::MaxHitsPerTrack; ++h) {
                        for (int f = 0; f < RecHitFeatures::HitFeatures; ++f) {
                            hitMatrix(h, f) = NaN;
                        }
                    }
                }
            }
        }
    };

// ------------------------------------------------------------------------------

    struct PixelTrackFilterKernel{ 
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxPreselectedTracks,
            const ::reco::TrackSoAConstView tracks,
            const ::reco::TrackHitSoAConstView track_hits,
            const int* selectedTrackIndices,
            const int* nSelectedTracks,
            const int* nKeptHits,
            ::reco::TrackSoAView tracks_out,
            ::reco::TrackHitSoAView track_hits_out
        ) const 
        {
        /**
            * Produces the final output TrackSoA by:
            *  - Copying selected tracks from the input TrackSoA
            *  - Copying and compacting the associated TrackHitSoA
            *
            * Inputs:
            *  - selectedTrackIndices[]: compact list of selected input track indices
            *  - nSelectedTracks: number of selected tracks
            *  - nKeptHits[]: inclusive prefix sum of per-track hit counts.
            *                 nKeptHits[i] stores the end offset of hits for track i.
            *
            * Outputs:
            *  - tracks_out           : compact TrackSoA containing selected tracks
            *  - track_hits_out       : compact TrackHitSoA containing selected hits
            *
            * Notes:
            *  - tracks_out.nTracks() is set by a single thread
            *  - Hit offsets in tracks_out are taken from nKeptHits[]
        */

            const int nTracks = alpaka::math::min(acc, *nSelectedTracks, maxPreselectedTracks);
            if (cms::alpakatools::once_per_block(acc)) 
                tracks_out.nTracks() = nTracks;
            
            for (auto i : cms::alpakatools::uniform_elements(acc, nTracks)) { 
                const int inputTrackIdx = selectedTrackIndices[i];
                if(inputTrackIdx >= 0){
                    const auto& track = tracks[inputTrackIdx];
                    tracks_out[i]     = track;
                    tracks_out[i].hitOffsets() = nKeptHits[i];

                    //Access the hits associated to the track:
                    uint32_t hitBegin  = (inputTrackIdx == 0) ? 0 : tracks[inputTrackIdx-1].hitOffsets();
                    uint32_t hitEnd    = track.hitOffsets();
                    uint32_t outStart  = (i == 0) ? 0 :  nKeptHits[i-1];
                    
                    for (uint32_t h = 0; h < (hitEnd - hitBegin); ++h) {
                        track_hits_out[outStart+h].id()    = track_hits[hitBegin + h].id();
                        track_hits_out[outStart+h].detId() = track_hits[hitBegin + h].detId();
                    }
                }
                else{
#ifdef KERNELS_DEBUG
                    printf("PixelTrackTorchHighPuritySelectorKernels: Error inputTrackIdx is negative");
#endif                    
                }
                
            } 
        }
    };

// ------------------------------------------------------------------------------

    struct ScoreSelectionMaskKernel{
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            const int maxPreselectedTracks,
            const double scoreThreshold,
            const int* preselectedTrackIndices,
            const int* nPreselectedTracks,
            const PixelTrackScoresSoA::View trackScores,
            int* selectionMask,
            int* nKeptHits,
            int* nKeptHits_copy
        ) const 
        {
        /**
            * Applies a DNN score threshold to preselected tracks.
            *
            * For each track slot:
            *  - Reads the Torch score
            *  - Marks the track as selected if:
            *      score >= scoreThreshold AND track is a valid preselected track
            *
            * Outputs:
            *  - selectionMask[i] = 1 if track is selected, 0 otherwise
            *  - nKeptHits_copy[] : copy of per-track hit counts (used later)
            *
            * Notes:
            *  - No compaction is performed in this kernel
        */
            const uint32_t nValid = alpaka::math::min(acc, *nPreselectedTracks, maxPreselectedTracks);
            for (auto i : cms::alpakatools::uniform_elements(acc, nValid)) {  
                nKeptHits_copy[i] = nKeptHits[i];
                if(i < nValid){
                    const float score = trackScores[i].score();  
                    selectionMask[i] = (score >= scoreThreshold) ? 1 : 0;
                }
            }
        }
    };

// ------------------------------------------------------------------------------

    struct FilterArray{
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            int* old_array,
            int* new_array,
            int* offsets,
            int old_size,
            int* new_size
        ) const 
        {
        /**
            * Compacts an input array using precomputed inclusive prefix-sum offsets.
            *
            * Inputs:
            *  - old_array[] : input array
            *  - offsets[]   : inclusive prefix sum of a selection mask
            *  - old_size    : size of the input array
            *
            * Outputs:
            *  - new_array[] : compacted array
            *  - new_size    : total number of selected elements
            *
            * Notes:
            *  - offsets[last] defines the size of the compacted array
            *  - Only the first occurrence of each offset value writes to new_array
        */
            if (cms::alpakatools::once_per_block(acc)) {
                const uint32_t new_size_local = offsets[old_size-1];
                *new_size = new_size_local;
            }

            for (auto i : cms::alpakatools::uniform_elements(acc, old_size)){
                bool is_first = 
                    offsets[i] > 0 &&
                    ((i == 0) || (offsets[i] != offsets[i-1]));

                if (is_first){
                    new_array[offsets[i]-1] = old_array[i];
                }
            }
        }
    };

// ------------------------------------------------------------------------------
// -------------------------- Definitions of Launchers --------------------------
// ------------------------------------------------------------------------------

    void launchCAPreselection(
        Queue& queue,
        const int maxNumberOfTracks,
        const int minNumberOfHits,
        const ::pixelTrack::Quality minimumTrackQuality,
        const ::reco::TrackSoAConstView tracks,
        int* preselectedTrackIndices,
        int* preselectionOffsets,
        int* nPreselectedTracks)
    {
        // Produce a preselection mask based on track quality and number of hits
        auto tmpPreselectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks);
        auto preselectionMask     = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks);
        
        alpaka::memset(queue, tmpPreselectedTrackIndices, 0);
        alpaka::memset(queue, preselectionMask, 0);

        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks = cms::alpakatools::divide_up_by(maxNumberOfTracks, threadsPerBlock);
        const auto workDiv    = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
        
        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            PreselectionMaskingKernel{},
            maxNumberOfTracks,
            minNumberOfHits,
            minimumTrackQuality,
            tracks,
            preselectionMask.data(),
            tmpPreselectedTrackIndices.data()
        );

        // Apply the preselection mask to compact the preselectedTrackIndices array 
        // and produce the final list of preselected tracks, 
        // while also counting the number of selected tracks
        constexpr auto threadsPrefixScan = 256;
        auto blocksPrefixScan = (maxNumberOfTracks + threadsPrefixScan - 1) / threadsPrefixScan;
        auto workDivPrefixScan = cms::alpakatools::make_workdiv<Acc1D>(blocksPrefixScan, threadsPrefixScan);
        auto bCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
        alpaka::memset(queue, bCounter, 0);

        // Launch prefix-scan over the preselection mask to compute offsets
        alpaka::exec<Acc1D>(
            queue,
            workDivPrefixScan,
            cms::alpakatools::multiBlockPrefixScan<int>(),
            preselectionMask.data(),
            preselectionOffsets,
            maxNumberOfTracks,
            blocksPrefixScan,
            bCounter.data(),
            alpaka::getPreferredWarpSize(alpaka::getDev(queue))
        );

        // Compact the preselectedTrackIndices array using the preselection offsets
        alpaka::exec<Acc1D>(
            queue,
            workDivPrefixScan,
            FilterArray{},
            tmpPreselectedTrackIndices.data(),
            preselectedTrackIndices,
            preselectionOffsets,
            maxNumberOfTracks,
            nPreselectedTracks
        );
    }

// ------------------------------------------------------------------------------

    void launchFeaturesExtractor(
        Queue& queue,
        const int maxPreselectedTracks,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        const ::reco::TrackingRecHitConstView hits,
        const int* preselectedTrackIndices,
        const int* nPreselectedTracks,
        PixelTrackFeaturesSoAView trackFeatures,
        PixelRecHitFeaturesSoAView hitFeatures,
        int* nKeptHits
    )
    {
        // Extract per-track and per-hit features for Torch inference
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks = cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
        const auto workDiv    = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            FeaturesExtractorKernel{},
            maxPreselectedTracks,
            tracks,
            track_hits,
            hits,
            preselectedTrackIndices,
            nPreselectedTracks,
            trackFeatures,
            hitFeatures,
            nKeptHits
        );
    }

// ------------------------------------------------------------------------------

    void launchScoreFilter(
        Queue& queue,
        const int maxPreselectedTracks,
        const double scoreThreshold,
        const PixelTrackScoresSoA::View trackScores,
        const int* preselectedTrackIndices,
        const int* nPreselectedTracks,
        int* selectedTrackIndices,
        int* nSelectedTracks,
        int* nKeptHits)
    {
        // Produce a selection mask out of the DNN scores
        auto selectionMask        = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks);
        auto selectionOffsets     = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks);
        auto nKeptHits_copy       = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks);

        alpaka::memset(queue, selectionMask, 0);
        alpaka::memset(queue, selectionOffsets, 0);
        alpaka::memset(queue, nKeptHits_copy, 0);

        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks = cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
        const auto workDiv    = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            ScoreSelectionMaskKernel{},
            maxPreselectedTracks,
            scoreThreshold,
            preselectedTrackIndices,
            nPreselectedTracks,
            trackScores,
            selectionMask.data(),
            nKeptHits,
            nKeptHits_copy.data()
        );
        
        // Apply the selection mask to compact the preselectedTrackIndices array 
        // and produce the final list of selected tracks, 
        // while also counting the number of kept tracks
        constexpr auto threadsPrefixScan = 256;
        auto blocksPrefixScan = (maxPreselectedTracks + threadsPrefixScan - 1) / threadsPrefixScan;
        auto workDivPrefixScan = cms::alpakatools::make_workdiv<Acc1D>(blocksPrefixScan, threadsPrefixScan);
        auto bCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
        alpaka::memset(queue, bCounter, 0);

        // Launch prefix-scan over the selection mask to compute offsets
        alpaka::exec<Acc1D>(
            queue,
            workDivPrefixScan,
            cms::alpakatools::multiBlockPrefixScan<int>(),
            selectionMask.data(),
            selectionOffsets.data(),
            maxPreselectedTracks,
            blocksPrefixScan,
            bCounter.data(),
            alpaka::getPreferredWarpSize(alpaka::getDev(queue))
        );

        // Compact the preselectedTrackIndices array using the selection offsets
        alpaka::exec<Acc1D>(
            queue,
            workDivPrefixScan,
            FilterArray{},
            const_cast<int*>(preselectedTrackIndices),
            selectedTrackIndices,
            selectionOffsets.data(),
            maxPreselectedTracks,
            nSelectedTracks
        );

        // Compact nKeptHits using the selection offsets
        alpaka::exec<Acc1D>(
            queue,
            workDivPrefixScan,
            FilterArray{},
            nKeptHits,
            nKeptHits_copy.data(),
            selectionOffsets.data(),
            maxPreselectedTracks,
            nSelectedTracks
        );

        // Finally, compute the prefix-scan of nKeptHits to get hit offsets
        alpaka::memset(queue, bCounter, 0);
        alpaka::exec<Acc1D>(
            queue,
            workDivPrefixScan,
            cms::alpakatools::multiBlockPrefixScan<int>(),
            nKeptHits_copy.data(),
            nKeptHits,
            maxPreselectedTracks,
            blocksPrefixScan,
            bCounter.data(),
            alpaka::getPreferredWarpSize(alpaka::getDev(queue))
        );
    }

// ------------------------------------------------------------------------------

    reco::TracksSoACollection launchProduceOutputTracks(
        Queue& queue,
        const int maxPreselectedTracks,
        const int avgHitsPerTrack,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        const int* selectedTrackIndices,
        const int* nSelectedTracks,
        const int* nKeptHits)
    {
        reco::TracksSoACollection tracks_out({{int(maxPreselectedTracks), int(maxPreselectedTracks * avgHitsPerTrack)}}, queue);
        
        constexpr uint32_t threadsPerBlock = 256;
        const uint32_t blocks = cms::alpakatools::divide_up_by(maxPreselectedTracks, threadsPerBlock);
        const auto workDiv    = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

        alpaka::exec<Acc1D>(
            queue,
            workDiv,
            PixelTrackFilterKernel{},
            maxPreselectedTracks,
            tracks,
            track_hits,
            selectedTrackIndices,
            nSelectedTracks,
            nKeptHits,
            tracks_out.view(),
            tracks_out.view<TrackHitSoA>()
        );

        return tracks_out;
    }
}