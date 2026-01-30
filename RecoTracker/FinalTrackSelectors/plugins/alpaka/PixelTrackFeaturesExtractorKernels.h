#ifndef PixelTrackFeaturesExtractorKernels_h
#define PixelTrackFeaturesExtractorKernels_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/interface/PixelRecHitFeaturesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    void launchCAPreselectionKernel(
        Queue& queue,
        const int maxNumberOfTracks,
        const int maxPreselectedTracks,
        const int minNumberOfHits,
        const ::pixelTrack::Quality minimumTrackQuality,
        const ::reco::TrackSoAConstView tracks,
        int* nKeptTracks,
        int* nKeptHits,
        int* originalTrackIndex
    );

    void launchFeaturesExtractorKernel(
        Queue& queue,
        const int maxPreselectedTracks,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        const ::reco::TrackingRecHitConstView hits,
        PixelTrackFeaturesSoA::View trackFeatures,
        RecHitFeatures::PixelRecHitFeaturesSoA::View hitFeatures,
        const int* nKeptTracks,
        int* originalTrackIndex
    );

    void launchScoreFilterKernel(
        Queue& queue,
        const int maxPreselectedTracks,
        const double scoreThreshold,
        int* originalTrackIndex,
        const PixelTrackScoresSoA::View trackScores
    );

    void launchPixelTrackFilterKernel(
        Queue& queue,
        const int maxTracksPreselection,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        int* originalTrackIndex,
        const int* nKeptTracks,
        const int* nKeptHits,
        ::reco::TrackSoAView tracks_out,
        ::reco::TrackHitSoAView track_hits_out
    );

    void launchHitOffsetCompactKernel(
        Queue& queue,
        const int maxPreselectedTracks,
        int* originalTrackIndex,
        int* nKeptTracks,
        int* nKeptHits
    );

    reco::TracksSoACollection launchProduceOutputTracks(
        Queue& queue,
        const int maxPreselectedTracks,
        const int avgHitsPerTrack,
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView track_hits,
        int* originalTrackIndex,
        const int* nKeptTracks,
        const int* nKeptHits
    );
}

#endif