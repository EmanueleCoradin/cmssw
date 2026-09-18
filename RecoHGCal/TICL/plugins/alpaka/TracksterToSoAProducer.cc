#include <array>
#include <cstdint>
#include <limits>
#include <string_view>

#include <Eigen/Core>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    Eigen::Vector3f toEigen(::ticl::Trackster::Vector const& vector) {
      return {vector.x(), vector.y(), vector.z()};
    }

  }  // namespace

  class TrackstersToSoAProducer : public stream::EDProducer<> {
  public:
    explicit TrackstersToSoAProducer(edm::ParameterSet const& config)
        : EDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("src"))},
          trackstersSoAToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription description;
      description.add<edm::InputTag>("src", edm::InputTag("ticlTrackstersCLUE3DHigh"));
      descriptions.addWithDefaultLabel(description);
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      edm::LogError("TrackstersToSoAProducer") << "producing";
      const auto& input = event.get(trackstersToken_);

      int32_t nVertices = 0;
      int32_t nMultiplicities = 0;
      int32_t nEdges = 0;
      int32_t nTrackIndices = 0;
      int32_t nGsfTrackIndices = 0;

      for (const auto& trackster : input) {
        if (trackster.vertices().size() != trackster.vertex_multiplicity().size()) {
          throw cms::Exception("TrackstersToSoAProducer")
              << "A Trackster has " << trackster.vertices().size() << " vertices but "
              << trackster.vertex_multiplicity().size() << " multiplicities.";
        }
        
        nVertices += trackster.vertices().size();
        nMultiplicities += trackster.vertex_multiplicity().size();
        nEdges += trackster.edges().size();
        nTrackIndices += trackster.trackIdxs().size();
        nGsfTrackIndices += trackster.gsftrackIdxs().size();
      }
      
      const auto nTracksters = static_cast<int32_t>(input.size());
      std::array<int32_t, ::ticl::TracksterSoA::blocksNumber> const sizes{
          nTracksters,
          nTracksters, nVertices,
          nTracksters, nMultiplicities,
          nTracksters, nEdges,
          nTracksters, nTrackIndices,
          nTracksters, nGsfTrackIndices,
          nTracksters, 0 // Legacy Trackster has no globalSeedingTracks vector (?).
      };

      ::ticl::TracksterHost output(event.queue(), sizes);
      output.zeroInitialise();
      auto view = output.view();

      auto vertexOffset = 0u;
      auto multiplicityOffset = 0u;
      auto edgeOffset = 0u;
      auto trackOffset = 0u;
      auto gsfTrackOffset = 0u;

      for (auto i = 0u; i < input.size(); ++i) {
        const auto& source = input[i];
        auto destination = view.tracksters()[i];

        destination.regressed_energy() = source.regressed_energy();
        destination.raw_energy() = source.raw_energy();
        destination.boundTime() = source.boundaryTime();
        destination.time() = source.time();
        destination.raw_em_energy() = source.raw_em_energy();
        destination.timeError() = source.timeError();
        destination.seedIndex() = source.seedIndex();

        destination.eigenvalues0() = source.eigenvalues()[0];
        destination.eigenvalues1() = source.eigenvalues()[1];
        destination.eigenvalues2() = source.eigenvalues()[2];
        destination.eigenvectors0() = toEigen(source.eigenvectors()[0]);
        destination.eigenvectors1() = toEigen(source.eigenvectors()[1]);
        destination.eigenvectors2() = toEigen(source.eigenvectors()[2]);
        destination.sigmas0() = source.sigmas()[0];
        destination.sigmas1() = source.sigmas()[1];
        destination.sigmas2() = source.sigmas()[2];
        destination.sigmasPCA0() = source.sigmasPCA()[0];
        destination.sigmasPCA1() = source.sigmasPCA()[1];
        destination.sigmasPCA2() = source.sigmasPCA()[2];

        destination.barycenterX() = source.barycenter().x();
        destination.barycenterY() = source.barycenter().y();
        destination.barycenterZ() = source.barycenter().z();

        destination.id_probabilities0() = source.id_probabilities()[0];
        destination.id_probabilities1() = source.id_probabilities()[1];
        destination.id_probabilities2() = source.id_probabilities()[2];
        destination.id_probabilities3() = source.id_probabilities()[3];
        destination.id_probabilities4() = source.id_probabilities()[4];
        destination.id_probabilities5() = source.id_probabilities()[5];
        destination.id_probabilities6() = source.id_probabilities()[6];
        destination.id_probabilities7() = source.id_probabilities()[7];
        destination.iterationIndex() = static_cast<uint8_t>(source.ticlIteration());

        for (auto vertex : source.vertices()) {
          view.vertices().content()[vertexOffset++].values() = vertex;
        }
        view.vertices().offsets()[i + 1].keys_offsets() = vertexOffset;

        for (auto multiplicity : source.vertex_multiplicity()) {
          view.multiplicity().content()[multiplicityOffset++].values() = multiplicity;
        }
        view.multiplicity().offsets()[i + 1].keys_offsets() = multiplicityOffset;

        for (const auto& edge : source.edges()) {
          view.edges().content()[edgeOffset++].values() = {edge[0], edge[1]};
        }
        view.edges().offsets()[i + 1].keys_offsets() = edgeOffset;

        for (auto trackIndex : source.trackIdxs()) {
          view.tracks().content()[trackOffset++].values() = trackIndex;
        }
        view.tracks().offsets()[i + 1].keys_offsets() = trackOffset;

        for (auto gsfTrackIndex : source.gsftrackIdxs()) {
          view.tracksterGsfTrack().content()[gsfTrackOffset++].values() = gsfTrackIndex;
        }
        view.tracksterGsfTrack().offsets()[i + 1].keys_offsets() = gsfTrackOffset;

        // There is no corresponding global seeding vector in the legacy Trackster.
      }

      event.emplace(trackstersSoAToken_, std::move(output));
    }

  private:
    edm::EDGetTokenT<::ticl::TracksterCollection> const trackstersToken_;
    edm::EDPutTokenT<::ticl::TracksterHost> const trackstersSoAToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackstersToSoAProducer);