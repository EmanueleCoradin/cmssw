#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/SoftTauDeviceTensor.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/AssociationMapDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateDeviceCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/plugins/alpaka/TransformKernel.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/AlpakaSession.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoTrace.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  namespace {
    enum class Substep : uint8_t { Sorting = 0, Reshaping = 1, Tagging = 2 };

    Substep parseSubstep(std::string_view value) {
      if (value == "sorting")
        return Substep::Sorting;
      if (value == "reshaping")
        return Substep::Reshaping;
      if (value == "tagging")
        return Substep::Tagging;
      throw cms::Exception("Configuration")
          << "Invalid substep '" << value << "'. Allowed values: sorting, reshaping, tagging.";
    }
  }  // namespace

  class SoftTauIdONNX : public stream::FixedQueueEDProducer<> {
  public:
    explicit SoftTauIdONNX(const edm::ParameterSet& params)
        : FixedQueueEDProducer<>(params),
          pf_candidates_token_(consumes(params.getParameter<edm::InputTag>("srcCandidates"))),
          bx_clusters_map_token_(consumes(params.getParameter<edm::InputTag>("srcBxClustersMap"))),
          cluster_cands_map_token_(consumes(params.getParameter<edm::InputTag>("srcClustersCandsMap"))),
          clusters_token_(consumes(params.getParameter<edm::InputTag>("srcClusters"))),
          cluster_cands_map_sorted_token_(produces("clusterCandsMapSorted")),
          model_(params.getParameter<edm::FileInPath>("model").fullPath(), cms::Ort::Backend::tensorRT),
          substep_(parseSubstep(params.getParameter<std::string>("substep"))),
          batch_size_(params.getParameter<uint32_t>("batchSize")) {
      if (batch_size_ == 0) {
        throw cms::Exception("Configuration") << "SoftTauIdONNX: batchSize must be positive";
      }
      if (substep_ >= Substep::Reshaping) {
        soft_tau_input_token_ = produces("softTauInputs");
      }
      if (substep_ >= Substep::Tagging) {
        soft_tau_output_token_ = produces("softTauOutputs");
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("srcCandidates");
      desc.add<edm::InputTag>("srcBxClustersMap");
      desc.add<edm::InputTag>("srcClustersCandsMap");
      desc.add<edm::InputTag>("srcClusters");
      desc.add<edm::FileInPath>("model");
      desc.add<std::string>("substep", "tagging");
      desc.add<uint32_t>("batchSize", 32u);
      descriptions.addWithDefaultLabel(desc);
    }

    void beginStream(edm::StreamID id, Queue queue) override {
      // Create the ONNX Runtime session on this producer's fixed Alpaka queue.
      // The selected execution provider is TensorRT (see constructor).
      if (substep_ == Substep::Tagging) {
        model_.bind(queue);

        edm::LogInfo("SoftTauIdONNX")
          << "Stream " << id.value()
          << ": using "
          << (model_.usedCudaFallback() ? "CUDA fallback" : "TensorRT")
          << " execution provider";

      }
    }

    void produce(device::Event& event, const device::EventSetup&) override {
      CMS_PERFETTO_SCOPE("SoftTauIdONNX produce()");
      const auto& pf = event.get(pf_candidates_token_);
      const auto& bx_clusters_map = event.get(bx_clusters_map_token_);
      const auto& cluster_cands_map = event.get(cluster_cands_map_token_);
      const auto& clusters = event.get(clusters_token_);

      auto cluster_cands_map_sorted =
          kernels::sortClustersCandsMap(event.queue(), pf, bx_clusters_map, cluster_cands_map, clusters);

      if (substep_ >= Substep::Reshaping) {
        SoftTauInputDeviceTensor input_tensor = kernels::transform(event.queue(), pf, cluster_cands_map_sorted);

        if (substep_ >= Substep::Tagging) {
          CMS_PERFETTO_SCOPE("Run Batched STT ONNX Inference");

          // Number of clusters; input and output tensors are allocated for this collection.
          const auto job_size = cluster_cands_map.const_view().offset().metadata().size() - 1;
          auto output_tensor = SoftTauOutputDeviceTensor(event.queue(), job_size);
          output_tensor.zeroInitialise(event.queue());

          auto input_records = input_tensor.view().records();
          auto output_records = output_tensor.view().records();

          // Each ONNX tensor corresponds to the same registered group as the PyTorch producer.
          // The ONNX graph must expose the input/output tensors in this exact order and dtype.
          for (std::size_t offset = 0; offset < static_cast<std::size_t>(job_size);
               offset += static_cast<std::size_t>(batch_size_)) {
            const auto batch_id = static_cast<int>(offset / batch_size_);
            cms::Ort::alpakatools::TensorCollection<Queue> inputs(static_cast<int>(batch_size_),
                                                                  static_cast<int>(job_size));
            cms::Ort::alpakatools::TensorCollection<Queue> outputs(static_cast<int>(batch_size_),
                                                                   static_cast<int>(job_size));

            inputs.add<SoftTauInputTensorSoA>("features", batch_id, input_records.features());
            inputs.add<SoftTauInputTensorSoA>("padding_masks", batch_id, input_records.dummy_mask());

            outputs.add<SoftTauOutputTensorSoA>("cls", batch_id, output_records.cls());
            outputs.add<SoftTauOutputTensorSoA>("pt", batch_id, output_records.pt());
            outputs.add<SoftTauOutputTensorSoA>("charge", batch_id, output_records.charge());

            model_.forward(event.queue(), inputs, outputs);
          }

          event.emplace(soft_tau_output_token_, std::move(output_tensor));
        }
        event.emplace(soft_tau_input_token_, std::move(input_tensor));
      }
      event.emplace(cluster_cands_map_sorted_token_, std::move(cluster_cands_map_sorted));
    }

  private:
    const device::EDGetToken<PFCandidateDeviceCollection> pf_candidates_token_;
    const device::EDGetToken<BxLookupDevice> bx_clusters_map_token_;
    const device::EDGetToken<AssociationMapDevice> cluster_cands_map_token_;
    const device::EDGetToken<ClustersDeviceCollection> clusters_token_;
    const device::EDPutToken<AssociationMapDevice> cluster_cands_map_sorted_token_;
    device::EDPutToken<SoftTauInputDeviceTensor> soft_tau_input_token_;
    device::EDPutToken<SoftTauOutputDeviceTensor> soft_tau_output_token_;
    ort::AlpakaSession model_;
    const Substep substep_;
    const uint32_t batch_size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::SoftTauIdONNX);
