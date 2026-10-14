#include "DataFormats/L1ScoutingSoA/interface/alpaka/SoftTauDeviceTensor.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/AssociationMapDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/plugins/alpaka/TransformKernel.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  enum class Substep : uint8_t {
    Sorting = 0, 
    Reshaping = 1, 
    Tagging = 2
  };

  Substep parseSubstep(std::string_view value) {
    if (value == "sorting") {
      return Substep::Sorting;
    }
    if (value == "reshaping") {
      return Substep::Reshaping;
    }
    if (value == "tagging") {
      return Substep::Tagging;
    }

    throw cms::Exception("Configuration")
        << "Invalid substep value '" << value << "'. "
        << "Allowed values are: sorting, reshaping, tagging.";
  }

  class SoftTauIdML : public stream::FixedQueueEDProducer<> {
  public:
    SoftTauIdML(const edm::ParameterSet &params)
        : FixedQueueEDProducer<>(params),
          pf_candidates_token_(consumes(params.getParameter<edm::InputTag>("srcCandidates"))),
          bx_clusters_map_token_(consumes(params.getParameter<edm::InputTag>("srcBxClustersMap"))),
          cluster_cands_map_token_(consumes(params.getParameter<edm::InputTag>("srcClustersCandsMap"))),
          clusters_token_(consumes(params.getParameter<edm::InputTag>("srcClusters"))),
          cluster_cands_map_sorted_token_(produces("clusterCandsMapSorted")),
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          substep_(parseSubstep(params.getParameter<std::string>("substep"))
          ),
          batch_size_(params.getParameter<uint32_t>("batchSize")) {
            if (substep_ >= Substep::Reshaping) {
              soft_tau_input_token_ = produces("softTauInputs");
            }
            if (substep_ >= Substep::Tagging) {
              soft_tau_output_token_ = produces("softTauOutputs");
            }
          }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
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

    void beginStream(edm::StreamID, Queue queue) {
      // Warming up the model with dummy data. This allows for fast inference since the first iteration
      // and should ensure a more homogeneous memory balance

      SoftTauInputDeviceTensor input_tensor(queue, batch_size_);
      SoftTauOutputDeviceTensor output_tensor(queue, batch_size_);
      auto input_records = input_tensor.view().records(); 
      auto output_records = output_tensor.view().records();

      for (auto ii = 0; ii < warmupIterations_; ++ii) {
        cms::torch::alpakatools::TensorCollection<Queue> inputs_d, outputs_d;

        // register inputs
        inputs_d.add<SoftTauInputTensorSoA>("features", input_records.features());
        inputs_d.add<SoftTauInputTensorSoA>("pad_mask", input_records.pad_mask());

        // register outputs
        outputs_d.add<SoftTauOutputTensorSoA>("cls", output_records.cls());
        outputs_d.add<SoftTauOutputTensorSoA>("pt", output_records.pt());
        outputs_d.add<SoftTauOutputTensorSoA>("charge", output_records.charge());

        // inference on dummy data
        model_.forward(queue, inputs_d, outputs_d, ::torch::kHalf);
      }
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      const auto &pf = event.get(pf_candidates_token_); // pf collection
      const auto &bx_clusters_map = event.get(bx_clusters_map_token_); // bx -> clusters map
      const auto &cluster_cands_map = event.get(cluster_cands_map_token_); // clusters->candidates map
      const auto &clusters = event.get(clusters_token_); // cluster ID for each candidate

      // sort the clusters->candidates association map by pt
      auto cluster_cands_map_sorted = kernels::sortClustersCandsMap(event.queue(), pf, bx_clusters_map, cluster_cands_map, clusters);

      if (substep_ >= Substep::Reshaping) {
        // get filled input tensor
        SoftTauInputDeviceTensor input_tensor = kernels::transform(event.queue(), pf, cluster_cands_map_sorted);

        if (substep_ >= Substep::Tagging) {
          // initialize output tensor
          const auto job_size = cluster_cands_map.const_view().offset().metadata().size() - 1; // number of elements to run the inference on, which is the number of clusters
          auto output_tensor = SoftTauOutputDeviceTensor(event.queue(), job_size);
          output_tensor.zeroInitialise(event.queue());
          
          // set batch size
          assert(batch_size_ > 0 && "batch_size_ is expected to be greater than zero, as unbatched inference will probably make the device go out of memory");

          // records
          auto input_records = input_tensor.view().records(); // pay A LOT OF attention here to constness
          auto output_records = output_tensor.view().records();
          
          // batched tensors
          cms::torch::alpakatools::BatchedTensorCollection<Queue> inputs, outputs;

          // register inputs
          inputs.addBatched<SoftTauInputTensorSoA>("features", batch_size_, input_records.features());
          inputs.addBatched<SoftTauInputTensorSoA>("pad_mask", batch_size_, input_records.pad_mask());

          // register outputs
          outputs.addBatched<SoftTauOutputTensorSoA>("cls", batch_size_, output_records.cls());
          outputs.addBatched<SoftTauOutputTensorSoA>("pt", batch_size_, output_records.pt());
          outputs.addBatched<SoftTauOutputTensorSoA>("charge", batch_size_, output_records.charge());

          // do inference
          model_.forward(event.queue(), inputs, outputs, ::torch::kHalf);
          
          // emplace output
          event.emplace(soft_tau_output_token_, std::move(output_tensor));
        }

        // emplace input
        event.emplace(soft_tau_input_token_, std::move(input_tensor));
      }

      // emplace sorted map
      event.emplace(cluster_cands_map_sorted_token_, std::move(cluster_cands_map_sorted)); 
    }

  private:
    // event query tokens
    const device::EDGetToken<PFCandidateDeviceCollection> pf_candidates_token_;
    // input bx -> clusters map 
    const device::EDGetToken<BxLookupDevice> bx_clusters_map_token_;
    // input clusters -> candidates map that must be sorted by pt
    const device::EDGetToken<AssociationMapDevice> cluster_cands_map_token_;
    // cluster ID for each candidate
    const device::EDGetToken<ClustersDeviceCollection> clusters_token_;
    // sorted clusters -> candidates map that is emplaced in the event
    const device::EDPutToken<AssociationMapDevice> cluster_cands_map_sorted_token_;
    // put ml input into event
    device::EDPutToken<SoftTauInputDeviceTensor> soft_tau_input_token_;
    // put ml output into event
    device::EDPutToken<SoftTauOutputDeviceTensor> soft_tau_output_token_;
    // model
    torch::AlpakaModel model_;
    // processing step
    Substep substep_;
    // scouting switch
    const uint32_t batch_size_;
    // warmup iterations
    const int warmupIterations_ = 10;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(l1sc::SoftTauIdML);