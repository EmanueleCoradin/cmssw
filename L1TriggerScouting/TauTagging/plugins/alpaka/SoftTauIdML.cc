#include "DataFormats/L1ScoutingSoA/interface/alpaka/SoftTauDeviceTensor.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/AssociationMapDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/BxLookupDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/plugins/alpaka/TransformKernel.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  class SoftTauIdML : public stream::EDProducer<> {
  public:
    SoftTauIdML(const edm::ParameterSet &params)
        : EDProducer<>(params),
          pf_candidates_token_(consumes(params.getParameter<edm::InputTag>("srcCandidates"))),
          cluster_cands_map_token_{consumes(params.getParameter<edm::InputTag>("srcClustersCandsMap"))},
          cluster_cands_map_sorted_token_{produces("clusterCandsMapSorted")},
          soft_tau_token_{produces("outputTensor")},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          step_{params.getParameter<uint32_t>("step")},
          max_batch_size_{params.getParameter<uint32_t>("maxBatchSize")} {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("srcCandidates");
      desc.add<edm::InputTag>("srcClustersCandsMap");
      desc.add<edm::FileInPath>("model");
      desc.add<uint32_t>("step", 0u);
      desc.add<uint32_t>("maxBatchSize", std::numeric_limits<uint32_t>::max());
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      const auto &pf = event.get(pf_candidates_token_); // pf collection
      const auto &cluster_cands_map = event.get(cluster_cands_map_token_); // clusters->candidates map

      // initialize output tensor
      const auto job_size = cluster_cands_map.const_view().offset().metadata().size() - 1; // number of elements to run the inference on, which is the number of clusters
      auto output_tensor = SoftTauOutputDeviceTensor(event.queue(), job_size);
      output_tensor.zeroInitialise(event.queue());

      // sort the clusters->candidates association map by pt
      auto cluster_cands_map_sorted = kernels::sortClustersCandsMap(event.queue(), pf, cluster_cands_map);
      
      if (step_ == 0u) {
        alpaka::wait(event.queue());
      }

      if ((step_ == 1u) || (step_ == 2u)) {
        // get filled input tensor
        SoftTauInputDeviceTensor input_tensor = kernels::transform(event.queue(), pf, cluster_cands_map_sorted);

        if (step_ == 1u) {
          alpaka::wait(event.queue());
        }

        if (step_ == 2u) {
          // set batch size
          // const auto batch_size = std::min<uint32_t>(job_size, max_batch_size_);
          const auto batch_size = job_size;
    
          // prepare input TensorCollection
          cms::torch::alpakatools::TensorCollection<Queue> inputs(batch_size);
          cms::torch::alpakatools::TensorCollection<Queue> outputs(batch_size);

          // records
          auto input_tensor_records = input_tensor.view().records();
          inputs.add<SoftTauInputTensorSoA>("features", 
            input_tensor_records.features()
          );
          inputs.add<SoftTauInputTensorSoA>("pad_mask", 
            input_tensor_records.pad_mask()
          );

          auto output_tensor_records = output_tensor.view().records();
          outputs.add<SoftTauOutputTensorSoA>("output_logits", 
            output_tensor_records.cls(), 
            output_tensor_records.vz(), 
            output_tensor_records.pt(), 
            output_tensor_records.charge()
          );

          model_.forward(event.queue(), inputs, outputs);
          alpaka::wait(event.queue());
        }
      }

      // put device-side product into event
      event.emplace(cluster_cands_map_sorted_token_, std::move(cluster_cands_map_sorted));
      event.emplace(soft_tau_token_, std::move(output_tensor));
    }

  private:
    // event query tokens
    const device::EDGetToken<PFCandidateDeviceCollection> pf_candidates_token_;
    // input clusters -> candidates map that must be sorted by pt
    const device::EDGetToken<AssociationMapDevice> cluster_cands_map_token_;
    // sorted clusters -> candidates map that is emplaced in the event
    const device::EDPutToken<AssociationMapDevice> cluster_cands_map_sorted_token_;
    // put ml output into event
    const device::EDPutToken<SoftTauOutputDeviceTensor> soft_tau_token_;
    // model
    torch::AlpakaModel model_;
    // do inference or not
    const uint32_t step_;
    // scouting switch
    const uint32_t max_batch_size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(l1sc::SoftTauIdML);