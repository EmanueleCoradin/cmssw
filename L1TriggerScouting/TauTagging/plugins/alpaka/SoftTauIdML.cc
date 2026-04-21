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
          bx_clusters_map_token_{consumes(params.getParameter<edm::InputTag>("srcBxClustersMap"))},
          cluster_cands_map_token_{consumes(params.getParameter<edm::InputTag>("srcClustersCandsMap"))},
          clusters_token_{consumes(params.getParameter<edm::InputTag>("srcClusters"))},
          cluster_cands_map_sorted_token_{produces("clusterCandsMapSorted")},
          soft_tau_token_{produces("outputTensor")},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          step_{params.getParameter<uint32_t>("step")},
          max_batch_size_{params.getParameter<uint32_t>("maxBatchSize")} {}

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("srcCandidates");
      desc.add<edm::InputTag>("srcBxClustersMap");
      desc.add<edm::InputTag>("srcClustersCandsMap");
      desc.add<edm::InputTag>("srcClusters");
      desc.add<edm::FileInPath>("model");
      desc.add<uint32_t>("step", 0u);
      desc.add<uint32_t>("maxBatchSize", std::numeric_limits<uint32_t>::max());
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      const auto &pf = event.get(pf_candidates_token_); // pf collection
      const auto &bx_clusters_map = event.get(bx_clusters_map_token_); // bx -> clusters map
      const auto &cluster_cands_map = event.get(cluster_cands_map_token_); // clusters->candidates map
      const auto &clusters = event.get(clusters_token_); // cluster ID for each candidate

      // initialize output tensor
      const auto job_size = cluster_cands_map.const_view().offset().metadata().size() - 1; // number of elements to run the inference on, which is the number of clusters
      auto output_tensor = SoftTauOutputDeviceTensor(event.queue(), job_size);
      output_tensor.zeroInitialise(event.queue());

      // sort the clusters->candidates association map by pt
      auto cluster_cands_map_sorted = kernels::sortClustersCandsMap(event.queue(), pf, bx_clusters_map, cluster_cands_map, clusters);
      
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
          const auto effective_batch_size = (max_batch_size_ > 0u) ? max_batch_size_ : 1u;

          // loop over batches
          for (auto begin = 0u; begin < job_size; begin += effective_batch_size) {
            // check batch size for the current batch
            const auto this_batch = std::min<uint32_t>(effective_batch_size, job_size - begin);

            // copy the content of the global input tensor that corresponds to the current batch
            auto batch_input = kernels::copyInputChunk(event.queue(), input_tensor, begin, this_batch);

            // initialize the output tensor corresponding to the current input batch
            auto batch_output = SoftTauOutputDeviceTensor(event.queue(), this_batch);
            batch_output.zeroInitialise(event.queue());
            
            // create tensor collections for the current batch
            cms::torch::alpakatools::TensorCollection<Queue> inputs(this_batch);
            cms::torch::alpakatools::TensorCollection<Queue> outputs(this_batch);

            // records
            auto in = batch_input.view().records();
            inputs.add<SoftTauInputTensorSoA>("features", in.features());
            inputs.add<SoftTauInputTensorSoA>("pad_mask", in.pad_mask());

            auto out = batch_output.view().records();
            outputs.add<SoftTauOutputTensorSoA>("cls", out.cls());
            outputs.add<SoftTauOutputTensorSoA>("vz", out.vz());
            outputs.add<SoftTauOutputTensorSoA>("pt", out.pt());
            outputs.add<SoftTauOutputTensorSoA>("charge", out.charge());
            outputs.change_order({"cls", "vz", "pt", "charge"});

            // do the inference
            model_.forward(event.queue(), inputs, outputs);

            // copy results from the batch output to the global output
            kernels::copyOutputChunk(event.queue(), batch_output, output_tensor, begin, this_batch);
          }

          // wait
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
    // input bx -> clusters map 
    const device::EDGetToken<BxLookupDevice> bx_clusters_map_token_;
    // input clusters -> candidates map that must be sorted by pt
    const device::EDGetToken<AssociationMapDevice> cluster_cands_map_token_;
    // cluster ID for each candidate
    const device::EDGetToken<ClustersDeviceCollection> clusters_token_;
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