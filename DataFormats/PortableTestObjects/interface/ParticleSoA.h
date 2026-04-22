#ifndef DataFormats_PortableTestObjects_interface_ParticleSoA_h
#define DataFormats_PortableTestObjects_interface_ParticleSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaMath/interface/float16_t.h"

namespace portabletest {

  GENERATE_SOA_LAYOUT(ParticleLayout, SOA_COLUMN(float, pt), SOA_COLUMN(float, eta), SOA_COLUMN(float, phi))
  GENERATE_SOA_LAYOUT(ParticleLayoutFP16,
                      SOA_COLUMN(cms::float16_t, pt),
                      SOA_COLUMN(cms::float16_t, eta),
                      SOA_COLUMN(cms::float16_t, phi))
  using ParticleSoA = ParticleLayout<>;
  using ParticleFP16SoA = ParticleLayoutFP16<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_ParticleSoA_h
