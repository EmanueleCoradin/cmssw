#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

GENERATE_SOA_LAYOUT(ThreeViewTestLayout, SOA_COLUMN(std::uint32_t, value))

using TestSoA = ThreeViewTestLayout<>;
using TestConstView = TestSoA::ConstView;
using TestMultiView = SoAConstMultiView<TestConstView, 3>;
using TestHostCollection = PortableHostCollection<TestSoA>;

namespace {

  void fill(TestHostCollection& collection, std::uint32_t base) {
    auto view = collection.view();
    for (cms::soa::size_type i = 0; i < view.metadata().size(); ++i) {
      view[i].value() = base + i;
    }
  }

}  // namespace

TEST_CASE("SoAConstMultiView maps three views correctly", "[SoAConstMultiView]") {
  constexpr std::array<cms::soa::size_type, 3> sizes{2, 3, 4};

  TestHostCollection first(cms::alpakatools::host(), sizes[0]);
  TestHostCollection second(cms::alpakatools::host(), sizes[1]);
  TestHostCollection third(cms::alpakatools::host(), sizes[2]);

  fill(first, 0);
  fill(second, 100);
  fill(third, 200);

  std::vector<std::reference_wrapper<TestHostCollection const>> collections;
  collections.push_back(first);
  collections.push_back(second);
  collections.push_back(third);

  TestMultiView multiView(collections, [](auto const& collection) { return collection.get().const_view(); });

  REQUIRE(multiView.numViews() == 3);
  REQUIRE(multiView.size() == sizes[0]+sizes[1]+sizes[2]);

  // Explicitly exercise the transitions between views.
  REQUIRE(multiView[0].value() == 0);
  REQUIRE(multiView[1].value() == 1);
  REQUIRE(multiView[2].value() == 100);  // first element of the second view
  REQUIRE(multiView[4].value() == 102);  // last element of the second view
  REQUIRE(multiView[5].value() == 200);  // first element of the third view
  REQUIRE(multiView[8].value() == 203);  // last element of the third view

  constexpr std::array<std::uint32_t, 9> expected{
      0, 1,
      100, 101, 102,
      200, 201, 202, 203};

  for (cms::soa::size_type i = 0; i < multiView.size(); ++i) {
    CAPTURE(i);
    REQUIRE(multiView[i].value() == expected[i]);
  }
}
