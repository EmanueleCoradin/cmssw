#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"
#include "DataFormats/PortableTestObjects/interface/TorchTestHostCollection.h"

#include "DataFormats/PortableTestObjects/interface/ParticleHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ImageHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/LogitsHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MultiHeadNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MaskHostCollection.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::TestHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::TestHostCollection2);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::TestHostCollection3);
SET_PORTABLEHOSTOBJECT_READ_RULES(portabletest::TestHostObject);

SET_PORTABLEHOSTCOLLECTION_READ_RULES(torchportabletest::ParticleHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(torchportabletest::SimpleNetHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(torchportabletest::MultiHeadNetHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(torchportabletest::ImageHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(torchportabletest::LogitsHostCollection);
