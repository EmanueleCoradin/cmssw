# import l1scouting options
from L1TriggerScouting.Phase2.options_cff import options, VarParsing

# extra options
options.register ("splitFactor", 
    1, 
    VarParsing.VarParsing.multiplicity.singleton, 
    VarParsing.VarParsing.varType.int, 
    "Number of sub-streams in which a single data stream is divided into."
)

options.register ("pipelineStep", 
    "tagging",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Step up to which run the pipeline (unpacking, clustering, sorting, reshaping, tagging)."
)

options.register ("model", 
    "L1TriggerScouting/TauTagging/data/softtauid_sigmoid_col.pt", 
    VarParsing.VarParsing.multiplicity.singleton, 
    VarParsing.VarParsing.varType.string,
    "PyTorch model to be used."
)

options.register ("batchSize", 
    8192, 
    VarParsing.VarParsing.multiplicity.singleton, 
    VarParsing.VarParsing.varType.int, 
    "Batch size (in terms of clusters) for model inference."
)

options.register ("dump", 
    [],
    VarParsing.VarParsing.multiplicity.list,
    VarParsing.VarParsing.varType.string,
    "List of objects to dump (candidates, cluster_indexes, soft_tau_inputs, soft_tau_outputs), compatible with the selected pipelineStep."
)

options.register ("synchronize", 
    False, 
    VarParsing.VarParsing.multiplicity.singleton, 
    VarParsing.VarParsing.varType.bool, 
    "Force synchronization after each module, must be used when benchmarking"
)

options.register ("reportEvery",
    10, 
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int, 
    "Print message after the specified number of events (proper events in this case)"
)