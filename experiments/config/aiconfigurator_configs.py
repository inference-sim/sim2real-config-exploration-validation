from experiments.config.topology import Topology, enumerate_topologies


def generate_aiconfigurator_configs(
    max_gpus: int = 8,
    topologies: list[Topology] | None = None,
) -> list[dict]:
    if topologies is None:
        topologies = enumerate_topologies(max_gpus)
    configs = []
    for topo in topologies:
        configs.append({
            "tp": topo.tp,
            "pp": topo.pp,
            "replicas": topo.replicas,
            "routing_policy": None,
        })
    return configs
