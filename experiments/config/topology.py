from dataclasses import dataclass

TP_VALUES = (1, 2, 4, 8)
PP_VALUES = (1, 2, 4)


@dataclass(frozen=True)
class Topology:
    tp: int
    pp: int
    replicas: int

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.replicas

    @property
    def cost_per_hour(self) -> float:
        return self.total_gpus * 3.20

    @property
    def is_multi_replica(self) -> bool:
        return self.replicas > 1


@dataclass(frozen=True)
class TopologyNoPP:
    tp: int
    replicas: int

    @property
    def total_gpus(self) -> int:
        return self.tp * self.replicas

    @property
    def cost_per_hour(self) -> float:
        return self.total_gpus * 3.20

    @property
    def is_multi_replica(self) -> bool:
        return self.replicas > 1


@dataclass(frozen=True)
class TopologyDP:
    tp: int
    pp: int
    dp: int

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.dp

    @property
    def cost_per_hour(self) -> float:
        return self.total_gpus * 3.20

    @property
    def is_multi_dp(self) -> bool:
        return self.dp > 1


def enumerate_topologies(max_gpus: int = 8) -> list[Topology]:
    result = []
    for tp in TP_VALUES:
        for pp in PP_VALUES:
            if tp * pp > max_gpus:
                continue
            max_replicas = max_gpus // (tp * pp)
            for r in range(1, max_replicas + 1):
                result.append(Topology(tp=tp, pp=pp, replicas=r))
    return result


def enumerate_topologies_no_pp(max_gpus: int = 8) -> list[TopologyNoPP]:
    result = []
    for tp in TP_VALUES:
        if tp > max_gpus:
            continue
        max_replicas = max_gpus // tp
        for r in range(1, max_replicas + 1):
            result.append(TopologyNoPP(tp=tp, replicas=r))
    return result


def enumerate_topologies_dp(max_gpus: int = 8) -> list[TopologyDP]:
    result = []
    for tp in TP_VALUES:
        for pp in PP_VALUES:
            if tp * pp > max_gpus:
                continue
            max_dp = max_gpus // (tp * pp)
            for dp in range(1, max_dp + 1):
                result.append(TopologyDP(tp=tp, pp=pp, dp=dp))
    return result
