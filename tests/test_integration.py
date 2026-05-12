from experiments.config.blis_configs import generate_blis_configs
from experiments.config.llmservingsim_configs import generate_llmservingsim_configs
from experiments.config.aiconfigurator_configs import generate_aiconfigurator_configs
from experiments.config.vidur_configs import generate_vidur_configs
from experiments.config.llm_optimizer_configs import generate_llm_optimizer_configs


def test_all_tools_generate_configs():
    blis = generate_blis_configs()
    llmsim = generate_llmservingsim_configs()
    aicfg = generate_aiconfigurator_configs()
    vidur = generate_vidur_configs()
    llmopt = generate_llm_optimizer_configs()

    assert len(blis) > 0
    assert len(llmsim) > 0
    assert len(aicfg) > 0
    assert len(vidur) > 0
    assert len(llmopt) > 0


def test_config_counts_summary():
    """Verify input config counts match spec topology/parameter expectations.

    Note: AIConfigurator returns 25 input topology triples; the ~250 output
    points come from its internal batch/ctx_tokens sweep at runtime.
    """
    counts = {
        "inference-sim": len(generate_blis_configs()),
        "LLMServingSim": len(generate_llmservingsim_configs()),
        "AIConfigurator": len(generate_aiconfigurator_configs()),
        "Vidur": len(generate_vidur_configs()),
        "llm-optimizer": len(generate_llm_optimizer_configs()),
    }
    assert 90_000 <= counts["inference-sim"] <= 110_000
    assert 12_000 <= counts["LLMServingSim"] <= 16_000
    assert counts["AIConfigurator"] == 25
    assert 3_000 <= counts["Vidur"] <= 4_500
    assert 10_000 <= counts["llm-optimizer"] <= 13_000
