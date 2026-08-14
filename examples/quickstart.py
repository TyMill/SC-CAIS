from sc_cais.agents import AgenticHeuristicAgent
from sc_cais.assurance import RuntimeAssurance
from sc_cais.environment import SupplyChainEnv
from sc_cais.evaluation import summarize
from sc_cais.network import create_reference_network
from sc_cais.scenarios import default_scenarios

scenario = next(s for s in default_scenarios() if s.name == "cyber_physical")
env = SupplyChainEnv(create_reference_network(), seed=42, disruptions=scenario.disruptions)
results = env.run(AgenticHeuristicAgent(), RuntimeAssurance(), horizon=24)
print(summarize(results))
