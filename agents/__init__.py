from agents.fql import FQLAgent
from agents.fql_iqn import FQLIQNAgent
from agents.floq import FloQAgent
from agents.bsc import BSCAgent


agents = dict(
    fql=FQLAgent,
    fqliqn = FQLIQNAgent,
    floq = FloQAgent,
    bsc=BSCAgent,
)