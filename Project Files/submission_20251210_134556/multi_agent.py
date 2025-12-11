"""
Multi-Agent Collaborative System - FIXED
Implements coordinated validation testing with specialized agents
"""

import numpy as np
from typing import Dict, List, Tuple
from agents.ucb_agent import UCBAgent
from agents.dqn_agent import DQNAgent


class SpecialistAgent:
    """Individual specialist agent with specific focus area"""

    def __init__(self, agent_id: int, specialization: str, base_agent, config: Dict):
        self.agent_id = agent_id
        self.specialization = specialization
        self.base_agent = base_agent
        self.config = config

        self.focus_categories = self._get_focus_categories()

        self.bugs_found = 0
        self.tests_conducted = 0
        self.shared_knowledge = []

    def _get_focus_categories(self) -> List[int]:
        """Define which categories this agent specializes in"""
        category_groups = {
            'security': [0, 1],  # adversarial, edge_case
            'correctness': [2, 3, 5],  # boundary, distribution_shift, logic_error
            'coverage': [6, 7, 8, 9]  # coverage_guided, random, metamorphic, stress_test
        }
        return category_groups.get(self.specialization, list(range(10)))

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action biased toward specialization"""
        if isinstance(self.base_agent, DQNAgent):
            action = self.base_agent.select_action(state, training=True)
        else:
            action = self.base_agent.select_action(state)

        category_idx = int(action[0])

        if category_idx not in self.focus_categories:
            if np.random.random() < 0.7:
                category_idx = np.random.choice(self.focus_categories)
                action[0] = float(category_idx)

        return action

    def update(self, action: np.ndarray, reward: float, bug_found: bool):
        """Update agent based on result"""
        self.tests_conducted += 1
        if bug_found:
            self.bugs_found += 1

        if isinstance(self.base_agent, DQNAgent):
            pass
        else:
            self.base_agent.update(action, reward)

    def share_knowledge(self, finding: Dict):
        """Share discovered knowledge with team"""
        self.shared_knowledge.append(finding)

    def get_performance_metrics(self) -> Dict:
        """Get agent-specific performance metrics"""
        return {
            'agent_id': self.agent_id,
            'specialization': self.specialization,
            'bugs_found': self.bugs_found,
            'tests_conducted': self.tests_conducted,
            'success_rate': self.bugs_found / max(1, self.tests_conducted),
            'knowledge_shared': len(self.shared_knowledge)
        }


class MultiAgentCoordinator:
    """Coordinates multiple specialist agents for collaborative testing"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.n_agents = config.get('multi_agent', {}).get('n_specialist_agents', 3)
        self.coordination_strategy = config.get('multi_agent', {}).get(
            'coordination_strategy', 'collaborative'
        )
        self.communication_interval = config.get('multi_agent', {}).get(
            'communication_interval', 5
        )
        self.knowledge_sharing = config.get('multi_agent', {}).get(
            'knowledge_sharing', True
        )

        self.agents = self._create_specialist_agents()

        self.knowledge_base = {
            'successful_tests': [],
            'bug_patterns': {},
            'coverage_gaps': []
        }

        self.steps_since_communication = 0
        self.total_collaborations = 0

    def _create_specialist_agents(self) -> List[SpecialistAgent]:
        """Create team of specialist agents"""
        specializations = ['security', 'correctness', 'coverage']
        agents = []

        for i in range(self.n_agents):
            spec = specializations[i % len(specializations)]
            base_agent = UCBAgent(10, self.config)
            specialist = SpecialistAgent(i, spec, base_agent, self.config)
            agents.append(specialist)

        return agents

    def select_active_agent(self, state: np.ndarray) -> int:
        """Select which agent should act next"""
        if self.coordination_strategy == 'round_robin':
            return self.steps_since_communication % self.n_agents

        elif self.coordination_strategy == 'performance_based':
            performances = [agent.bugs_found / max(1, agent.tests_conducted)
                          for agent in self.agents]
            exp_perf = np.exp(np.array(performances) * 5)
            probs = exp_perf / exp_perf.sum()
            return np.random.choice(self.n_agents, p=probs)

        else:  # collaborative - consider coverage
            coverage = state[:10]

            agent_scores = []
            for agent in self.agents:
                focus_coverage = np.mean([coverage[cat] for cat in agent.focus_categories])
                # Prefer agents with low coverage in their area
                agent_scores.append(1.0 - focus_coverage)

            scores = np.array(agent_scores)

            # FIXED: Handle case when all scores are zero or sum to zero
            if scores.sum() == 0 or np.any(np.isnan(scores)):
                # Fallback to uniform random selection
                return np.random.choice(self.n_agents)

            # Add small epsilon to avoid division by zero
            scores = scores + 1e-10
            probs = scores / scores.sum()

            # Double check for NaN
            if np.any(np.isnan(probs)):
                return np.random.choice(self.n_agents)

            return np.random.choice(self.n_agents, p=probs)

    def coordinate_action(self, state: np.ndarray) -> Tuple[np.ndarray, int]:
        """Coordinate team action"""
        active_idx = self.select_active_agent(state)
        active_agent = self.agents[active_idx]

        action = active_agent.select_action(state)

        self.steps_since_communication += 1
        if self.steps_since_communication >= self.communication_interval:
            self._communicate_knowledge()
            self.steps_since_communication = 0

        return action, active_idx

    def update_agents(self, active_idx: int, action: np.ndarray,
                     reward: float, bug_info: Dict, state: np.ndarray):
        """Update agents based on experience"""
        active_agent = self.agents[active_idx]
        bug_found = bug_info.get('has_bug', False)

        active_agent.update(action, reward, bug_found)

        if bug_found and self.knowledge_sharing:
            finding = {
                'agent_id': active_idx,
                'action': action.copy(),
                'bug_type': bug_info.get('bug_info', {}).get('type'),
                'severity': bug_info.get('bug_info', {}).get('severity'),
                'state': state.copy()
            }

            self.knowledge_base['successful_tests'].append(finding)

            for agent in self.agents:
                if agent.agent_id != active_idx:
                    agent.share_knowledge(finding)

            self.total_collaborations += 1

    def _communicate_knowledge(self):
        """Inter-agent communication session"""
        for agent in self.agents:
            if agent.shared_knowledge:
                for finding in agent.shared_knowledge[-5:]:
                    bug_type = finding.get('bug_type')
                    if bug_type:
                        if bug_type not in self.knowledge_base['bug_patterns']:
                            self.knowledge_base['bug_patterns'][bug_type] = []
                        self.knowledge_base['bug_patterns'][bug_type].append(finding)

    def get_team_statistics(self) -> Dict:
        """Get statistics for entire team"""
        agent_stats = [agent.get_performance_metrics() for agent in self.agents]

        total_bugs = sum(a['bugs_found'] for a in agent_stats)
        total_tests = sum(a['tests_conducted'] for a in agent_stats)

        return {
            'team_size': self.n_agents,
            'total_bugs_found': total_bugs,
            'total_tests_conducted': total_tests,
            'team_success_rate': total_bugs / max(1, total_tests),
            'collaborations': self.total_collaborations,
            'knowledge_base_size': len(self.knowledge_base['successful_tests']),
            'unique_bug_patterns': len(self.knowledge_base['bug_patterns']),
            'agent_statistics': agent_stats
        }

    def get_collaboration_bonus(self) -> float:
        """Calculate bonus reward for collaboration"""
        if not self.knowledge_sharing:
            return 0.0

        recent_collaborations = min(self.total_collaborations, 10)
        return recent_collaborations * self.config['rewards'].get('collaboration_bonus', 5)


class FallbackManager:
    """Manages fallback strategies when agents get stuck or perform poorly"""

    def __init__(self, config: Dict):
        self.config = config
        self.fallback_enabled = config.get('fallback', {}).get('enabled', True)
        self.strategies = config.get('fallback', {}).get('strategies', [])

        self.steps_without_bugs = 0
        self.fallback_activations = 0
        self.last_fallback_type = None

    def check_fallbacks(self, state: np.ndarray, recent_bugs: int,
                       steps: int) -> Tuple[bool, str, np.ndarray]:
        """Check if fallback strategy should be activated"""
        if not self.fallback_enabled:
            return False, None, None

        coverage = state[:10]
        avg_coverage = np.mean(coverage)

        for strategy in self.strategies:
            strategy_type = strategy.get('type')

            if strategy_type == 'coverage_fallback':
                trigger_coverage = strategy.get('trigger_coverage', 0.3)
                if avg_coverage < trigger_coverage:
                    low_coverage_categories = np.where(coverage < 0.3)[0]
                    if len(low_coverage_categories) > 0:
                        category = np.random.choice(low_coverage_categories)
                        action = self._generate_exploration_action(category)
                        self.fallback_activations += 1
                        self.last_fallback_type = strategy_type
                        return True, strategy_type, action

            elif strategy_type == 'stagnation_fallback':
                trigger_steps = strategy.get('trigger_steps', 50)
                if recent_bugs == 0 and steps > trigger_steps:
                    self.steps_without_bugs = steps
                    action = self._generate_random_exploration()
                    self.fallback_activations += 1
                    self.last_fallback_type = strategy_type
                    return True, strategy_type, action

            elif strategy_type == 'diversity_fallback':
                min_categories = strategy.get('min_categories', 5)
                categories_used = np.sum(coverage > 0.1)
                if categories_used < min_categories:
                    unused = np.where(coverage < 0.1)[0]
                    if len(unused) > 0:
                        category = np.random.choice(unused)
                        action = self._generate_exploration_action(category)
                        self.fallback_activations += 1
                        self.last_fallback_type = strategy_type
                        return True, strategy_type, action

        return False, None, None

    def _generate_exploration_action(self, category: int) -> np.ndarray:
        """Generate action for specific category"""
        return np.array([
            float(category),
            np.random.uniform(0.5, 1.0),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])

    def _generate_random_exploration(self) -> np.ndarray:
        """Generate completely random exploratory action"""
        return np.array([
            float(np.random.randint(0, 10)),
            np.random.uniform(0, 1),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])

    def reset_stagnation_counter(self):
        """Reset counter when bug is found"""
        self.steps_without_bugs = 0

    def get_statistics(self) -> Dict:
        """Get fallback statistics"""
        return {
            'total_activations': self.fallback_activations,
            'last_fallback_type': self.last_fallback_type,
            'steps_without_bugs': self.steps_without_bugs
        }