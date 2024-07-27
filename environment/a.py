import torch as T
import torch.nn.functional as F


def learn(self, memory):
    if not memory.ready():
        return

    actor_state, state, action, reward, actor_new_state, state_, done = memory.sample_buffer()

    state = T.tensor(state, dtype=T.float)
    action = T.tensor(action, dtype=T.float)
    reward = T.tensor(reward)
    state_ = T.tensor(state_, dtype=T.float)
    done = T.tensor(done)

    agents_actions = []

    actions_for_value_update = []
    log_probs_for_value_update = []
    for agent_idx, agent in enumerate(self.agents):
        a_state = T.tensor(actor_state[agent_idx], dtype=T.float)
        v_action, v_log_probs = agent.actor.sample_normal(
            a_state, reparameterize=False)
        v_log_probs = v_log_probs.view(-1)

        actions_for_value_update.append(v_action)
        log_probs_for_value_update.append(v_log_probs)

    actions_for_value_update = T.hstack((actions_for_value_update[:]))

    for agent_idx, agent in enumerate(self.agents):
        agents_actions.append(action[agent_idx])

        value = agent.value(state).view(-1)

        q1_new_policy = agent.critic1.forward(
            state=state, action=actions_for_value_update)
        q2_new_policy = agent.critic2.forward(
            state=state, action=actions_for_value_update)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        agent.value.optimizer.zero_grad()
        value_target = - \
            log_probs_for_value_update[agent_idx].to(T.float32) + critic_value
        value_loss = 0.5 * F.mse_loss(value, value_target.detach())

        value_loss.backward(retain_graph=True)
        agent.value.optimizer.step()

        agent.update_network_parameters()

    actions = T.cat([acts for acts in agents_actions], dim=1)

    actions_for_actor_update = []
    log_probs_for_actor_update = []

    for agent_idx, agent in enumerate(self.agents):

        a_state = T.tensor(actor_state[agent_idx], dtype=T.float)
        a_action, a_log_probs = agent.actor.sample_normal(
            a_state.clone(), reparameterize=True)
        a_log_probs = a_log_probs.view(-1)
        actions_for_actor_update.append(a_action)
        # actions_for_actor_update.append(a_action)
        log_probs_for_actor_update.append(a_log_probs)

    actions_for_actor_update = (
        T.hstack((actions_for_actor_update[:]))).float()

    for agent_idx, agent in enumerate(self.agents):
        # print(agent_idx)
        q1_new_policy = agent.critic1.forward(
            state=state, action=actions_for_actor_update)
        q2_new_policy = agent.critic2.forward(
            state=state, action=actions_for_actor_update)
        critic_value = T.min(q1_new_policy.clone(), q2_new_policy.clone())
        critic_value = critic_value.view(-1).clone()

        actor_loss = log_probs_for_actor_update[agent_idx].to(
            T.float32).clone() - critic_value
        actor_loss = T.mean(actor_loss)
        agent.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        agent.actor.optimizer.step()

        value_ = agent.target_value(state_).view(-1)
        value_[done[:, 0]] = 0.0

        agent.critic1.optimizer.zero_grad()
        agent.critic2.optimizer.zero_grad()
        q_hat = 1.0 * reward[:, agent_idx].to(T.float32) + agent.gamma * value_

        q1_old_policy = agent.critic1.forward(
            state=state, action=actions).view(-1)
        q2_old_policy = agent.critic2.forward(
            state=state, action=actions).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        agent.critic1.optimizer.step()
        agent.critic2.optimizer.step()
