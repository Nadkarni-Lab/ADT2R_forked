import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


#############################################
# MaskedCausalAttention with adaptive multi-head
#############################################

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, device, n_tokens):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.max_T = max_T  # Now using the passed max_T (which may be 2*T or 3*T)
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.n_tokens = n_tokens

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)
        self.proj_nets = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(n_heads)])

        # The pooling kernel is set based on n_tokens and head dimension
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, (self.h_dim // self.n_heads) * self.n_tokens))
        self.mse_none = nn.MSELoss(reduction="none")

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        self.register_buffer('mask', mask)

    def orthogonal_constraint(self, x):
        B, D1, D2 = x.shape
        normalized_x = F.normalize(x, p=2, dim=-1)
        factor_transpose = normalized_x.transpose(2, 1)
        multiplied = torch.bmm(normalized_x, factor_transpose)
        identity = torch.eye(D1, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        mse = self.mse_none(input=multiplied, target=identity)
        return torch.norm(mse, p="fro", dim=[-2, -1]).mean()

    def forward(self, x, sindices=None):
        B, T, H = x.shape
        N, D = self.n_heads, H // self.n_heads
        # Compute queries, keys, and values
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)  # [B, N, T, D]
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # Scaled dot-product attention
        weights = q @ k.transpose(2, 3) / math.sqrt(D)  # [B, N, T, T]
        # Create mask based on actual sequence length T
        current_mask = self.mask[..., :T, :T]
        weights = weights.masked_fill(current_mask == 0, float('-inf'))
        normalized_weights = F.softmax(weights, dim=-1)
        attention_ = self.att_drop(normalized_weights @ v)  # [B, N, T, D]

        # Compute regularization loss for diverse attention heads
        loss_reg = self.orthogonal_constraint(attention_.reshape(B, N, -1))

        # Transpose to [B, T, N, D]
        attention = attention_.transpose(1, 2)

        # Compute cluster indices using average pooling over each head’s attention
        if sindices is None:
            # Reshape based on actual T rather than self.max_T
            cluster_idx_ = self.avg_pool(attention_.reshape(B, N, int(T / self.n_tokens), -1).transpose(1, 2))
            # Apply softmax and take argmax along the pooled dimension
            cluster_idx_ = torch.argmax(F.softmax(cluster_idx_.squeeze(dim=-1), dim=-1),
                                        dim=-1)  # shape: [B, T/self.n_tokens]
            # Repeat cluster indices if necessary
            cluster_idx = cluster_idx_.unsqueeze(dim=2).repeat(1, 1, self.n_tokens).reshape(B, -1)
        else:
            cluster_idx_ = sindices
            cluster_idx = sindices.unsqueeze(dim=2).repeat(1, 1, self.n_tokens).reshape(B, -1)

        # Flatten the attention output for indexing:
        # attention is [B, T, N*D] → reshape to [B*T, N*D]
        attention = attention.contiguous().view(B, T, N * D)
        outs = [self.proj_nets[i](attention) for i in range(self.n_heads)]
        outs = torch.stack(outs, dim=-1)  # [B, T, N*D, n_heads]
        outs = outs.view(-1, outs.shape[2], outs.shape[-1])  # flatten batch and time → [B*T, N*D, n_heads]

        # Here, cluster_idx is expected to have shape [B, T] (if n_tokens=1)
        cidx_flat = cluster_idx.reshape(-1)  # [B*T]
        # Now index: for each row in outs, select the column given by cidx_flat
        final_outs = outs[torch.arange(outs.shape[0]), :, cidx_flat]
        final_outs = final_outs.view(B, T, -1)

        return loss_reg, final_outs, cluster_idx


#############################################
# Block: One transformer block with adaptive attention and FFN.
#############################################

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, device, n_tokens):
        super().__init__()
        self.n_heads = n_heads
        self.h_dim = h_dim
        self.max_t = max_T  # This max_T should match the expected sequence length (e.g. 2*T if interleaved)
        self.drop_p = drop_p
        self.device = device
        self.n_tokens = n_tokens

        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p, device, n_tokens)
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        ) for _ in range(n_heads)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_heads)])
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x, sindices=None):
        B, T, H = x.shape
        loss_reg, attended, cidx = self.attention(x, sindices)
        x = x + attended
        x = [self.ln1[i](x) for i in range(self.n_heads)]
        x = [x[i] + self.mlp[i](x[i]) for i in range(self.n_heads)]
        x_ = torch.stack(x, dim=-1)  # [B, T, H, n_heads]
        x_ = x_.view(-1, x_.shape[2], x_.shape[-1])  # [B*T, H, n_heads]

        # For interleaved tokens, assume cidx shape is [B, T] (if n_tokens==1)
        cidx_flat = cidx.reshape(-1)  # [B*T]
        final = x_[torch.arange(x_.shape[0]), :, cidx_flat]
        final = final.view(B, T, -1)
        x = self.ln2(final)
        return x, loss_reg, cidx


#############################################
# ADT2R: Main model architecture
#############################################

class ADT2R(nn.Module):
    def __init__(self, state_dim, act_dim, h_dim, n_heads, drop_p, max_t, device,
                 lr, w_decay, lr_decay, lr_step, lam_actor, lam_critic, lam_reg, gamma, tau, n_tokens=1):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim  # should be set to d_e (e.g. 60)
        self.n_heads = n_heads  # e.g. 4
        self.drop_p = drop_p
        self.max_t = max_t  # this is T (number of timesteps)
        self.lr = lr
        self.w_decay = w_decay
        self.lr_decay = lr_decay
        self.lr_step = lr_step
        self.lam_actor = lam_actor
        self.lam_critic = lam_critic
        self.lam_reg = lam_reg
        self.gamma = gamma
        self.tau = tau
        self.n_tokens = n_tokens

        # Embedding modules:
        self.embed_timestep = nn.Embedding(self.max_t, h_dim)
        self.embed_state = nn.Linear(self.state_dim, h_dim)
        self.embed_action = nn.Embedding(self.act_dim, h_dim)
        self.embed_mortality = nn.Linear(1, h_dim)
        self.embed_estimated_state = nn.Linear(1, h_dim)
        self.embed_hiddens_low = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout()
        )

        # Critic and actor networks:
        self.critic = nn.Linear(h_dim, self.act_dim)
        self.critic_target = deepcopy(self.critic)
        self.actor = nn.Linear(h_dim, self.act_dim)
        self.actor_target = deepcopy(self.actor)

        # Projection layer for interleaving tokens:
        self.projection_layer = nn.Linear(2 * h_dim, h_dim)

        # Blocks: we use interleaving, so each timestep produces 2 tokens.
        # Therefore, the value estimation branch expects sequence length 2*T.
        self.ve_adt = Block(h_dim, self.max_t * 2, n_heads, drop_p, device, n_tokens)
        # The treatment recommendation branch may use a longer horizon, e.g., 3*T.
        self.tr_adt = Block(h_dim, self.max_t * 3, n_heads, drop_p, device, n_tokens)

        self.embed_ln_tr = nn.LayerNorm(h_dim)
        self.embed_ln_high = nn.LayerNorm(h_dim)
        self.policy = nn.Linear(h_dim, act_dim)
        # Transformer branches for the actor and high-level treatment recommendation.
        self.transformer = Block(h_dim, self.max_t * 2, n_heads, drop_p, device, n_tokens)
        self.transformer_high = Block(h_dim, self.max_t * 3, n_heads, drop_p, device, n_tokens)

        # Optimizers
        self.optimiser_actor = torch.optim.RAdam(
            list(self.embed_timestep.parameters()) +
            list(self.embed_state.parameters()) + list(self.embed_action.parameters()) +
            list(self.ve_adt.parameters()) + list(self.actor.parameters()),
            lr=self.lr, weight_decay=self.w_decay
        )
        self.optimiser_critic = torch.optim.RAdam(
            list(self.embed_timestep.parameters()) +
            list(self.embed_state.parameters()) + list(self.embed_action.parameters()) +
            list(self.ve_adt.parameters()) + list(self.critic.parameters()),
            lr=self.lr, weight_decay=self.w_decay
        )
        self.optimiser_all = torch.optim.RAdam(
            list(self.embed_timestep.parameters()) +
            list(self.embed_state.parameters()) + list(self.embed_action.parameters()) +
            list(self.tr_adt.parameters()) + list(self.embed_mortality.parameters()) +
            list(self.embed_estimated_state.parameters()) + list(self.policy.parameters()),
            lr=self.lr, weight_decay=self.w_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser_all, gamma=self.lr_decay,
                                                         step_size=self.lr_step)
        self.ce_none = nn.CrossEntropyLoss(reduction="none")

        # Define a predict_action method using the policy.
        self.predict_action = lambda h: self.policy(h)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def forward(self, records, is_train=True):
        # Get inputs
        timesteps = records["sequences"].type(torch.LongTensor).to(self.device)
        states = records["observation"]
        actions = records["action"]
        demos = records["demo"]
        B, T, _ = states.shape

        # Get additional features (e.g., SOFA scores) and concatenate with states
        S24 = records["sofa_24hours"]
        SALL = S24.unsqueeze(dim=2) / 4.0
        states = torch.cat((states, SALL, demos), dim=-1)

        # ---- Token Embedding Module with Interleaving ----
        time_emb = self.embed_timestep(timesteps)  # [B, T, h_dim]
        state_emb = self.embed_state(states) + time_emb  # [B, T, h_dim]
        action_emb = self.embed_action(actions) + time_emb  # [B, T, h_dim]
        # Interleave state and action tokens:
        sa_stack = torch.stack((state_emb, action_emb), dim=1)  # [B, 2, T, h_dim]
        sa_stack = sa_stack.permute(0, 2, 1, 3)  # [B, T, 2, h_dim]
        interleaved = sa_stack.reshape(B, 2 * T, self.h_dim)  # [B, 2*T, h_dim]
        token_embeddings = self.embed_ln(interleaved)  # [B, 2*T, h_dim]

        # ---- Value Estimation Module ----
        h_ve, loss_reg_ve, cidx_ve = self.ve_adt(token_embeddings)
        # Reshape back to separate modalities:
        h_ve = h_ve.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)  # [B, 2, T, h_dim]
        h_state = h_ve[:, 0]  # Use the state branch [B, T, h_dim]
        v_hat = self.critic(h_state)
        v_size = list(v_hat.size())
        v_hat_flat = v_hat.view(-1, self.act_dim)
        actions_flat = actions.view(-1)
        v_hat_at = v_hat_flat[torch.arange(v_hat_flat.size(0)), actions_flat].view(v_size)

        v_hat_next = self.critic_target(h_state)
        v_hat_next = torch.cat((v_hat_next[:, 1:], torch.zeros(size=(B, 1, v_size[-1]), device=self.device)), dim=1)
        v_hat_next = v_hat_next.view(-1, self.act_dim)

        a_hat_target = torch.argmax(F.softmax(self.actor_target(h_state), dim=-1), dim=-1)
        a_hat_target_next = torch.cat((a_hat_target[:, 1:], torch.zeros(size=(B, 1), device=self.device)), dim=1).view(
            -1)
        v_hat_at_next = v_hat_next[torch.arange(v_hat_next.size(0)), a_hat_target_next].view(v_size)

        R_T = records["reward"]
        next_val = R_T + self.gamma * v_hat_at_next
        td = v_hat_at - next_val
        loss_critic = self.lam_critic * (td ** 2 * records["seq_mask"]).sum() / records["seq_mask"].sum()
        if is_train:
            self.optimiser_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.optimiser_critic.step()

        # ---- Actor Branch ---- (using interleaved tokens)
        # Recompute token embeddings for actor branch using interleaving:
        sa_stack_actor = torch.stack((state_emb, action_emb), dim=1)  # [B, 2, T, h_dim]
        sa_stack_actor = sa_stack_actor.permute(0, 2, 1, 3)
        interleaved_actor = sa_stack_actor.reshape(B, 2 * T, self.h_dim)
        actor_tokens = self.embed_ln(interleaved_actor)  # [B, 2*T, h_dim]
        h_actor, loss_reg_actor, cidx_actor = self.transformer(actor_tokens)
        # Reshape back to [B, T, 2, h_dim] and select the state branch:
        h_actor = h_actor.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)[:, 0]
        actor_logit = self.actor(h_actor)
        actor_prob = F.softmax(actor_logit, dim=-1)
        log_prob = torch.log(actor_prob + 1e-5)
        v_hat_at_actor = v_hat_at  # from earlier critic computation
        loss_actor = -self.lam_actor * (
                    log_prob.view(-1, self.act_dim)[torch.arange(B * T), actions.view(-1)] * v_hat_at_actor.view(
                -1)).sum() / records["seq_mask"].sum()
        if is_train:
            self.optimiser_actor.zero_grad()
            loss_actor.backward(retain_graph=True)
            self.optimiser_actor.step()
            ADT2R.soft_update(self.critic, self.critic_target, self.tau)
            ADT2R.soft_update(self.actor, self.actor_target, self.tau)

        # ---- Treatment Recommendation Module ----
        # For high-level treatment recommendation, we may use a longer sequence.
        # Here, we interleave tokens as before.
        sa_stack_tr = torch.stack((state_emb, action_emb), dim=1)
        sa_stack_tr = sa_stack_tr.permute(0, 2, 1, 3)
        interleaved_tr = sa_stack_tr.reshape(B, 2 * T, self.h_dim)
        tr_tokens = self.embed_ln(interleaved_tr)
        h_tr, loss_reg_tr, cidx_tr = self.transformer_high(tr_tokens)
        h_tr = h_tr.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)[:, 0]

        goal_embeddings = self.embed_mortality(records["outcome"][:, -1].unsqueeze(dim=1)).unsqueeze(dim=1)
        zero_value = torch.zeros(size=(B, 1), device=self.device)
        v_hats = torch.cat((zero_value, v_hat_at[:, :-1]), dim=1)
        v_hats_embeddings = self.embed_estimated_state(v_hats.unsqueeze(dim=2).detach()) + time_emb
        goal_action_embeddings = torch.cat((goal_embeddings, action_emb[:, :-1, :]), dim=1)
        gap_state = torch.stack((goal_action_embeddings, v_hats_embeddings, state_emb), dim=1)
        gap_state = gap_state.permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        high_tokens = self.embed_ln_high(gap_state)
        h_high, loss_reg_high, cidx_high = self.transformer_high(high_tokens, sindices=cidx)
        h_high = h_high.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)[:, 2]
        action_logits = self.predict_action(h_high)
        action_probs = F.softmax(action_logits, dim=-1)
        action_preds = torch.argmax(action_probs, dim=-1)
        loss_action = self.ce_none(action_logits.view(-1, self.act_dim), actions.view(-1))
        loss_action = loss_action.sum() / records["seq_mask"].view(-1).sum()
        reg_loss = self.lam_reg * (loss_reg_tr + loss_reg_high)
        action_reg_loss = loss_action + reg_loss
        if is_train:
            self.optimiser_all.zero_grad()
            action_reg_loss.backward()
            self.optimiser_all.step()

        return loss_action, reg_loss, loss_actor, loss_critic, action_probs, action_preds
