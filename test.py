class ExchangeResidualConnectMLPWithAlpha(nn.Module):
    def __init__(
            self,
            sent_dim,
            concept_dim,
            hidden_size,
            num_layers,
            dropout):
        super().__init__()

        self.mlp = MLP(
            sent_dim +
            concept_dim,
            hidden_size,
            sent_dim +
            concept_dim,
            num_layers,
            dropout)
        self.exchange = Exchange(sent_dim, concept_dim)
        self.alpha = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, inp):
        return self.alpha * \
            self.exchange(inp) + (1 - self.alpha) * self.mlp(inp)
