import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List
import json

class SpecialTransformer(nn.Module):
    """
    A specialized transformer model for tabular data with context.

    Configurable via a JSON configuration file.

    Args:
        config_path (str): Path to the JSON configuration file.
    """
    def __init__(self, config_path: str):
        super().__init__()

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.feature_dim = config['feature_dim']
        self.context_dim = config['context_dim']
        embedding_dim = config['embedding_dim']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        ffn_dim = config['ffn_dim']
        dropout = config['dropout']
        output_dim = config['output_dim']

        self.feature_embedding = nn.Linear(self.feature_dim, embedding_dim)
        # No separate context embedding here, as it's already encoded

        # Transformer Encoder for features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True  # Important: Input is (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Context-aware aggregation (simple concatenation + linear layer)
        self.context_integration = nn.Linear(embedding_dim, self.context_dim)
        
        # RMSnorm
        self.rmsNorm = nn.LayerNorm(self.context_dim)

        # Output layer
        self.output_layer = nn.Linear(self.context_dim, output_dim)

        self.num_heads = num_heads  # Store for attention analysis

    def forward(self, features: torch.Tensor, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            features: Input features (batch_size, num_features, feature_dim).
            context_embedding: Encoded context (batch_size, context_dim).

        Returns:
            torch.Tensor: Predicted output (batch_size, output_dim).
        """

        # 1. Embed features
        embedded_features = self.feature_embedding(features)  # (batch_size, num_features, embedding_dim)

        # 2. Transformer Encoder (on features only)
        encoded_features = self.transformer_encoder(embedded_features)  # (batch_size, num_features, embedding_dim)

        # 3. Integrate context (concatenation)
        integrated_representation = torch.cat([encoded_features, context_embedding], dim=1)  # (batch_size, embedding_dim + context_dim)
        integrated_representation = F.relu(self.context_integration(integrated_representation))  # (batch_size, embedding_dim)
        
        # 4. Aggregate feature representations (e.g., mean pooling)
        aggregated_features = torch.mean(integrated_representation, dim=1)  # (batch_size, embedding_dim)
        norm_agg_features = self.rmsNorm(aggregated_features)

        # 5. Output layer
        output = self.output_layer(norm_agg_features)  # (batch_size, output_dim)

        return output

    def get_attention_weights(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Gets the attention weights from all transformer encoder layers.
        """
        embedded_features = self.feature_embedding(features)
        attention_weights = []
        
        # Manually iterate through the encoder layers
        x = embedded_features
        for layer in self.transformer_encoder.layers:
            x, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attention_weights.append(attn_weights)
            x = layer.dropout1(x)
            x = layer.norm1(x + layer.dropout1(x))
            x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
        return attention_weights


# --- Example Usage ---
if __name__ == '__main__':
    # --- Create dummy config file---
    config_data = {
        "feature_dim": 5,
        "context_dim": 64,
        "embedding_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 256,
        "dropout": 0.1,
        "output_dim": 1
    }

    with open("config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    # Load the model from the configuration file
    config_path = "config.json"
    model = SpecialTransformer(config_path)

    # Example data dimensions (replace with your actual values after preprocessing)
    batch_size = 32
    num_features = 10 # Number of feature
    # Create dummy input data
    features = torch.randn(batch_size, num_features, model.feature_dim)  # Example features
    context_embedding = torch.randn(batch_size, model.context_dim)    # Example encoded context

    # Forward pass
    output = model(features, context_embedding)
    print("Output shape:", output.shape)

    # Get attention weights
    attention_weights = model.get_attention_weights(features, context_embedding)
    print("Number of attention weight tensors:", len(attention_weights))
    print("Shape of first layer's attention weights:", attention_weights[0].shape)